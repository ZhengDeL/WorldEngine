import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import file_collection as fc


class FileCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "fresh"
        self.size, self.world = 7, 3
        self.ids = ["token_%d" % x for x in range(self.size)]
        self.sha = fc.create_collection(self.root, run_id="run_1", world_size=self.world,
                                        size=self.size, source_sha256="a" * 64,
                                        sample_field="token", expected_ids=self.ids)

    def kwargs(self, **changes):
        result = dict(size=self.size, world_size=self.world, root=self.root,
                      run_id="run_1", contract_sha256=self.sha,
                      deadline_utc=(datetime.now(timezone.utc) + timedelta(seconds=3)).isoformat())
        result.update(changes)
        return result

    def part(self, rank):
        return [dict(token=self.ids[i], ordinal=i, signed_zero=-0.0)
                for i in fc.sampler_indices(self.size, self.world, rank)]

    def publish(self, rank, part=None, **changes):
        return fc.publish_part(self.part(rank) if part is None else part,
                               rank=rank, **self.kwargs(**changes))

    def all_parts(self):
        for rank in range(self.world):
            self.publish(rank)

    def corrupt_manifest(self, rank, **changes):
        path = self.root / ("complete_%06d.json" % rank)
        data = json.loads(path.read_text())
        data.update(changes)
        path.write_text(json.dumps(data))

    def test_original_padded_interleave_order(self):
        self.all_parts()
        result = fc.wait_and_merge(**self.kwargs())
        self.assertEqual([row["ordinal"] for row in result], list(range(self.size)))
        self.assertEqual(len(list(self.root.glob("part_*.pkl"))), self.world)
        self.assertEqual(len(list(self.root.glob("complete_*.json"))), self.world)
        self.assertTrue((self.root / "merge_complete.json").is_file())
        self.assertEqual(pickle.dumps(result[0]), pickle.dumps(self.part(0)[0]))

    def test_real_rank_zero_waits_for_delayed_peers(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = pool.submit(fc.collect_results_file, self.part(0), rank=0, **self.kwargs())
            self.assertIsNone(fc.collect_results_file(self.part(2), rank=2, **self.kwargs()))
            self.assertIsNone(fc.collect_results_file(self.part(1), rank=1, **self.kwargs()))
            self.assertEqual([x["token"] for x in pending.result(timeout=4)], self.ids)

    def test_rank_report_dicts_use_separate_collective(self):
        root = Path(self.temp.name) / "reports"
        sha = fc.create_collection(root, run_id="report_1", world_size=3, size=3,
                                   source_sha256="b" * 64)
        args = self.kwargs(root=root, run_id="report_1", size=3, contract_sha256=sha)
        for rank in range(3):
            fc.publish_part([{"rank_report": rank}], rank=rank, **args)
        self.assertEqual(fc.wait_and_merge(**args), [{"rank_report": r} for r in range(3)])

    def test_padding_when_world_exceeds_size(self):
        self.assertEqual([fc.sampler_indices(2, 5, r) for r in range(5)], [[0], [1], [0], [1], [0]])

    def test_navtest_rank1_missing_unique_count(self):
        self.assertEqual(len(fc.sampler_indices(12146, 16, 1)), 760)
        others = {i for r in range(16) if r != 1 for i in fc.sampler_indices(12146, 16, r)}
        missing = set(range(12146)) - others
        self.assertEqual(len(others), 11387)
        self.assertEqual(len(missing), 759)
        self.assertEqual(set(fc.sampler_indices(12146, 16, 1)) & others, {1})

    def test_short_part_rejected_before_publication(self):
        with self.assertRaisesRegex(fc.CollectionError, "row count"):
            self.publish(1, self.part(1)[:-1])
        self.assertFalse(list(self.root.glob("complete_*.json")))

    def test_wrong_local_token_order_rejected(self):
        with self.assertRaisesRegex(fc.CollectionError, "sample order"):
            self.publish(1, list(reversed(self.part(1))))

    def test_wrong_host_source_hash_rejected(self):
        with self.assertRaisesRegex(fc.CollectionError, "contract SHA"):
            self.publish(1, contract_sha256="b" * 64)

    def test_wrong_run_world_size_or_rank_rejected(self):
        for changes in ({"run_id": "old_run"}, {"world_size": 2}, {"size": 6}):
            with self.subTest(changes=changes), self.assertRaises(fc.CollectionError):
                self.publish(1, **changes)
        with self.assertRaises(fc.CollectionError):
            self.publish(3)

    def test_rank_cannot_overwrite_saved_part(self):
        self.publish(1)
        original = (self.root / "part_000001.pkl").read_bytes()
        with self.assertRaisesRegex(fc.CollectionError, "overwrite"):
            self.publish(1)
        self.assertEqual((self.root / "part_000001.pkl").read_bytes(), original)

    def test_truncated_part_sha_rejected(self):
        self.all_parts()
        path = self.root / "part_000001.pkl"
        path.write_bytes(path.read_bytes()[:-8])
        with self.assertRaisesRegex(fc.CollectionError, "bytes/SHA"):
            fc.wait_and_merge(**self.kwargs())

    def test_manifest_partial_write_rejected(self):
        self.all_parts()
        (self.root / "complete_000001.json").write_bytes(b'{"schema":')
        with self.assertRaisesRegex(fc.CollectionError, "invalid completion"):
            fc.wait_and_merge(**self.kwargs())

    def test_completed_manifest_wrong_source_rejected(self):
        self.all_parts()
        self.corrupt_manifest(1, source_sha256="b" * 64)
        with self.assertRaisesRegex(fc.CollectionError, "run/source/rank/order"):
            fc.wait_and_merge(**self.kwargs())

    def test_completed_manifest_wrong_rank_count_rejected(self):
        self.all_parts()
        self.corrupt_manifest(1, world_size=99)
        with self.assertRaisesRegex(fc.CollectionError, "run/source/rank/order"):
            fc.wait_and_merge(**self.kwargs())

    def test_completed_payload_wrong_order_even_with_valid_sha_rejected(self):
        self.all_parts()
        path = self.root / "part_000001.pkl"
        raw = pickle.dumps(list(reversed(self.part(1))), protocol=4)
        path.write_bytes(raw)
        self.corrupt_manifest(1, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        with self.assertRaisesRegex(fc.CollectionError, "sample order"):
            fc.wait_and_merge(**self.kwargs())

    def test_absent_rank_not_filled_from_reference_or_short_zip(self):
        self.publish(0)
        self.publish(2)
        deadline = (datetime.now(timezone.utc) + timedelta(milliseconds=30)).isoformat()
        with self.assertRaisesRegex(fc.CollectionError, "missing ranks are not filled"):
            fc.wait_and_merge(**self.kwargs(deadline_utc=deadline), poll_seconds=0.01)
        self.assertFalse((self.root / "merge_complete.json").exists())

    def test_part_without_completion_remains_incomplete(self):
        self.publish(0)
        (self.root / "part_000001.pkl").write_bytes(pickle.dumps(self.part(1)))
        self.publish(2)
        deadline = (datetime.now(timezone.utc) + timedelta(milliseconds=30)).isoformat()
        with self.assertRaisesRegex(fc.CollectionError, "deadline"):
            fc.wait_and_merge(**self.kwargs(deadline_utc=deadline), poll_seconds=0.01)
        self.assertTrue((self.root / "part_000001.pkl").exists())

    def test_stop_blocks_publish_and_merge(self):
        stop = Path(self.temp.name) / "STOP"
        stop.touch()
        with self.assertRaisesRegex(fc.CollectionError, "STOP"):
            self.publish(1, stop_file=stop)
        with self.assertRaisesRegex(fc.CollectionError, "STOP"):
            fc.wait_and_merge(**self.kwargs(stop_file=stop))

    def test_expired_or_naive_deadline_rejected(self):
        for deadline in ("2020-01-01T00:00:00Z", "2030-01-01T00:00:00", "invalid"):
            with self.subTest(deadline=deadline), self.assertRaises(fc.CollectionError):
                self.publish(1, deadline_utc=deadline)

    def test_extra_rank_completion_rejected(self):
        self.all_parts()
        (self.root / "complete_000003.json").write_text("{}")
        with self.assertRaisesRegex(fc.CollectionError, "unexpected rank"):
            fc.wait_and_merge(**self.kwargs())

    def test_existing_root_and_second_merge_rejected(self):
        with self.assertRaises(FileExistsError):
            fc.create_collection(self.root, run_id="reuse", size=3, world_size=3,
                                 source_sha256="a" * 64)
        self.all_parts()
        fc.wait_and_merge(**self.kwargs())
        with self.assertRaisesRegex(fc.CollectionError, "overwrite"):
            fc.wait_and_merge(**self.kwargs())

    def test_module_imports_are_stdlib_only(self):
        tree = ast.parse(Path(fc.__file__).read_text())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(x.name.split(".")[0] for x in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module.split(".")[0])
        self.assertFalse(set(imported) & {"torch", "torch_npu", "numpy", "mmcv"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
