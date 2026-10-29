"""CPU-only source/options/prefix/RNG guards; no NPU acceptance claim."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import multiprocessing as mp
import os
import sys
import tempfile
import time
import unittest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import optimized_journal_inference as entry
from optimized_journal_inference import (OPTIMIZED_SOURCE_SHA, SOURCE_SHA, optimization_contract,
                                        claim_attempt_output, Guard)
from rank_journal import RankJournal, sha
from test_rank_journal import RNG, row


def delayed_claim_worker(output, rank, claim, queue):
    """Stall the actual rank0 claim writer while all 15 peers enter its reader."""
    if rank == 0:
        def delayed(path, raw):
            with path.open('xb') as stream:
                middle = len(raw) // 2
                stream.write(raw[:middle])
                stream.flush()
                os.fsync(stream.fileno())
                time.sleep(.3)
                stream.write(raw[middle:])
                stream.flush()
                os.fsync(stream.fileno())

        entry.write_file = delayed
    try:
        claim_attempt_output(Path(output), claim, rank,
                             Guard(datetime.now(timezone.utc) + timedelta(seconds=20), None))
        queue.put((rank, 'ok'))
    except BaseException as error:
        queue.put((rank, type(error).__name__, str(error)))


class OptimizedJournalContractTests(unittest.TestCase):
    def test_delayed_rank0_claim_and_15_early_peers(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'fresh_attempt'
            claim = dict(attempt_id='a'*32, contract_sha256='b'*64)
            context = mp.get_context('fork' if 'fork' in mp.get_all_start_methods() else 'spawn')
            queue = context.Queue()
            processes = [context.Process(target=delayed_claim_worker, args=(str(output), rank, claim, queue))
                         for rank in range(16)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(25)
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, 0)
            outcomes = [queue.get(timeout=2) for _ in processes]
            self.assertTrue(all(row[1] == 'ok' for row in outcomes), outcomes)
            guard = Guard(datetime.now(timezone.utc) + timedelta(seconds=5), None)
            with self.assertRaisesRegex(RuntimeError, 'another launch'):
                claim_attempt_output(output, dict(claim, attempt_id='c'*32), 1, guard)
            with self.assertRaises(FileExistsError):
                claim_attempt_output(output, claim, 0, guard)

    def test_allowlisted_sources_and_explicit_options(self):
        self.assertEqual(optimization_contract(SOURCE_SHA, SOURCE_SHA, 'none', 2048, 'reference')['mode'], 'none')
        for mode in ('point-parallel', 'point-parallel-last-frame'):
            for chunk in (2048, 4096, 8192):
                with self.subTest(mode=mode, chunk=chunk):
                    self.assertEqual(optimization_contract(OPTIMIZED_SOURCE_SHA, OPTIMIZED_SOURCE_SHA,
                                                          mode, chunk, 'cuda-compatible')['query_chunk'], chunk)

    def test_wrong_source_legacy_optimization_and_invalid_options_rejected(self):
        cases = [('x'*64, 'x'*64, 'none', 2048, 'reference'),
                 (OPTIMIZED_SOURCE_SHA, SOURCE_SHA, 'point-parallel', 2048, 'cuda-compatible'),
                 (SOURCE_SHA, SOURCE_SHA, 'point-parallel', 2048, 'cuda-compatible'),
                 (OPTIMIZED_SOURCE_SHA, OPTIMIZED_SOURCE_SHA, 'none', 8192, 'cuda-compatible'),
                 (OPTIMIZED_SOURCE_SHA, OPTIMIZED_SOURCE_SHA, 'point-parallel', 8192, 'reference'),
                 (OPTIMIZED_SOURCE_SHA, OPTIMIZED_SOURCE_SHA, 'unknown', 2048, 'cuda-compatible'),
                 (OPTIMIZED_SOURCE_SHA, OPTIMIZED_SOURCE_SHA, 'point-parallel', 16384, 'cuda-compatible')]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                optimization_contract(*values)

    def test_changed_mode_chunk_source_or_RNG_preserve_saved_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            tokens = ['t%d' % index for index in range(22)]
            rng = RNG()
            state = rng.state()
            options = optimization_contract(OPTIMIZED_SOURCE_SHA, OPTIMIZED_SOURCE_SHA,
                                            'point-parallel-last-frame', 8192, 'cuda-compatible')
            contract = dict(world_size=16, source_sha256=OPTIMIZED_SOURCE_SHA, inference_optimization=options)
            with RankJournal(root, contract, tokens, 16, 0, 'fresh') as journal:
                journal.initial_rng(state)
                journal.append(0, row('t0'), state)
            prefix = root / 'rank_00' / 'steps' / '000000'
            before = {p.name: sha(p) for p in prefix.iterdir()}
            for changed in (dict(options, mode='point-parallel'), dict(options, query_chunk=4096)):
                with self.assertRaisesRegex(RuntimeError, 'contract changed'):
                    RankJournal(root, dict(contract, inference_optimization=changed), tokens, 16, 0, 'resume')
            with self.assertRaisesRegex(RuntimeError, 'contract changed'):
                RankJournal(root, dict(contract, source_sha256=SOURCE_SHA), tokens, 16, 0, 'resume')
            with RankJournal(root, contract, tokens, 16, 0, 'resume') as journal:
                with self.assertRaisesRegex(RuntimeError, 'differs exactly'):
                    journal.append(0, row('t0'), dict(state, cpu=b'wrong-but-nonempty'))
                with self.assertRaisesRegex(RuntimeError, 'differs exactly'):
                    journal.append(0, row('t0', 2.), state)
                self.assertEqual(journal.append(0, row('t0'), state), 'replayed_exactly_no_overwrite')
            self.assertEqual(before, {p.name: sha(p) for p in prefix.iterdir()})
            prefix.rename(prefix.with_name('000001'))
            with self.assertRaisesRegex(RuntimeError, 'hole'):
                RankJournal(root, contract, tokens, 16, 0, 'resume')


if __name__ == '__main__':
    unittest.main(verbosity=2)
