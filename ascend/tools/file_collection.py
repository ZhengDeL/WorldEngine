"""Bounded shared-filesystem result collection; no device communication.

The host creates one fresh directory per logical collective, then gives every
rank the same run ID and contract hash. Pickle input is trusted output from that
owned run, not an exchange format for untrusted files. No payload is synthesized.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import time
import uuid


class CollectionError(RuntimeError):
    pass


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _hash_string(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def sampler_indices(size, world_size, rank):
    """Original non-shuffled DistributedSampler: pad cyclically, then stride."""
    if any(type(x) is not int for x in (size, world_size, rank)):
        raise CollectionError("size, world_size and rank must be integers")
    if size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise CollectionError("invalid size/world_size/rank")
    count = math.ceil(size / world_size)
    return [(rank + j * world_size) % size for j in range(count)]


def _validate_contract(contract):
    if not isinstance(contract, dict) or set(contract) != {
            "schema", "run_id", "world_size", "size", "source_sha256",
            "ordering", "sample_field", "expected_ids"}:
        raise CollectionError("invalid contract fields")
    if contract["schema"] != 1 or contract["ordering"] != "mmcv-zip-interleave-truncate":
        raise CollectionError("unsupported contract")
    run_id = contract["run_id"]
    if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id) is None:
        raise CollectionError("invalid run_id")
    sampler_indices(contract["size"], contract["world_size"], 0)
    if not _hash_string(contract["source_sha256"]):
        raise CollectionError("source_sha256 must identify the actual run source")
    field, ids = contract["sample_field"], contract["expected_ids"]
    if field is None:
        if ids is not None:
            raise CollectionError("expected_ids requires sample_field")
    elif (not isinstance(field, str) or not field or not isinstance(ids, list)
          or len(ids) != contract["size"] or any(not isinstance(x, str) or not x for x in ids)):
        raise CollectionError("invalid ordered sample identity contract")


def _atomic_new(path, content):
    """Publish a complete file without overwriting an earlier producer."""
    path = Path(path)
    temp = path.with_name(path.name + ".tmp." + str(os.getpid()) + "." + uuid.uuid4().hex)
    with temp.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        # A same-filesystem hard link publishes the completed inode atomically,
        # and fails if another producer has already used the final path.
        os.link(temp, path)
    except FileExistsError as exc:
        raise CollectionError("refusing to overwrite " + path.name) from exc
    finally:
        temp.unlink()


def create_collection(root, *, run_id, world_size, size, source_sha256,
                      sample_field=None, expected_ids=None):
    """Host-only initialization before starting ranks; an existing root is rejected."""
    contract = dict(schema=1, run_id=run_id, world_size=world_size, size=size,
                    source_sha256=source_sha256,
                    ordering="mmcv-zip-interleave-truncate", sample_field=sample_field,
                    expected_ids=expected_ids)
    _validate_contract(contract)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    raw = _json(contract)
    _atomic_new(root / "contract.json", raw)
    return _sha(raw)


def _load_contract(root, run_id, contract_sha256, size, world_size):
    path = root / "contract.json"
    if path.is_symlink() or not path.is_file():
        raise CollectionError("missing or nonregular contract")
    raw = path.read_bytes()
    if not _hash_string(contract_sha256) or _sha(raw) != contract_sha256:
        raise CollectionError("host contract SHA mismatch")
    try:
        contract = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise CollectionError("invalid contract JSON") from exc
    _validate_contract(contract)
    if (contract["run_id"], contract["world_size"], contract["size"]) != (run_id, world_size, size):
        raise CollectionError("run/rank-count/size differs from host contract")
    return contract


def _guard(deadline_utc, stop_file, guard_callback=None):
    if guard_callback is not None:
        guard_callback()
    try:
        deadline = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise CollectionError("deadline must be an explicit UTC timestamp") from exc
    if deadline.tzinfo is None or deadline.utcoffset().total_seconds() != 0:
        raise CollectionError("deadline must be an explicit UTC timestamp")
    if stop_file is not None and Path(stop_file).exists():
        raise CollectionError("STOP requested")
    if datetime.now(timezone.utc) >= deadline:
        raise CollectionError("collection deadline reached; missing ranks are not filled")


def _validate_part(part, contract, rank):
    indices = sampler_indices(contract["size"], contract["world_size"], rank)
    if not isinstance(part, list) or len(part) != len(indices):
        raise CollectionError("rank part row count differs from original sampler")
    field = contract["sample_field"]
    if field is not None:
        expected = contract["expected_ids"]
        for position, (row, index) in enumerate(zip(part, indices)):
            if not isinstance(row, dict) or row.get(field) != expected[index]:
                raise CollectionError("rank sample order mismatch at position " + str(position))
    return indices


def _paths(root, rank):
    return root / ("part_%06d.pkl" % rank), root / ("complete_%06d.json" % rank)


def publish_part(result_part, size, *, rank, world_size, root, run_id,
                 contract_sha256, deadline_utc, stop_file=None, guard_callback=None):
    """Persist this rank immediately; completion is published after data+SHA."""
    root = Path(root)
    _guard(deadline_utc, stop_file, guard_callback)
    contract = _load_contract(root, run_id, contract_sha256, size, world_size)
    indices = _validate_part(result_part, contract, rank)
    part_path, manifest_path = _paths(root, rank)
    payload = pickle.dumps(result_part, protocol=4)
    manifest = dict(schema=1, run_id=run_id, contract_sha256=contract_sha256,
                    source_sha256=contract["source_sha256"], rank=rank,
                    world_size=world_size, size=size, rows=len(result_part),
                    indices_sha256=_sha(_json(indices)), file=part_path.name,
                    bytes=len(payload), sha256=_sha(payload))
    _guard(deadline_utc, stop_file, guard_callback)
    _atomic_new(part_path, payload)
    _guard(deadline_utc, stop_file, guard_callback)
    _atomic_new(manifest_path, _json(manifest))
    return manifest


def _read_rank(root, contract, contract_sha256, rank):
    part_path, manifest_path = _paths(root, rank)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise CollectionError("nonregular completion manifest")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise CollectionError("incomplete or invalid completion manifest") from exc
    indices = sampler_indices(contract["size"], contract["world_size"], rank)
    expected = dict(schema=1, run_id=contract["run_id"], contract_sha256=contract_sha256,
                    source_sha256=contract["source_sha256"], rank=rank,
                    world_size=contract["world_size"], size=contract["size"],
                    rows=len(indices), indices_sha256=_sha(_json(indices)), file=part_path.name)
    if not isinstance(manifest, dict) or set(manifest) != set(expected) | {"bytes", "sha256"}:
        raise CollectionError("completion manifest fields mismatch")
    if any(manifest[key] != value for key, value in expected.items()):
        raise CollectionError("completion manifest run/source/rank/order mismatch")
    if not _hash_string(manifest["sha256"]) or type(manifest["bytes"]) is not int or manifest["bytes"] < 1:
        raise CollectionError("invalid part hash or byte count")
    if part_path.is_symlink() or not part_path.is_file():
        raise CollectionError("completed part is missing or nonregular")
    raw = part_path.read_bytes()
    if len(raw) != manifest["bytes"] or _sha(raw) != manifest["sha256"]:
        raise CollectionError("part bytes/SHA mismatch")
    try:
        part = pickle.loads(raw)
    except Exception as exc:
        raise CollectionError("completed part cannot be decoded") from exc
    _validate_part(part, contract, rank)
    return part, manifest


def wait_and_merge(size, *, world_size, root, run_id, contract_sha256,
                   deadline_utc, stop_file=None, poll_seconds=0.1, guard_callback=None):
    """Rank 0 only: require all completed ranks, then retain original MMCV order."""
    if not isinstance(poll_seconds, (int, float)) or not 0 < poll_seconds <= 1:
        raise CollectionError("poll_seconds must be in (0,1]")
    root = Path(root)
    contract = _load_contract(root, run_id, contract_sha256, size, world_size)
    expected_files = {_paths(root, r)[1].name for r in range(world_size)}
    while True:
        _guard(deadline_utc, stop_file, guard_callback)
        present = {path.name for path in root.glob("complete_*.json")}
        if present - expected_files:
            raise CollectionError("unexpected rank completion file")
        # Validate already completed parts immediately; corrupt output is not
        # treated as a late rank and is never silently skipped.
        parts = {}
        manifests = {}
        for rank in range(world_size):
            if _paths(root, rank)[1].name in present:
                parts[rank], manifests[rank] = _read_rank(root, contract, contract_sha256, rank)
        if len(parts) == world_size:
            break
        time.sleep(poll_seconds)
    _guard(deadline_utc, stop_file, guard_callback)
    # Exactly the original collect_results_cpu merge: zip rank parts then [:size].
    merged = [item for row in zip(*(parts[r] for r in range(world_size))) for item in row][:size]
    if len(merged) != size:
        raise CollectionError("incomplete merged result")
    if contract["sample_field"] is not None:
        if [row[contract["sample_field"]] for row in merged] != contract["expected_ids"]:
            raise CollectionError("merged sample order differs from full dataset")
    receipt = dict(schema=1, run_id=run_id, contract_sha256=contract_sha256,
                   source_sha256=contract["source_sha256"], world_size=world_size,
                   size=size, padded_rows=sum(len(x) for x in parts.values()),
                   rank_parts=[manifests[r] for r in range(world_size)],
                   merge="zip-rank-interleave-then-truncate", all_parts_retained=True)
    _atomic_new(root / "merge_complete.json", _json(receipt))
    return merged


def collect_results_file(result_part, size, *, rank, world_size, root, run_id,
                         contract_sha256, deadline_utc, stop_file=None, guard_callback=None):
    publish_part(result_part, size, rank=rank, world_size=world_size, root=root,
                 run_id=run_id, contract_sha256=contract_sha256,
                 deadline_utc=deadline_utc, stop_file=stop_file, guard_callback=guard_callback)
    if rank != 0:
        return None
    return wait_and_merge(size, world_size=world_size, root=root, run_id=run_id,
                          contract_sha256=contract_sha256, deadline_utc=deadline_utc,
                          stop_file=stop_file, guard_callback=guard_callback)
