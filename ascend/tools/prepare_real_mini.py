"""Prepare a genuine NAVTEST subset for ascend/split_mini.py, without downloads.

Run on the data server using its trusted official pickle files. Images are copied
by default; symlink mode requires the original camera root at the same absolute
path in every later container. No model, accelerator or label generation runs.
Selected driving logs are disjoint, but only selected anchors and their exact
three preceding frames are retained: this is not the complete official benchmark.
"""

import argparse
import copy
import hashlib
import json
import math
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import yaml
from PIL import Image


METRICS = (
    "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
    "time_to_collision_within_bound", "comfort", "score",
)
SCOPE = (
    "Genuine NAVTEST subset for functional adaptation training and independent "
    "small-data evaluation, split by driving log; not full NAVTRAIN/NAVTEST "
    "training, convergence, distribution or official benchmark acceptance"
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def safe_relative(raw):
    raw = str(raw)
    if "\\" in raw or "\0" in raw:
        raise ValueError("Camera paths must be POSIX paths without NUL")
    p = PurePosixPath(raw)
    if p.is_absolute() or not p.parts or any(x in ("..", "") for x in p.parts):
        raise ValueError("Unsafe camera relative path: " + raw)
    if ":" in p.parts[0]:
        raise ValueError("Drive-qualified camera path is not supported")
    return p


def resolve_camera(raw, root):
    """Map only exact relative paths or an explicit sensor_blobs/{test,trainval} prefix."""
    text = str(raw)
    if "\\" in text or "\0" in text or ".." in PurePosixPath(text).parts:
        raise ValueError("Unsafe original camera path: " + text)
    original = Path(text)
    if PurePosixPath(text).is_absolute() or original.is_absolute():
        try:
            rel = safe_relative(original.relative_to(root).as_posix())
            mapping = "absolute_under_supplied_camera_root"
        except ValueError:
            rel = None
    else:
        rel = safe_relative(text)
        mapping = "relative_to_supplied_camera_root"
    # Official annotations may retain the original asset-root prefix. A suffix
    # is used only after this explicit directory pair, never a basename search.
    if rel is None or not (root / rel).is_file():
        parts = PurePosixPath(text).parts
        markers = [i for i in range(len(parts) - 2)
                   if parts[i] == "sensor_blobs" and parts[i + 1] in ("test", "trainval")]
        if len(markers) != 1:
            raise FileNotFoundError("Cannot resolve exact camera path below --cameras: " + text)
        rel = safe_relative(PurePosixPath(*parts[markers[0] + 2:]).as_posix())
        mapping = "explicit_sensor_blobs_partition_suffix"
    candidate = root / Path(rel.as_posix())
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(root)  # Reject a symlink escaping the supplied root.
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError("Missing or empty camera file: " + str(candidate))
    return resolved, rel.as_posix(), mapping


def check_rewards(row):
    if not isinstance(row, dict):
        raise ValueError("PDM token row must be a dictionary")
    for metric in METRICS:
        a = row.get(metric)
        if not isinstance(a, np.ndarray) or a.shape != (8192,):
            raise ValueError(metric + " must retain its original ndarray[8192]")
        if a.dtype.kind != "f" or not np.isfinite(a).all():
            raise ValueError(metric + " must contain only finite floating point values")


def prepare(args):
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("--output must not exist, including an empty directory")
    if args.logs < 3 or not 2 <= args.anchors_per_log <= 1000:
        raise ValueError("Need >=3 logs and 2..1000 anchors per log to retain rare and same-log normal anchors")
    for name in ("annotations", "pdm", "rare_split"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    camera_root = args.cameras.resolve(strict=True)
    if not camera_root.is_dir():
        raise ValueError("--cameras must be a directory")
    inputs = {name: file_record(getattr(args, name))
              for name in ("annotations", "pdm", "rare_split")}
    rare_filter = yaml.safe_load(args.rare_split.read_text(encoding="utf-8"))
    if not isinstance(rare_filter, dict) or not isinstance(rare_filter.get("tokens"), list):
        raise ValueError("--rare-split must contain an original tokens list")
    rare = set(rare_filter["tokens"])
    if not rare:
        raise ValueError("--rare-split has no tokens")
    with args.annotations.open("rb") as f:
        annotations = pickle.load(f)
    infos = annotations.get("infos") if isinstance(annotations, dict) else annotations
    if not isinstance(infos, list) or not infos:
        raise ValueError("Annotations must be a nonempty list or a dictionary with infos")
    with args.pdm.open("rb") as f:
        pdm = pickle.load(f)
    if not isinstance(pdm, dict) or not pdm:
        raise ValueError("PDM must be a nonempty original token dictionary")
    tokens = [x.get("token") if isinstance(x, dict) else None for x in infos]
    if any(not isinstance(t, str) or not t for t in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("Annotation tokens must be unique nonempty strings")
    candidates = defaultdict(list)
    rejected = Counter()
    camera_cache = {}
    mapping_counts = Counter()
    reward_cache = {}
    window_by_anchor = {}

    def frame_cameras(frame):
        cams = frame.get("cams")
        if not isinstance(cams, dict) or len(cams) != 8:
            raise ValueError("Expected all eight original camera entries")
        rows = []
        for camera_name, camera in cams.items():
            if not isinstance(camera, dict) or "data_path" not in camera:
                raise ValueError("Camera entry is missing data_path")
            raw = str(camera["data_path"])
            if raw not in camera_cache:
                source, relative, mapping = resolve_camera(raw, camera_root)
                # Actual JPEG decode validates the selected image, without a
                # model pipeline or an accelerator import.
                with Image.open(source) as image:
                    if image.format != "JPEG" or min(image.size) <= 0:
                        raise ValueError("Camera input must be a nonempty JPEG")
                    image.load()
                camera_cache[raw] = (source, relative, mapping)
            rows.append((camera_name, camera_cache[raw]))
        if len({row[1][1] for row in rows}) != 8:
            raise ValueError("Eight cameras must have eight distinct original paths")
        if len({row[1][0] for row in rows}) != 8:
            raise ValueError("Eight cameras must resolve to eight distinct original files")
        return rows

    potential = defaultdict(list)
    original_rare_logs = set()
    for index, frame in enumerate(infos):
        log = frame.get("log_name")
        if not isinstance(log, str) or not log:
            continue
        if frame["token"] in rare:
            original_rare_logs.add(log)
        if index >= 3 and frame["token"] in pdm:
            potential[log].append(index)
    selected_logs = []
    rare_logs = []
    normal_logs = []
    # Only inspect enough candidate logs/anchors for this subset. This does not
    # scan or decode all NAVTEST JPEGs. For a rare log, rare anchors are tried
    # first, so the bounded selection cannot accidentally omit its rare anchor.
    for rare_phase in (True, False):
        for log in sorted(potential):
            if (log in original_rare_logs) != rare_phase:
                continue
            order = potential[log]
            if rare_phase:
                order = ([i for i in order if infos[i]["token"] in rare] +
                         [i for i in order if infos[i]["token"] not in rare])
            for index in order:
                frame = infos[index]
                token = frame["token"]
                # Reserve at least one slot for a genuine same-log normal
                # anchor. The original fine-tune loader retrieves normal data
                # only from the failed anchor's own log, not the other logs.
                if rare_phase and token in rare and sum(infos[i]["token"] in rare
                        for i in candidates[log]) >= args.anchors_per_log - 1:
                    continue
                window = infos[index - 3:index + 1]
                try:
                    scene = frame["log_token"]
                    if not isinstance(scene, str) or not scene:
                        raise ValueError("Missing driving log identity")
                    if any(x.get("log_name") != log or x.get("log_token") != scene for x in window):
                        raise ValueError("History crosses original driving log/scene")
                    times = [float(x["timestamp"]) for x in window]
                    if not all(math.isfinite(t) for t in times) or not all(a < b for a, b in zip(times, times[1:])):
                        raise ValueError("History timestamps must be strictly increasing")
                    if token not in reward_cache:
                        check_rewards(pdm[token])
                        reward_cache[token] = True
                    for each in window:
                        frame_cameras(each)
                except (KeyError, TypeError, ValueError, FileNotFoundError, OSError) as error:
                    rejected[type(error).__name__ + ": " + str(error).split(":")[0]] += 1
                    continue
                candidates[log].append(index)
                window_by_anchor[token] = list(range(index - 3, index + 1))
                has_rare = any(infos[i]["token"] in rare for i in candidates[log])
                has_normal = any(infos[i]["token"] not in rare for i in candidates[log])
                if len(candidates[log]) >= args.anchors_per_log and (not rare_phase or (has_rare and has_normal)):
                    break
            if not candidates[log] or (rare_phase and not (
                    any(infos[i]["token"] in rare for i in candidates[log]) and
                    any(infos[i]["token"] not in rare for i in candidates[log]))):
                continue
            candidates[log].sort()
            (rare_logs if rare_phase else normal_logs).append(log)
            if rare_phase or len(normal_logs) >= args.logs - 1:
                break
        if rare_phase and not rare_logs:
            break
    if not rare_logs or len(normal_logs) < args.logs - 1:
        raise ValueError("Need one genuine rare log with a valid same-log non-rare anchor and >=%d separate non-rare logs with valid "
                         "4-frame/8-camera/PDM anchors; found rare=%d nonrare=%d; rejected=%s" %
                         (args.logs - 1, len(rare_logs), len(normal_logs), dict(rejected)))
    selected_logs = [rare_logs[0]] + normal_logs[:args.logs - 1]
    anchor_indices = []
    for log in selected_logs:
        available = candidates[log]
        chosen = available[:args.anchors_per_log]
        if log == rare_logs[0] and not any(infos[i]["token"] in rare for i in chosen):
            first_rare = next(i for i in available if infos[i]["token"] in rare)
            chosen = sorted(set(chosen[:args.anchors_per_log - 1] + [first_rare]))
        anchor_indices.extend(chosen)
    selected_rare = [infos[i]["token"] for i in anchor_indices if infos[i]["token"] in rare]
    same_log_normal = {rare_logs[0]: [infos[i]["token"] for i in anchor_indices
                       if infos[i]["log_name"] == rare_logs[0] and infos[i]["token"] not in rare]}
    if not selected_rare or not same_log_normal[rare_logs[0]]:
        raise ValueError("Selected rare anchors need at least one genuine same-log normal anchor")
    anchors = [infos[i]["token"] for i in sorted(anchor_indices)]
    anchor_set = set(anchors)
    retained_indices = sorted({i for t in anchors for i in window_by_anchor[t]})
    # Retain original global order. Check exact original predecessor identities
    # after compaction, which is essential because the original loader uses
    # consecutive array positions rather than searching for history timestamps.
    retained = [copy.deepcopy(infos[i]) for i in retained_indices]
    compact_positions = {x["token"]: i for i, x in enumerate(retained)}
    for token in anchors:
        position = compact_positions[token]
        actual = [x["token"] for x in retained[position - 3:position + 1]]
        expected = [infos[i]["token"] for i in window_by_anchor[token]]
        if position < 3 or actual != expected:
            raise ValueError("Compaction changed an anchor's exact original history")
    # Validate all inputs before creating a fresh output. A later I/O failure
    # intentionally leaves a partial directory; it must never be silently reused.
    out = args.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    if args.camera_mode == "copy":
        (out / "cameras").mkdir()
    else:
        # A directory link is recognized by the unchanged run_container.py,
        # which mounts its resolved target read-only. Per-file absolute links
        # would otherwise escape that launcher's automatic camera-root bind.
        (out / "cameras").symlink_to(camera_root, target_is_directory=True)
    image_records = {}
    for frame in retained:
        for camera_name, (source, relative, mapping) in frame_cameras(frame):
            target = out / "cameras" / relative
            if relative not in image_records:
                digest_before = sha256(source)
                if args.camera_mode == "copy":
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                if sha256(target) != digest_before or sha256(source) != digest_before:
                    raise ValueError("Original/copied JPEG bytes changed")
                image_records[relative] = {"bytes": source.stat().st_size,
                                           "sha256": digest_before, "mapping": mapping}
                mapping_counts[mapping] += 1
            frame["cams"][camera_name]["data_path"] = relative
    payload = copy.copy(annotations) if isinstance(annotations, dict) else {"infos": []}
    payload["infos"] = retained
    with (out / "worldengine_navtest_smoke.pkl").open("xb") as f:
        pickle.dump(payload, f, protocol=4)
    with (out / "worldengine_navtest_smoke_pdm.pkl").open("xb") as f:
        pickle.dump({token: pdm[token] for token in anchors}, f, protocol=4)
    scene_filter = copy.deepcopy(rare_filter)
    scene_filter.pop("scenario_tokens", None)
    scene_filter.update(tokens=anchors, log_names=sorted(selected_logs))
    (out / "worldengine_navtest_smoke.yaml").write_text(
        yaml.safe_dump(scene_filter, sort_keys=False), encoding="utf-8")
    # Original FineTune.load_annotations matches rare tokens against every
    # retained info, including history-only frames outside scene_filter. Supply
    # only the selected legal rare anchors so those extra history frames are
    # never accidentally sampled with missing/different original predecessors.
    mini_rare_filter = copy.deepcopy(rare_filter)
    mini_rare_filter.pop("scenario_tokens", None)
    mini_rare_filter.update(tokens=selected_rare, log_names=[rare_logs[0]])
    (out / "worldengine_navtest_smoke_rare.yaml").write_text(
        yaml.safe_dump(mini_rare_filter, sort_keys=False), encoding="utf-8")
    # Full hashes establish the original PKL/YAML inputs were unchanged.
    after = {name: file_record(getattr(args, name)) for name in inputs}
    if after != inputs:
        raise ValueError("An original annotation, PDM or rare-split input changed")
    manifest = {
        "result": "prepared_real_subset_not_training_acceptance",
        "dataset_scope": SCOPE, "trajectory_labels_changed": False,
        "camera_data_path_changed": True, "camera_paths_relative": True,
        "portable_camera_paths": args.camera_mode == "copy",
        "camera_mode": args.camera_mode, "camera_root": str(camera_root),
        "symlink_requires_original_root_same_absolute_mount": args.camera_mode == "symlink",
        "source_inputs": inputs, "source_inputs_unchanged": True,
        "official_provenance_not_verified_by_filename": True,
        "selected_logs": sorted(selected_logs), "logs": len(selected_logs),
        "anchors": len(anchors), "frames": len(retained),
        "max_anchors_per_log": args.anchors_per_log,
        "anchors_per_log": dict(Counter(infos[i]["log_name"] for i in anchor_indices)),
        "retained_original_indices": retained_indices,
        "anchor_history_tokens": {t: [infos[i]["token"] for i in window_by_anchor[t]] for t in anchors},
        "exact_original_history_verified": True,
        "rare_tokens": sorted(anchor_set & rare),
        "same_log_normal_tokens": same_log_normal,
        "same_log_normal_pool_verified": True,
        "mini_rare_filter": "worldengine_navtest_smoke_rare.yaml",
        "mini_finetuning_requires_selected_rare_filter": True,
        "original_rare_filter_used_to_define_normal": True,
        "history_only_original_rare_tokens": sorted({x["token"] for x in retained}
                                                   & rare - anchor_set),
        "expected_heldout_log": sorted(normal_logs[:args.logs - 1])[-1],
        "split_rule": "Existing split_mini chooses last sorted selected non-rare log as eval",
        "camera_files": len(image_records), "jpeg_decode_verified": True,
        "camera_path_mapping_counts": dict(mapping_counts),
        "camera_files_sha256": image_records,
        "pdm_metrics": list(METRICS), "pdm_candidates": 8192,
        "pdm_original_rows_preserved": True, "pdm_finite_checked": True,
        "rejected_anchor_counts": dict(rejected),
        "prepared_files": {name: file_record(out / name) for name in
                           ("worldengine_navtest_smoke.pkl", "worldengine_navtest_smoke_pdm.pkl",
                            "worldengine_navtest_smoke.yaml", "worldengine_navtest_smoke_rare.yaml")},
        "tool_sha256": sha256(Path(__file__)),
        "scope_not_verified": ["original dataset constructor and model pipeline",
                               "NPU training", "convergence", "official benchmark or GPU parity"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--pdm", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--rare-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs", type=int, default=3)
    parser.add_argument("--anchors-per-log", type=int, default=22)
    parser.add_argument("--camera-mode", choices=("copy", "symlink"), default="copy")
    args = parser.parse_args()
    try:
        report = prepare(args)
    except Exception as error:
        print(json.dumps({"result": "fail", "error_type": type(error).__name__,
                          "error": str(error), "output": str(args.output)}), file=sys.stderr)
        return 1
    print(json.dumps({key: report[key] for key in
                      ("result", "logs", "anchors", "frames", "camera_files",
                       "rare_tokens", "same_log_normal_tokens", "mini_rare_filter",
                       "expected_heldout_log", "dataset_scope")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
