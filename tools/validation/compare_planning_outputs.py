#!/usr/bin/env python3
"""Compare exported planning trajectories and official score CSVs on CPU.

This is result analysis, not an inference runner, scorer, or quality gate.
The prediction format is the original AlgEngine list of NumPy records.
No Torch, CUDA, Ascend runtime, model, or device APIs are imported.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np

METRICS = (
    "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
    "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def original_input(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("An existing regular input file is required")
    return path.resolve(strict=True)


def latin1_bytes(value, encoding, /):
    # Original NumPy protocol2 exports encode bytes by exactly this operation.
    if type(value) is not str or type(encoding) is not str or encoding != "latin1":
        raise ValueError("Only strict original NumPy Latin1 reconstruction is allowed")
    return value.encode("latin1", "strict")


class PredictionUnpickler(pickle.Unpickler):
    """Allow NumPy arrays/scalars; reject arbitrary imports and Torch storages."""
    def find_class(self, module, name):
        if (module, name) == ("_codecs", "encode"):
            return latin1_bytes
        allowed = {("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}
        for prefix in ("numpy.core", "numpy._core"):
            allowed[(prefix + ".multiarray", "_reconstruct")] = np.core.multiarray._reconstruct
            allowed[(prefix + ".multiarray", "scalar")] = np.core.multiarray.scalar
            allowed[(prefix + ".numeric", "_frombuffer")] = np.core.numeric._frombuffer
        try:
            return allowed[(module, name)]
        except KeyError:
            raise ValueError("Unexpected prediction-pickle global: " + module + "." + name) from None


def predictions(path, poses, candidates):
    with path.open("rb") as stream:
        records = PredictionUnpickler(stream).load()
        if stream.read(1):
            raise ValueError("Trailing bytes after prediction object")
    if not isinstance(records, list) or not records:
        raise ValueError("A nonempty original AlgEngine prediction list is required")
    result = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Prediction record must be a dictionary")
        token = record.get("token")
        trajectory = record.get("trajectory")
        chosen = record.get("chosen_ind")
        if type(token) is not str or not token or token in result:
            raise ValueError("Prediction tokens must be nonempty and unique")
        if not isinstance(trajectory, np.ndarray) or trajectory.shape != (poses, 3):
            raise ValueError("Each trajectory must be a NumPy array of shape (poses, 3)")
        if trajectory.dtype.kind != "f" or not np.isfinite(trajectory).all():
            raise ValueError("Trajectories must contain finite floating-point values")
        if type(chosen) is not int or not 0 <= chosen < candidates:
            raise ValueError("chosen_ind is outside the declared vocabulary size")
        result[token] = {"trajectory": trajectory, "chosen_ind": chosen}
    return result


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Finite nonempty measurements are required")
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def compare(left, right, poses):
    if set(left) != set(right):
        raise ValueError("Prediction token sets differ; no implicit intersection is selected")
    rows = []
    for token in sorted(left):
        first, second = left[token], right[token]
        difference = second["trajectory"].astype(np.float64) - first["trajectory"].astype(np.float64)
        position = np.linalg.norm(difference[:, :2], axis=1)
        heading = np.abs(np.arctan2(np.sin(difference[:, 2]), np.cos(difference[:, 2])))
        rows.append({
            "token": token, "chosen_ind_left": first["chosen_ind"], "chosen_ind_right": second["chosen_ind"],
            "chosen_ind_equal": first["chosen_ind"] == second["chosen_ind"],
            "trajectory_bitwise_equal": first["trajectory"].dtype == second["trajectory"].dtype and first["trajectory"].tobytes() == second["trajectory"].tobytes(),
            "trajectory_exact_numeric_equal": bool(np.array_equal(first["trajectory"], second["trajectory"])),
            "pose_position_difference_metres": position.tolist(),
            "pose_heading_circular_difference_radians": heading.tolist(),
            "mean_position_difference_metres": float(position.mean()),
            "final_position_difference_metres": float(position[-1]),
            "maximum_position_difference_metres": float(position.max()),
            "maximum_heading_circular_difference_radians": float(heading.max()),
            "max_abs_component_difference": {"x": float(np.abs(difference[:, 0]).max()), "y": float(np.abs(difference[:, 1]).max()), "unwrapped_heading": float(np.abs(difference[:, 2]).max())},
        })
    return {"tokens": len(rows), "poses_per_token": poses, "compared_poses": len(rows) * poses,
            "chosen_ind_equal_count": sum(row["chosen_ind_equal"] for row in rows),
            "exact_trajectory_equal_count": sum(row["trajectory_bitwise_equal"] for row in rows),
            "all_pose_position_difference_metres": statistics([x for row in rows for x in row["pose_position_difference_metres"]]),
            "per_token_final_position_difference_metres": statistics([row["final_position_difference_metres"] for row in rows]),
            "all_pose_heading_circular_difference_radians": statistics([x for row in rows for x in row["pose_heading_circular_difference_radians"]]),
            "per_token": rows}


def official_scores(path, tokens):
    per_token = {}
    ordered = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        expected = {"token", "valid", *METRICS}
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) or set(reader.fieldnames) != expected:
            raise ValueError("Expected original token/valid/seven-metric official CSV columns")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Official score row has missing or extra CSV cells")
            token = row["token"]
            if not token or token in per_token or row["valid"].lower() != "true":
                raise ValueError("Official score rows must be unique and valid")
            values = {name: float(row[name]) for name in METRICS}
            if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values.values()):
                raise ValueError("Official score metrics must be finite values in [0, 1]")
            per_token[token] = values
            ordered.append(values)
    if set(per_token) != set(tokens):
        raise ValueError("Official score tokens do not exactly match prediction tokens")
    means = {name: float(np.asarray([row[name] for row in ordered], dtype=np.float64).mean()) for name in METRICS}
    return {"valid_tokens": len(ordered), "metrics": means, "per_token": [{"token": token, **per_token[token]} for token in sorted(per_token)]}


def identity(path):
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("left-predictions", "right-predictions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--left-score-csv", type=Path)
    parser.add_argument("--right-score-csv", type=Path)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--comparison-kind", choices=("same_checkpoint_inference", "separately_trained", "unspecified"), default="unspecified")
    parser.add_argument("--poses", type=int, default=40)
    parser.add_argument("--candidates", type=int, default=8192)
    parser.add_argument("--provenance", type=Path, help="Optional existing provenance JSON; record its hash, without claiming its contents are independently verified")
    args = parser.parse_args()
    if args.poses < 1 or args.candidates < 1:
        parser.error("Positive pose and vocabulary counts are required")
    if (args.left_score_csv is None) != (args.right_score_csv is None):
        parser.error("Supply both original official score CSVs or neither")
    if args.output.exists() or args.output.is_symlink():
        parser.error("Fresh output JSON required; existing results are never overwritten")
    files = {"left_predictions": original_input(args.left_predictions), "right_predictions": original_input(args.right_predictions)}
    for name in ("left_score_csv", "right_score_csv", "provenance"):
        if getattr(args, name) is not None:
            files[name] = original_input(getattr(args, name))
    if "provenance" in files:
        with files["provenance"].open(encoding="utf-8") as stream:
            if not isinstance(json.load(stream), (dict, list)):
                parser.error("Provenance must be an existing JSON object or list")
    before = {name: identity(path) for name, path in files.items()}
    left = predictions(files["left_predictions"], args.poses, args.candidates)
    right = predictions(files["right_predictions"], args.poses, args.candidates)
    values = compare(left, right, args.poses)
    result = {"schema_version": 1, "result": "complete_CPU_planning_output_measurement_no_quality_threshold", "labels": {"left": args.left_label, "right": args.right_label}, "comparison_kind": args.comparison_kind, "compared_values": values, "quality_equivalence_accepted": False, "quality_threshold": None, "device_operations_performed": False, "model_inference_or_official_scoring_performed": False, "input_before": before, "analysis_tool_sha256": sha256(Path(__file__)), "numpy_version": np.__version__, "interpretation": "Same-checkpoint labels alone do not verify checkpoint identity. Compare inference provenance separately. Separately trained checkpoints include training-run differences. The original official scorer uses eight sampled poses; this analysis measures every exported pose. No quality, convergence, repeatability, or GPU/NPU-equivalence threshold is applied."}
    if args.left_score_csv is not None:
        scores = {"left": official_scores(files["left_score_csv"], left), "right": official_scores(files["right_score_csv"], right)}
        result["original_official_score_CSVs"] = scores
        result["official_metric_delta_right_minus_left"] = {name: scores["right"]["metrics"][name] - scores["left"]["metrics"][name] for name in METRICS}
        result["official_metric_per_token_delta_right_minus_left"] = [{"token": a["token"], **{name:b[name]-a[name] for name in METRICS}} for a,b in zip(scores["left"]["per_token"],scores["right"]["per_token"])]
    result["input_after"] = {name: identity(path) for name, path in files.items()}
    if before != result["input_after"]:
        raise ValueError("An original result input changed during comparison")
    if any(name == "torch" or name.startswith(("torch.", "torch_npu", "ascend.")) for name in sys.modules):
        raise RuntimeError("Result analysis unexpectedly imported an accelerator framework")
    raw = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"result": result["result"], "tokens": values["tokens"], "compared_poses": values["compared_poses"], "chosen_ind_equal_count": values["chosen_ind_equal_count"], "quality_equivalence_accepted": False, "output_sha256": sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
