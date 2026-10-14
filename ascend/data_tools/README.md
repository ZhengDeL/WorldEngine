# Genuine SimScale subset preparation

Three SHA-locked CLIs convert genuine SimScale records, compute WE-derived rewards and export a training subset. Fresh server CPU replay completed on October 13, 2026 at 20:22:57 UTC: all four phases exited 0 without OOM, covering 26 converted clips, 832 history JPEGs, 8,192 newly computed reward candidates and 104 exported payloads. `CPU_REPRODUCTION_VERIFIED.json` records the tool/input hashes and actual outcomes.

The new export also passed the original dataset CPU checks: both original configurations and all four synthetic/normal requests returned from one prepare call, with no None or retry, exact tokens/image geometry and six finite 8,192-element vectors. Two SDK import-time device-name queries used previously observed chip metadata; the original getter was restored and the NPU remained uninitialized. This is not an unshimmed CPU import claim. See `GETITEM_CPU_VERIFIED.json`.

Fresh post8 one-rank controls completed at 20:54:02 UTC, and fresh two-rank controls at 21:00:51 UTC on October 13. Both configurations ran eight continuous updates and resumed their own checkpoint at update 4 to update 8. All original strict comparisons were bitwise equal with zero tensor/loss difference and identical tokens, LR and final RNG. Finetune compared 989 model tensors and 1,291/1,293 complete tensors for one/two ranks; cotrain compared 909 and 1,748/1,750. Complete counts include rank RNG. See `NPU_SINGLE_RANK_VERIFIED.json` and `NPU_TWO_RANK_VERIFIED.json`.

The result covers one genuine scene, one explicitly filtered normal anchor per configuration and the declared WE-derived reward protocol. The other six rollout/replay/BWM configurations still lack complete genuine exports. Full NAVTRAIN convergence, GPU comparison and strict sixteen-rank numerical acceptance remain incomplete. The validated converter/scorer/input-adapter function ASTs are unchanged; the exporter also passed 18 local queue, archive-safety and default dry-run fixtures. The recorded checks apply to the tool/input hashes in the evidence files.

The recipe uses one genuine SimScale scene, original NAVTRAIN normal records and original WorldEngine four-second CPU scoring. It does not reproduce published SimScale-v2 16384 rewards or run SimEngine rendering. Original JPEG bytes, calibration, GT, vocabulary order and normal PDM entries remain unchanged.

`INPUT_LOCKS.json` supplies exact publisher paths, revisions, byte counts, SHA256 and official/mirror URLs. Downloads and extraction must run on the server; reuse existing locked files. Minimum inputs total **47,500,431,504 bytes (47.50 GB)**, excluding maps/source/image/output: NAVTRAIN annotation16,125,493,728 and PDM23,736,976,138; synthetic metadata3,320,437,526; hist65 cameras241,575,885; mini12 cameras2,781,004,086; raw-mini metadata534,344,190; vocabulary3,932,288; history index1,824,152; two initializers434,885,463 and319,958,048. Complete annotation/PDM archives are required because the publisher supplies monolithic PKLs, even though the final normal subset contains eight records and two reward entries. The 19,305,089,518-byte incompatible published reward is not a training prerequisite. No 449 GB training-camera download, all32 mini-camera archives, 5 TB assets, LiDAR or future cameras are needed for this recipe.

Reuse genuine nuPlan maps v1.0 JSON/GPKG files. Their independent total size is unknown and excluded above. Use the locked image `local/worldengine:26.1.0-a3-clean-20261013`, ID `sha256:a182f52df8e9e343214e7631597230560877fce594199c6a0ea26a988832461f`, preserving its dependency PYTHONPATH. CPU containers mount source/input/tools read-only and fresh output writable, without NPU devices.

## Download and verify the initializer checkpoints

Both files are pinned to WorldEngine dataset revision `d0e88d858533ca407a3387a26577c083aa7b3617`. Download only on the server. Reuse existing complete files after checking `INPUT_LOCKS.json`; the commands below are for a fresh download directory. Enable `set -euo pipefail` before running. Stop on any size/hash failure and preserve the `.part` file; it is not an accepted initializer.

| Use | File | Bytes | SHA256 |
| --- | --- | ---: | --- |
| Explicit finetune override | `e2e_vadv2_100pct_ep8.pth` | 434,885,463 | `fe86c0bf7d992c172f24138b9df35d39b5c26ea04d90b16ead83c5d1c0a3587a` |
| Original cotrain initializer | `track_map_nuplan_r50_navtrain_100pct_bs1x8.pth` | 319,958,048 | `9849e940c49126602440e010e9ccae48fb6a1343212ded0897c51da8a79edc5a` |

The official HF and domestic hf-mirror routes use the same fixed revision, publisher path and content hash; complete URLs are in `INPUT_LOCKS.json`. The finetune 100pct checkpoint is an explicit override. The original `e2e_vadv2_ep8.pth` basename remains unavailable; do not rename the published checkpoint to imply equivalence.

```bash
# Server host only. Use a fresh input directory for downloads.
set -euo pipefail
INPUTS='<host locked public input directory>'
mkdir -p "$INPUTS"
PUBLISHER_REV=d0e88d858533ca407a3387a26577c083aa7b3617
HF_BASE="https://huggingface.co/datasets/OpenDriveLab/WorldEngine/resolve/$PUBLISHER_REV"
# If the official route is unavailable, select this exact-revision mirror instead:
# HF_BASE="https://hf-mirror.com/datasets/OpenDriveLab/WorldEngine/resolve/$PUBLISHER_REV"

test ! -e "$INPUTS/e2e_vadv2_100pct_ep8.pth"
curl --fail --location --retry 3 --continue-at - \
  --output "$INPUTS/e2e_vadv2_100pct_ep8.pth.part" \
  "$HF_BASE/data/alg_engine/ckpts/vadv2/e2e_vadv2_100pct_ep8.pth"
test "$(stat -c %s "$INPUTS/e2e_vadv2_100pct_ep8.pth.part")" = 434885463
printf '%s  %s\n' fe86c0bf7d992c172f24138b9df35d39b5c26ea04d90b16ead83c5d1c0a3587a \
  "$INPUTS/e2e_vadv2_100pct_ep8.pth.part" | sha256sum --check -
mv --no-clobber -- "$INPUTS/e2e_vadv2_100pct_ep8.pth.part" "$INPUTS/e2e_vadv2_100pct_ep8.pth"

test ! -e "$INPUTS/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth"
curl --fail --location --retry 3 --continue-at - \
  --output "$INPUTS/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth.part" \
  "$HF_BASE/data/alg_engine/ckpts/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth"
test "$(stat -c %s "$INPUTS/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth.part")" = 319958048
printf '%s  %s\n' 9849e940c49126602440e010e9ccae48fb6a1343212ded0897c51da8a79edc5a \
  "$INPUTS/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth.part" | sha256sum --check -
mv --no-clobber -- "$INPUTS/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth.part" \
  "$INPUTS/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth"
```

The tools are published in `ascend/data_tools/`, with independent hashes in `TOOLS_MANIFEST.json`. The defined runtime fingerprint covers `projects/AlgEngine/**/*.py` and `ascend/*.py`; nested preparation tools have their own SHA manifest. This explicitly separates runtime numerical code from data production identity. Actual annotation/filter/PDM/vocabulary content remains locked by the training contract. Data preparation records runtime identity without configuring training CANN flags or changing the resume contract. Each tool requires `--expected-runtime-sha256` and records original converter/SimEngine math hashes, selected ASTs, maps and output hashes. Each result applies to its recorded tool and runtime revision.

All three implemented CLIs default to dry-run; add `--execute` for CPU preparation. Conversion requires a fresh output. History appends only previously absent images and its own fresh receipt to an accepted conversion. Whole archives are audited for absolute/traversal paths, links, special members and duplicates before selected extraction; extractall is never used. All stages have bounded deadlines and preserve failures.

Set `WE_SOURCE`, `TOOLS`, `RUNTIME_FP`, `INPUTS`, `MAPS`, `OUT` and a future UTC `DEADLINE`. The exact commands are in README_zh.md and use CLI parameters without private addresses. CPU8GiB/4CPU runs original parse_log on26 hist65 clips, retaining2Hz,1.5-second past,4-second future and filter_boxes=False. The output parent must not contain `meta_datas`, because the original converter replaces that directory string. History then extracts/decodes832 original referenced JPEGs. Neither step rewrites converter math.

CPU64GiB/16CPU and the original eight scoring workers run `worldengine_8192_logged_reward.py`. Explicit sample identities are scene `fff002cb15c15a7a-000`, log `2021.06.09.17.23.18_veh-38_00773_01140`, folder `synthetic_reaction_pdm_v1.0-0`, and actual anchor `ab75b7addff3512d-000`. Twelve literal synthetic tokens correspond to consecutive raw indices213–224 in the734-frame source log; the scene ID is not the current anchor. Output is a separate `worldengine_8192_logged_traffic_derived_rewards.pkl`, not a truncation or replacement of published16384 labels.

The declared adapter gives the reference planner four genuine history observations and scoring the current plus eight future logged observations. Original NAVSIM creates global detections including logged ego. Publisher route IDs match; synthetic ego uses the original WE converter's zero-acceleration/steering policy. Original four-second simulator/scorer,8192 row order,eight workers and per-candidate progress normalization against the PDM reference are retained. This is not SimScale-v2 reward, full engine-lifecycle or native nuPlanDB-input equivalence.

The implemented `export_simscale_minimum.py` runs in a CPU container capped at128GiB/4CPU with at least192GiB host reserve. It directly validates the published full annotation/PDM/camera archive and current portable conversion/history/reward receipts, including source/math/tool/output SHA. It selects the original contiguous predecessor queue for explicit normal anchors `4b96572a4b9d5ea7` and `fff002cb15c15a7a`, preserving raw records/PDM and64 original JPEGs. Do not look up history at exact500000-microsecond timestamps. Each configuration produces35 payloads; shared synthetic assets produce34, totaling104. One-anchor normal whitelists intentionally narrow the candidate pool and do not establish full-data random-retrieval equivalence. The exporter validates the published inputs and portable stage records directly.

After actual portable subset reproduction, run the original two SimScale configurations using their own data root and `filters/navtrain_actual_normal_anchor.yaml`, sharing genuine synthetic assets. Each runs8 updates and restores its own iter4 to8, with unchanged CPU loader/comparator and tensor1e-6/1e-5,loss<1e-4 gates. Finetune uses published `e2e_vadv2_100pct_ep8.pth` as an explicit initializer override; the original `e2e_vadv2_ep8.pth` basename remains unfound. Cotrain uses `track_map_nuplan_r50_navtrain_100pct_bs1x8.pth`. Data tools do not configure training queue/precision flags; select a measured runtime/source configuration and retain each run's actual settings.

Fresh CPU production passed with whole input/source/math locks,832 history JPEGs,8192 finite candidate rewards and104 exported payloads. Original dataset/four-branch getitem checks also passed. The fresh one- and two-rank regressions also passed; each reproduction needs its own commands, image identity, actual exits/OOM, source/input/weight hashes and retained nonpassing results. The versioned one- and two-rank results apply to the recorded subset hashes and do not transfer automatically to newly generated data. The default cc91/post8 sixteen-rank controls contain tensors outside the original tolerances; synchronous controls have separate, single-repetition evidence in [the execution supplement](../execution/README.md). The other six genuine rollout/replay/BWM exports remain separate missing inputs. Full NAVTRAIN convergence, GPU parity and formal quality remain unaccepted.


Use the following commands inside the stated CPU containers. Set the quoted
placeholders to the selected source fingerprint and future UTC runtime deadline.
Keep the image dependency PYTHONPATH. Omitting `--execute` prints the plan.

```bash
WE_SOURCE=/workspace/WorldEngine
TOOLS=/tools
RUNTIME_FP='<actual source SHA256>'
INPUTS=/inputs
MAPS=/maps
OUT=/output/new_reproduction
DEADLINE='<authorized future UTC timestamp>'

# CPU8GiB/4CPU: original converter and all referenced history cameras.
python "$TOOLS/simscale_convert.py" --mode convert \
  --source "$WE_SOURCE" --expected-runtime-sha256 "$RUNTIME_FP" \
  --archive "$INPUTS/simscale_pdm_v1.0-0_meta_datas.tar.gz" \
  --hist-index "$INPUTS/simscale_pdm_v1.0-0_sensor_blobs_hist.json" \
  --output "$OUT/converted" --deadline "$DEADLINE" --execute
python "$TOOLS/simscale_convert.py" --mode history \
  --source "$WE_SOURCE" --expected-runtime-sha256 "$RUNTIME_FP" \
  --archive "$INPUTS/simscale_pdm_v1.0-0_sensor_blobs_hist_65.tar.gz" \
  --output "$OUT/converted" --deadline "$DEADLINE" --execute

# CPU64GiB/16CPU: unchanged original WE scoring with eight workers.
python "$TOOLS/worldengine_8192_logged_reward.py" \
  --source "$WE_SOURCE" --expected-runtime-sha256 "$RUNTIME_FP" \
  --metadata "$INPUTS/openscene_metadata_mini.tgz" \
  --conversion-root "$OUT/converted" --vocab "$INPUTS/test_8192_kmeans.npy" \
  --maps "$MAPS" --scene-id fff002cb15c15a7a-000 \
  --original-log 2021.06.09.17.23.18_veh-38_00773_01140 \
  --folder-name synthetic_reaction_pdm_v1.0-0 \
  --expected-anchor-token ab75b7addff3512d-000 \
  --output "$OUT/reward" --deadline "$DEADLINE" --execute

# CPU128GiB/4CPU, host reserve >=192GiB: genuine train-only104 payloads.
python "$TOOLS/export_simscale_minimum.py" \
  --source "$WE_SOURCE" --expected-runtime-sha256 "$RUNTIME_FP" \
  --annotation "$INPUTS/nuplan_openscene_navtrain.pkl" \
  --pdm "$INPUTS/pdm_8192_gt_cache_navtrain.pkl" \
  --camera-archive "$INPUTS/openscene_sensor_mini_camera_12.tgz" \
  --conversion-root "$OUT/converted" --reward-root "$OUT/reward" \
  --scene-id fff002cb15c15a7a-000 --folder-name synthetic_reaction_pdm_v1.0-0 \
  --original-log 2021.06.09.17.23.18_veh-38_00773_01140 \
  --expected-anchor-token ab75b7addff3512d-000 \
  --finetune-normal-token 4b96572a4b9d5ea7 --cotrain-normal-token fff002cb15c15a7a \
  --output "$OUT/subsets" --deadline "$DEADLINE" --execute
```

Use the following original launcher commands for a fresh reproduction and evaluate its own receipts. Use an owned deadline supervisor, positively idle hardware and fresh outputs; keep the source/image unchanged for each cold/resumed pair. Inspect all actual exits, initializer matches, original branches, finite losses/gradients, saved-state restoration and before/after precision receipts.

```bash
# Run on the server host after the CPU export and original dataset checks pass.
# Fill host paths; SOURCE/MAPS/INPUTS below are host paths, not container paths.
SOURCE='<host checkout of the verified runtime>'
SUBSETS='<host output of export_simscale_minimum.py>'
MAPS='<host official maps directory>'
INPUTS='<host locked public input directory>'
RUNS='<host fresh evidence directory>'
DEVICE='<one confirmed idle logical chip>'
IMAGE='<verified training image tag>'
IMAGE_ID='<verified training image ID>'

for CASE in finetune cotrain; do
  if [ "$CASE" = finetune ]; then
    NAME=e2e_vadv2_simscale_finetune_pdm_1round_8epoch
    INIT="$INPUTS/e2e_vadv2_100pct_ep8.pth"
  else
    NAME=e2e_vadv2_simscale_co_train_pdm_1round_8epoch
    INIT="$INPUTS/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth"
  fi
  COMMON=(--devices "$DEVICE" --config "$SOURCE/projects/AlgEngine/configs/simscale/$NAME.py"
    --checkpoint "$INIT" --vocab "$INPUTS/test_8192_kmeans.npy"
    --data-root "$SUBSETS/$NAME/data"
    --train-filter "$SUBSETS/$NAME/filters/navtrain_actual_normal_anchor.yaml"
    --synthetic-root "$SUBSETS/synthetic" --maps "$MAPS"
    --runtime --steps 8 --checkpoint-interval 4 --check-interval 1
    --batch-size 1 --workers 0 --deterministic --precision fp32
    --upstream-schedule --memory 64g --cpus 12 --pids-limit 4096
    --image "$IMAGE" --expected-image-id "$IMAGE_ID")
  python "$SOURCE/ascend/run_container.py" "${COMMON[@]}" \
    --output "$RUNS/${CASE}_continuous"
  python "$SOURCE/ascend/run_container.py" "${COMMON[@]}" \
    --resume "$RUNS/${CASE}_continuous/iter_4.pth" \
    --output "$RUNS/${CASE}_resumed"
done
```

Compare each continuous/resumed `iter_8.pth` and their JSONL records with the unchanged `ascend/compare_resume.py` in a CPU-only container. Use the official TorchNPU preimport/runpy recipe in [the runtime operations guide](../execution/README.md#original-numerical-comparison-on-cpu); importing the registered official loader must not initialize NPU. Original tensor tolerance remains `atol=1e-6, rtol=1e-5` and loss tolerance `1e-4`. A runnable job alone is not a passing numerical result. This recipe does not provide a full NAVTRAIN convergence or independent quality test.
