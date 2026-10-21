#!/usr/bin/env bash
# Shared host-side arguments; no model or environment installation occurs here.

resolve_algengine_config() {
  local candidate=$1
  if [[ "$candidate" != /* ]]; then
    if [[ -f "$WORLDENGINE_SOURCE/$candidate" ]]; then
      candidate="$WORLDENGINE_SOURCE/$candidate"
    else
      candidate="$WORLDENGINE_SOURCE/projects/AlgEngine/$candidate"
    fi
  fi
  [[ -f "$candidate" ]] || { printf 'Configuration missing: %s\n' "$candidate" >&2; return 2; }
  realpath "$candidate"
}

make_common_arguments() {
  local chips=$1 mode=$2 i ids=${DEVICE_IDS:-}
  local msda_backend=${MSDA_BACKEND:-reference}
  [[ "$chips" =~ ^([1-9]|1[0-6])$ ]] || { echo 'Select 1–16 logical chips on one host.' >&2; return 2; }
  : "${IMAGE:?Set IMAGE from the installation guide}"
  : "${IMAGE_ID:?Set IMAGE_ID to the actual docker image inspect ID}"
  : "${VOCAB:?Set VOCAB to test_8192_kmeans.npy}"
  : "${RUN_OUTPUT:?Set RUN_OUTPUT to a fresh output directory}"
  : "${CPU_MEMORY:?Set CPU_MEMORY, for example 64g for a single-chip mini run}"
  if [[ -z "$ids" ]]; then
    for ((i=0;i<chips;i++)); do ids+="${ids:+,}$i"; done
  fi
  local -a selected_devices
  IFS=',' read -r -a selected_devices <<< "$ids"
  [[ ${#selected_devices[@]} -eq "$chips" ]] || { echo 'DEVICE_IDS count must match num_chips.' >&2; return 2; }
  COMMON_ARGS=(--devices "$ids" --image "$IMAGE" --expected-image-id "$IMAGE_ID"
    --output "$RUN_OUTPUT" --vocab "$VOCAB" --memory "$CPU_MEMORY"
    --cpus "${CPUS:-16}" --precision fp32 --msda-backend "$msda_backend")
  if [[ -n "${MINI_PREPARED:-}" && -z "${DATA_ROOT:-}" ]]; then
    COMMON_ARGS+=(--mini-prepared "$MINI_PREPARED")
  elif [[ -n "${DATA_ROOT:-}" && -z "${MINI_PREPARED:-}" ]]; then
    COMMON_ARGS+=(--data-root "$DATA_ROOT")
  else
    echo 'Set exactly one of MINI_PREPARED or DATA_ROOT.' >&2; return 2
  fi
  [[ -z "${MAPS:-}" ]] || COMMON_ARGS+=(--maps "$MAPS")
  [[ -z "${METRIC_CACHE:-}" ]] || COMMON_ARGS+=(--official-metric-cache "$METRIC_CACHE")
  if [[ "$mode" == train ]]; then
    COMMON_ARGS+=(--runtime --upstream-schedule --batch-size "${BATCH_SIZE:-1}" --workers "${WORKERS:-0}")
    [[ -z "${RARE_SPLIT:-}" ]] || COMMON_ARGS+=(--rare-split "$RARE_SPLIT")
    [[ -z "${SYNTHETIC_ROOT:-}" ]] || COMMON_ARGS+=(--synthetic-root "$SYNTHETIC_ROOT")
  fi
}

check_extra_arguments() {
  local option explicit_backend=0 i
  for option in "$@"; do
    case "$option" in
      --msda-backend|--msda-backend=*) explicit_backend=1 ;;
    esac
  done
  if [[ "${MSDA_BACKEND:-reference}" != reference && "${MSDA_BACKEND:-reference}" != cuda-compatible ]]; then
    if [[ "$explicit_backend" == 0 ]]; then
      echo 'MSDA_BACKEND must be reference or cuda-compatible.' >&2; return 2
    fi
    # Do not let a stale environment value fail argparse before the explicit
    # option. The original Python CLI still validates every explicit value.
    for ((i=0;i<${#COMMON_ARGS[@]};i++)); do
      if [[ "${COMMON_ARGS[i]}" == --msda-backend ]]; then
        COMMON_ARGS[i+1]=reference
        break
      fi
    done
  fi
  for option in "$@"; do
    case "${option%%=*}" in
      --devices|--image|--expected-image-id|--output|--vocab|--checkpoint|--config|--task|--precision|--data-root|--mini-prepared|--memory|--cpus|--batch-size|--workers|--rare-split|--synthetic-root|--official-metric-cache|--maps)
        printf 'Set %s through the documented environment variable instead.\n' "$option" >&2; return 2 ;;
      --profile|--train-filter|--lr|--steps|--epochs|--resume|--stop-after|--stop-after-epochs|--duration-seconds|--checkpoint-interval|--check-interval|--pids-limit|--shm-size|--validate|--deterministic|--debug-anomaly|--inference-after|--detach|--runtime|--upstream-schedule|--msda-backend)
        ;; # Complete original option names only; never forward abbreviations.
      --*)
        printf 'Unsupported or abbreviated launcher option: %s\n' "$option" >&2; return 2 ;;
    esac
  done
}

check_inference_arguments() {
  check_extra_arguments "$@" || return
  local option
  for option in "$@"; do
    case "${option%%=*}" in
      --pids-limit|--shm-size|--deterministic|--debug-anomaly|--detach|--msda-backend)
        ;; # Applied to inference by the original Python launcher.
      --*)
        printf 'Training-only option is not supported by inference: %s\n' "$option" >&2; return 2 ;;
    esac
  done
}
