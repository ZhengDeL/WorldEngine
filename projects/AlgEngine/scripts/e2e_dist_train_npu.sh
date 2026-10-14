#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash scripts/e2e_dist_train_npu.sh <config> <num_chips> [resume_checkpoint] [-- launcher_options]' >&2
  exit 2
fi
SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORLDENGINE_SOURCE=$(cd "$SCRIPT_DIRECTORY/../../.." && pwd)
source "$WORLDENGINE_SOURCE/ascend/tools/launcher_common.sh"
CONFIG=$(resolve_algengine_config "$1")
CHIPS=$2
shift 2
RESUME=''
if [[ $# -gt 0 && "$1" != --* ]]; then RESUME=$1; shift; fi
if [[ ${1:-} == -- ]]; then shift; fi
make_common_arguments "$CHIPS" train
check_extra_arguments "$@"
: "${INIT_CHECKPOINT:?Set INIT_CHECKPOINT to an official warm-start or fine-tuning initializer}"
TRAIN_ARGS=(--task train --config "$CONFIG" --checkpoint "$INIT_CHECKPOINT")
[[ -z "$RESUME" ]] || TRAIN_ARGS+=(--resume "$RESUME")
exec "${PYTHON_BIN:-python3}" "$WORLDENGINE_SOURCE/ascend/run_container.py" "${COMMON_ARGS[@]}" "${TRAIN_ARGS[@]}" "$@"
