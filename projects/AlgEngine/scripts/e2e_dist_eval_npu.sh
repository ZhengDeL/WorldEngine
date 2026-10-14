#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo 'Usage: bash scripts/e2e_dist_eval_npu.sh <dumped_config> <checkpoint> <num_chips> [-- launcher_options]' >&2
  exit 2
fi
SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORLDENGINE_SOURCE=$(cd "$SCRIPT_DIRECTORY/../../.." && pwd)
source "$WORLDENGINE_SOURCE/ascend/tools/launcher_common.sh"
CONFIG=$(resolve_algengine_config "$1")
CHECKPOINT=$2
CHIPS=$3
shift 3
if [[ ${1:-} == -- ]]; then shift; fi
make_common_arguments "$CHIPS" inference
check_extra_arguments "$@"
exec "${PYTHON_BIN:-python3}" "$WORLDENGINE_SOURCE/ascend/run_container.py" "${COMMON_ARGS[@]}" \
  --task inference --config "$CONFIG" --checkpoint "$CHECKPOINT" "$@"
