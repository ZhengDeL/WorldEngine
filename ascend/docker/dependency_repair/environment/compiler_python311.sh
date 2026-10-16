#!/bin/sh
set -eu
# Actual installed SDK components and their real CPython3.11 dependency prefix.
# The model interpreter and its protobuf4 environment are never added here.
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH=/opt/worldengine-sdk-components:/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe
exec /opt/worldengine-sdk-tools311/bin/python "$@"
