#!/bin/bash
# pd_qwen.sh <prefill|decode|balancer> <prefill ip> <decode ip>: the Qwen3-0.6B 1P1D validation (NIXL LIBFABRIC over EFA),
# the examples/qwen3_0p6b scripts in the dev environment (the NIXL overlay first on PYTHONPATH)
source /mnt/nvme/work/hy4/env.sh
export PYTHONPATH=/mnt/nvme/work/nixl_overlay:$PYTHONPATH; unset NIXL_PLUGIN_DIR
export SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC
export MODEL_PATH=/mnt/nvme/models/Qwen3-0.6B CACHE_DIR=/mnt/nvme/work/pdcache_qwen06
E=/mnt/nvme/sglang-neuron/examples/qwen3_0p6b
case $1 in
  prefill) HOST=$2 exec bash $E/run_prefill.sh ;;
  decode) HOST=$3 CORES=0-3 PORT=30001 exec bash $E/run_decode.sh ;;
  balancer) PREFILL_URL=http://$2:30000 DECODE_URL=http://$3:30001 exec bash $E/run_balancer.sh ;;
esac
