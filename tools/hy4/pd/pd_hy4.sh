#!/bin/bash
# pd_hy4.sh <prefill|decode|balancer> <prefill ip> <decode ip>: Hy4-FP8 1P1D, TP64 EP64 on each node (NIXL LIBFABRIC
# over EFA), the dev node's best decode flags; both roles the same KV layout (page, context, pool), the decode with DFly
source /mnt/nvme/work/hy4/env.sh
export PYTHONPATH=/mnt/nvme/work/nixl_overlay:$PYTHONPATH; unset NIXL_PLUGIN_DIR
export SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC NEURON_RT_VISIBLE_CORES=0-63
export HY4_NKI_MOE_SUB=1 HY4_HC_CHAIN=1 HY4_NKI_ATTN_IN=1 HY4_MOE_TP=16 HY4_NKI_ATTN_CORE=1 HY4_NKI_LAYER=1 \
  HY4_QKV_SH=1 HY4_NKI_INDEX_DECODE=1 HY4_IDXQ_SH=1 ${HY4_EXTRA:-}
C=/mnt/nvme/work/hy4/cache_pd_$1_$(cat /mnt/nvme/sglang-neuron/sglang_neuron/models/hy4/kernels/*.py | md5sum | cut -c1-8)
export TORCH_NEURONX_NEFF_CACHE_DIR=$C/neff TORCH_NEURONX_HLO_CACHE_DIR=$C/hlo NKI_COMPILE_CACHE_URL=$C/nki
M=/mnt/nvme/models/Hy4-FP8-text
COMMON=(--model-path $M --tp-size 64 --ep-size 64 --page-size 1024 --context-length ${CTX:-524288}
  --chunked-prefill-size ${CHUNK:-1024} --max-running-requests ${RUNNING:-4} --max-total-tokens ${TOKENS:-3145728}
  --watchdog-timeout 3600 --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 8998)
case $1 in
  prefill) exec python -m sglang.launch_server "${COMMON[@]}" --disaggregation-mode prefill --host $2 --port 30000 ;;
  decode) export SGLANG_NEURON_SAMPLER=${SAMPLER:-greedy}
    exec python -m sglang.launch_server "${COMMON[@]}" --cuda-graph-bs-decode ${DECODE_BS:-1 4} \
      --speculative-algorithm DFLASH --speculative-draft-model-path $M --speculative-num-draft-tokens 7 \
      --disable-overlap-schedule --disaggregation-mode decode --host $3 --port 30001 ;;
  balancer) exec python -m sglang_neuron.disaggregation.balancer --prefill http://$2:30000 --decode http://$3:30001 ;;
esac
