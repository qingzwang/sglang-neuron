#!/bin/bash
# pd_hy4.sh <prefill|decode|balancer> <prefill ip> <decode ip>: Hy4-FP8 1P1D, TP64 EP64 on each node (NIXL LIBFABRIC
# over EFA), the dev node's best decode flags; both roles the same KV layout (page, context, pool), the decode with DFly
source /mnt/nvme/work/hy4/env.sh
export PYTHONPATH=/mnt/nvme/work/nixl_overlay:$PYTHONPATH; unset NIXL_PLUGIN_DIR
export SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC NEURON_RT_VISIBLE_CORES=0-63
# long-context prefill is the bottleneck at concurrency (4 cold 128-380K sessions queue ~9 min of prefill): SGLang's
# 300 s defaults made the decode give up on a request still queued on the prefill ("timed out after 300.0s in
# KVPoll.WaitingForInput"); an aggregated server just queues it
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=${PD_WAIT_TIMEOUT:-3600} SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=${PD_WAIT_TIMEOUT:-3600}
export HY4_NKI_MOE_SUB=1 HY4_HC_CHAIN=1 HY4_NKI_ATTN_IN=1 HY4_MOE_TP=16 HY4_NKI_ATTN_CORE=1 HY4_NKI_LAYER=1 \
  HY4_QKV_SH=1 HY4_NKI_INDEX_DECODE=1 HY4_IDXQ_SH=1 ${HY4_EXTRA:-}
C=/mnt/nvme/work/hy4/cache_pd_$1_$(cat /mnt/nvme/sglang-neuron/sglang_neuron/models/hy4/kernels/*.py | md5sum | cut -c1-8)
export TORCH_NEURONX_NEFF_CACHE_DIR=$C/neff TORCH_NEURONX_HLO_CACHE_DIR=$C/hlo NKI_COMPILE_CACHE_URL=$C/nki
# no Hy4 NKI kernel changed between launches of this cache (its directory is keyed by their md5): restamp it for edits
# elsewhere in the tree, as run_dfly.sh does
[ "$1" != balancer ] && for d in neff hlo nki; do python -m sglang_neuron.backend.compile_cache stamp $C/$d >/dev/null 2>&1; done
M=/mnt/nvme/models/Hy4-FP8-text
COMMON=(--model-path $M --tp-size 64 --ep-size 64 --page-size 1024 --context-length ${CTX:-524288}
  --chunked-prefill-size ${CHUNK:-1024} --max-running-requests ${RUNNING:-4} --max-total-tokens ${TOKENS:-3145728}
  --watchdog-timeout 3600 --disaggregation-transfer-backend ${PD_BACKEND:-nixl} --disaggregation-bootstrap-port 8998)
# PD_BACKEND=mooncake: Mooncake's EFA transport (sgl-project/sglang-neuron#1); libfabric registers Neuron HBM through
# the Neuron SDK's libraries
[ "${PD_BACKEND:-nixl}" = mooncake ] && export LD_LIBRARY_PATH=/opt/aws/neuron/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}
# AGENTX=1: the AgentX launch's serving flags (served name, metrics, cache report; the decode samples top-k / top-p)
[ -n "$AGENTX" ] && COMMON+=(--served-model-name tencent/Hy4-FP8 --enable-metrics --enable-cache-report)
# the device sampler, the same on both roles (the prefill samples each request's first token)
export SGLANG_NEURON_SAMPLER=${SAMPLER:-$([ -n "$AGENTX" ] && echo topk_topp || echo greedy)}
case $1 in
  prefill) exec python -m sglang.launch_server "${COMMON[@]}" --disaggregation-mode prefill --host $2 --port 30000 ;;
  decode)
    exec python -m sglang.launch_server "${COMMON[@]}" --cuda-graph-bs-decode ${DECODE_BS:-1 4} \
      --speculative-algorithm DFLASH --speculative-draft-model-path $M --speculative-num-draft-tokens 7 \
      --disable-overlap-schedule --disaggregation-mode decode --host $3 --port 30001 ;;
  balancer) exec python -m sglang_neuron.disaggregation.balancer --prefill http://$2:30000 --decode http://$3:30001 \
      --host ${BALANCER_HOST:-127.0.0.1} --port 8000 ;;
esac
