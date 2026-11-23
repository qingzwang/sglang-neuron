#!/bin/bash
# agentx_pd.sh: AgentX points against the Hy4 1P1D pair (pd_hy4.sh AGENTX=1), the client on this dev node; fresh
# servers per point. CONCS (default "1 4"), OUT.
set -u
H=/mnt/nvme/work/hy4; OUT=${OUT:-/mnt/nvme/work/agentx-results-trn2/dfly_pd_${PD_BACKEND:-nixl}}; mkdir -p $OUT
PF=172.31.42.175; DC=172.31.37.223
for CONC in ${CONCS:-1 4}; do
  echo "=== conc $CONC $(date -u +%FT%TZ)"
  $H/pd.sh prefill 'systemctl stop pdh-balancer pdh-prefill pdh-check 2>/dev/null; systemctl reset-failed 2>/dev/null; pkill -9 -f "sglang::"; true' 120 >/dev/null
  $H/pd.sh decode 'systemctl stop pdh-decode 2>/dev/null; systemctl reset-failed 2>/dev/null; pkill -9 -f "sglang::"; true' 120 >/dev/null
  sleep 5
  $H/pd.sh decode "systemd-run --unit=pdh-decode --setenv=AGENTX=1 --setenv=PD_BACKEND=${PD_BACKEND:-nixl} --property=StandardOutput=file:/mnt/nvme/pdh_decode_c$CONC.log --property=StandardError=file:/mnt/nvme/pdh_decode_c$CONC.log /mnt/nvme/work/hy4/pd_hy4.sh decode $PF $DC" 60 >/dev/null
  $H/pd.sh prefill "systemd-run --unit=pdh-prefill --setenv=AGENTX=1 --setenv=PD_BACKEND=${PD_BACKEND:-nixl} --property=StandardOutput=file:/mnt/nvme/pdh_prefill_c$CONC.log --property=StandardError=file:/mnt/nvme/pdh_prefill_c$CONC.log /mnt/nvme/work/hy4/pd_hy4.sh prefill $PF $DC" 60 >/dev/null
  for i in $(seq 360); do curl -sf http://$PF:30000/health >/dev/null && curl -sf http://$DC:30001/health >/dev/null && break; sleep 10; done
  curl -sf http://$DC:30001/health >/dev/null || { echo "servers not healthy for conc $CONC"; continue; }
  $H/pd.sh prefill "systemd-run --unit=pdh-balancer --setenv=BALANCER_HOST=$PF --property=StandardOutput=file:/mnt/nvme/pdh_balancer_c$CONC.log --property=StandardError=file:/mnt/nvme/pdh_balancer_c$CONC.log /mnt/nvme/work/hy4/pd_hy4.sh balancer $PF $DC" 60 >/dev/null
  for i in $(seq 30); do curl -sf http://$PF:8000/health >/dev/null && break; sleep 5; done
  cat > $OUT/point_conc$CONC.env <<P
SERVER_URL=http://$PF:8000
SERVER_METRICS_URLS=http://$PF:30000/metrics,http://$DC:30001/metrics
MODEL=tencent/Hy4-FP8
SERVED_MODEL_NAME=tencent/Hy4-FP8
TOKENIZER=/mnt/nvme/models/Hy4-FP8-text
MODEL_PREFIX=hy4
HARDWARE=trn2
FRAMEWORK=sglang
PRECISION=fp8
IMAGE=sglang-neuron-local
TP=64
EP=64
PP=1
DCP=1
PCP=1
DP_ATTENTION=false
SPEC_DECODING=dfly
DISAGG=true
KV_P2P_TRANSFER=${PD_BACKEND:-nixl}
PREFILL_NUM_WORKERS=1
PREFILL_TP=64
DECODE_NUM_WORKERS=1
DECODE_TP=64
DECODE_EP=64
KV_OFFLOADING=none
CONC=$CONC
OUTPUT_ROOT=$OUT
P
  (cd /mnt/nvme/agentx-bench-main && bash client/run_point.sh $OUT/point_conc$CONC.env > $OUT/client_conc$CONC.log 2>&1)
  echo "client exit $? conc $CONC $(date -u +%FT%TZ)"
done
echo "=== all done $(date -u +%FT%TZ)"
