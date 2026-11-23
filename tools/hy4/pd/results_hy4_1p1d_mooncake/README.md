# Hy4-FP8 1P1D over Mooncake's EFA transport (sgl-project/sglang-neuron#1 + Hy4 served)

- Same nodes and flags as results_hy4_1p1d (NIXL), `PD_BACKEND=mooncake` (pd_hy4.sh): `--disaggregation-transfer-backend
  mooncake`, MOONCAKE_PROTOCOL=efa, LD_LIBRARY_PATH with /opt/aws/neuron/lib, NEURON_RT_MAP_HBM=1;
  mooncake-transfer-engine-non-cuda 0.3.14 (cp312) in the venv. The engine logs "protocol: efa".
- dfly_check's 6 prompts: outputs bit-identical to the aggregated server and to the NIXL pair, accept lengths
  identical (2.61 3.87 2.78 6.73 2.56 6.4); request times within noise of NIXL (4.43 2.96 3.78 1.98 3.99 1.99 s vs
  3.97 2.87 3.74 1.94 4.00 2.05 s).
- EFA hardware counters on the prefill node after the runs: rdma_write_bytes 517 GB over all 16 devices (since boot,
  NIXL runs included).
