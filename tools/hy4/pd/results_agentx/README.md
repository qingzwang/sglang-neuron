# AgentX (agentx-bench), Hy4-FP8 on trn2: aggregated vs 1P1D PD over Mooncake (EFA)

Fresh servers per point (agentx-bench `fresh_server_per_point`), 3600 s measurement, seed 42, the
`client/run_point.sh` defaults; TP/EP fields in chips (trn2.48xlarge: 16 chips; Neuron TP64 EP64 at LNC2 = 16), the
aggregated points re-imported with them (`client/import_point.sh`). Every point 0% request errors.

- aggregated: one trn2.48xlarge, TP64 EP64, DFly, the dev node's best decode flags (tools/hy4/pd/pd_hy4.sh's list),
  512K context, top-k/top-p device sampler.
- PD: prefill node TP64 EP64 + decode node TP64 EP64 with DFly, the same flags, Mooncake's EFA transport
  (sgl-project/sglang-neuron#1 + Hy4 served), sglang_neuron.disaggregation.balancer.

| AgentX | aggregated c1 | PD c1 | aggregated c4 | PD c4 |
|---|---|---|---|---|
| output tok/s | 24.7 | 24.7 | 32.9 | **36.9** |
| ITL avg / p50 / p90 (ms) | 33.5 / 28.2 / 42.3 | 29.8 / 26.4 / 37.5 | 102.3 / 88.0 / 126.6 | **73.1 / 73.9 / 101.1** |
| TTFT avg / p50 / p90 (s) | 8.0 / 3.6 / 16.8 | 8.2 / 4.2 / 15.7 | 22.1 / 4.8 / 62.8 | **16.6 / 4.6 / 39.7** |
| request latency avg / p90 (s) | 39.5 / 95.8 | 36.7 / 83.7 | 96.1 / 293.5 | **78.2 / 248.0** |
| requests in the hour | 105 | 105 | 145 | 159 |
| prompt cache read | 96.9% | 96.9% | 96.4% | 96.2% |
| chips | 16 | 32 | 16 | 32 |

- At conc1 (one session) PD barely matters: prefill seldom competes with decode (ITL -6..-11%).
- At conc4 the decode no longer waits behind other sessions' long prefills: ITL -20..-29%, TTFT p90 -37%, request
  latency -19%; but on twice the chips (per-chip output throughput is lower: 36.9 / 32 vs 32.9 / 16).
- KV transfer (prefill TP0, sglang:kv_transfer_*): c1 145 ms avg (p50 74, p90 263) for 273 MB a rank (~2.6 GB/s a rank,
  ~160 GB/s over the 64 ranks); c4 328 ms avg (p50 68, p90 1036), 1.4 GB/s a rank. A small share of TTFT, which is
  prefill compute (~1.2 s a 1024-token chunk).
- The c4 decode verifies 4 x 7 = 28 rows a step, past the <= 8-row fused decode kernels: ~74 ms ITL; the next decode
  target for concurrency above 1.
- One c4 warmup request timed out on the decode (SGLang's 300 s SGLANG_DISAGGREGATION_WAITING_TIMEOUT: 4 cold
  sessions queued ~9 min of prefill); pd_hy4.sh now sets 3600 s. No transfer failed in the measurement.
