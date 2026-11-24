# AgentX, Hy4 1P1D (Mooncake over EFA), v2 code (2026-11-24)

Same deployment and method as `../` (fresh servers per point, 3600 s, seed 42); the decode adds
`HY4_NKI_SPARSE_DECODE=1` (the NKI sparse attention for >8-row verifies) and the prefill `fn_hilo` (hc_pre weights
split at load). 0% errors at both points.

| AgentX | PD v1 c1 | PD v2 c1 | PD v1 c4 | PD v2 c4 |
|---|---|---|---|---|
| output tok/s | 24.7 | 24.65 | 36.9 | 38.8 |
| ITL avg / p50 / p90 (ms) | 29.8 / 26.4 / 37.5 | 29.5 / 27.3 / 38.1 | 73.1 / 73.9 / 101.1 | 74.2 / 78.3 / 96.7 |
| TTFT avg / p50 / p90 (s) | 8.2 / 4.2 / 15.7 | 8.4 / 4.2 / 14.7 | 16.6 / 4.6 / 39.7 | 15.8 / 4.9 / 37.6 |
| request latency avg / p90 (s) | 36.7 / 83.7 | 34.6 / 79.2 | 78.2 / 248.0 | 77.6 / 243.8 |
| requests | 105 | 105 | 159 | 165 |
