# Hy4-FP8 1P1D on two trn2.48xlarge, NIXL LIBFABRIC over EFA

- prefill i-03a3d5f11d86e5db3 (TP64 EP64), decode i-0aa05f3deb2f888f1 (TP64 EP64, DFly, greedy), the dev node's best
  decode flags on both (tools/hy4/pd/pd_hy4.sh), 512K context capacity, page 1024, 3.1M-token pool.
- The KV store's regions (78 layers' MLA rows + 21 full layers' index keys; each page's rows one run a rank) are
  registered by Hy4's accounting pool (models/hy4/kv.py get_contiguous_buf_infos / get_kv_layer_ids); with equal TP
  and the same context-parallel split, prefill rank r's pages land in decode rank r's rows.
- First attempt refused at registration: "Layer metadata length must match transfer entries: src metadata=2
  entries=99" (the accounting pool's one-layer ids); fixed with get_kv_layer_ids.
- dfly_check's 6 prompts (128 tokens) through the balancer: outputs bit-identical to the aggregated server with the
  same flags (logs/ab_cmpchk), accept lengths identical: 2.61 3.87 2.78 6.73 2.56 6.4 (check.client, check.json).
