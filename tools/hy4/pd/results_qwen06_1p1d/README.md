# Qwen3-0.6B 1P1D on two trn2.48xlarge (capacity block cr-03a4e6edbb3f4a771), NIXL LIBFABRIC over EFA

- prefill i-03a3d5f11d86e5db3 (172.31.42.175), decode i-0aa05f3deb2f888f1 (172.31.37.223); 16 EFA interfaces each;
  examples/qwen3_0p6b scripts (TP4 each) via hy4/pd_qwen.sh, NIXL overlay from tools/build_nixl.sh.
- First attempt: EFA "Unreachable remote" (fi_pingpong) and "Decode transfer failed": the default security group had
  the inbound self rule but only a 0.0.0.0/0 egress; EFA needs an egress rule to the group itself (added:
  sgr-0070bd40e7cd928b5).
- After: fi_pingpong ~10 us one way (small messages), 658 MB/s at 8 KB (one QP, ping-pong).
- 17 x 23 through the balancer: \boxed{391} (request_17x23.txt).
- GSM8K first 100 questions through PD: accuracy 0.440, invalid 0.000, latency 3.78 s, output 2729 tok/s
  (the trn3 reference layout reports about 0.47).
