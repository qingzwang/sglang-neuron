# Hy4 prefill/decode disaggregation on two trn2.48xlarge (NIXL LIBFABRIC over EFA)

Scripts used to bring up and test 1P1D on two capacity-block trn2 nodes from a dev node:

- `nics.json`, `userdata.sh`: the launch's 16 EFA interfaces (one per network card) and the instance-store RAID0 at
  /mnt/nvme. The nodes have no auto public IP (multiple interfaces): each gets an Elastic IP for SSM.
- The security group must allow all traffic **to and from itself, inbound and outbound**: with only a 0.0.0.0/0
  egress, EFA fails ("Unreachable remote" in fi_pingpong, "Decode transfer failed" in serving).
- `pd.sh`: run a command on either node over SSM (base64-wrapped, so it runs verbatim).
- `fetch_env.sh`, `fetch_model.sh`: the dev node's environment (the /opt Neuron venv, /source-repos, the code, the
  NIXL overlay from tools/build_nixl.sh) and the checkpoint, pulled from the dev node's file server at the same paths.
- `pd_qwen.sh`: the Qwen3-0.6B validation (examples/qwen3_0p6b), `pd_hy4.sh`: Hy4 1P1D.
- `results_qwen06_1p1d/`, `results_hy4_1p1d/`: what they showed.
