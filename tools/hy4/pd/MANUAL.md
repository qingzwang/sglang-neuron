# Hy4-FP8 PD 分离部署手册（两台 trn2.48xlarge，EFA，Mooncake / NIXL）

本手册记录在两台 trn2.48xlarge 上把 Hy4-FP8 跑成 1P1D（一台 prefill、一台 decode）的完整流程：启动实例、配置
EFA、测试 EFA、分发环境与模型、安装 Mooncake（或 NIXL overlay）、启动 prefill / decode / balancer、正确性检查、
跑 AgentX，以及过程中遇到的每个问题和解决办法。所有命令都在一台"开发机"（已有环境、模型和编译缓存的 trn2）上
执行，通过 SSM 控制两台新机器。本目录的脚本就是实际用过的版本。

目录：
1. [前提与架构](#1-前提与架构)
2. [启动实例（capacity block，16 个 EFA 接口）](#2-启动实例)
3. [网络与 EFA 配置](#3-网络与-efa-配置)
4. [测试 EFA](#4-测试-efa)
5. [分发环境和模型](#5-分发环境和模型)
6. [传输后端：Mooncake（推荐）或 NIXL](#6-传输后端)
7. [启动 prefill、decode、balancer](#7-启动-prefilldecodebalancer)
8. [正确性检查](#8-正确性检查)
9. [跑 AgentX](#9-跑-agentx)
10. [问题排查（实际踩过的坑）](#10-问题排查)
11. [清理](#11-清理)

---

## 1. 前提与架构

- 区域 ap-south-2，可用区 ap-south-2b；一个 2 台 trn2.48xlarge 的 capacity block（本次 `cr-03a4e6edbb3f4a771`）。
- 开发机：同 VPC、同子网、同安全组的一台 trn2.48xlarge，有 `/mnt/nvme/{sglang-neuron, work, models}`、`/opt`
  下的 Neuron venv、`/source-repos`（torch-neuronx 的 editable 源码），实例角色 `ec2-role`（可调 EC2 / SSM）。
- 每台 trn2.48xlarge：64 个逻辑 NeuronCore（LNC2），16 张网卡、最多 16 个 EFA 接口，4 块实例存储 NVMe。
- 部署：
  ```
  client ──> balancer (prefill 节点 :8000) ──┬──> prefill server  TP64 EP64 (prefill 节点 :30000, bootstrap :8998)
                                             └──> decode  server  TP64 EP64 + DFly (decode 节点 :30001)
  prefill ══ KV over EFA (Mooncake 或 NIXL LIBFABRIC，HBM → HBM) ══> decode
  ```
- Hy4 的 KV 是模型自己的 context-parallel store（`models/hy4/kv.py`）：每层一个 MLA 行张量 `[rows, 576]`，每个
  full 层一个 index key 张量 `[rows, 128]`，每个 1024-token 的 page 在每个 rank 上是 16 行、一段连续内存。两边
  attention TP 相同（64）、切分相同，所以 prefill rank r 的 page 直接写进 decode rank r 的对应行。PD 传输用的
  regions 由 accounting pool 的 `get_contiguous_buf_infos` / `get_kv_layer_ids` 上报（78 个 latent + 21 个 index =
  99 个 entry）。

## 2. 启动实例

### 2.1 网卡：16 个 EFA 接口

每张网卡一个 EFA 接口；第 0 张的 DeviceIndex 为 0，其余为 1（`nics.json`）：

```bash
python3 - <<'EOF' > nics.json
import json
nics = [{"NetworkCardIndex": i, "DeviceIndex": 0 if i == 0 else 1, "InterfaceType": "efa",
         "SubnetId": "subnet-0eb3568afbabb2c86", "Groups": ["sg-019c6da2b1582a74d"],
         "DeleteOnTermination": True} for i in range(16)]
print(json.dumps(nics))
EOF
```

### 2.2 user-data：实例存储 RAID0 挂到 /mnt/nvme

路径必须和开发机一致（`/mnt/nvme`）：编译缓存的 key 和插件安装路径有关。见 `userdata.sh`（mdadm RAID0 →
ext4 → mount，建 `ssm-user`，最后 touch `/mnt/nvme/.ready`）。

### 2.3 启动（先 dry-run）

```bash
R=ap-south-2
aws ec2 run-instances --region $R --dry-run --image-id ami-03ee0aee4a81418f5 --instance-type trn2.48xlarge --count 2 \
  --instance-market-options '{"MarketType":"capacity-block"}' \
  --capacity-reservation-specification '{"CapacityReservationTarget":{"CapacityReservationId":"cr-03a4e6edbb3f4a771"}}' \
  --network-interfaces file://nics.json --iam-instance-profile Arn=arn:aws:iam::<account>:instance-profile/ec2-role \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":300,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --user-data file://userdata.sh --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=hy4-pd}]'
# 返回 DryRunOperation 即参数正确；去掉 --dry-run 真正启动
```

- AMI 用开发机同一个（Deep Learning AMI Neuron, Ubuntu 24.04, SDK 2.33），EFA 驱动和 libfabric 已在
  `/opt/amazon/efa`。
- 根盘默认 100 GB，不够放 9 GB 的 venv 等，给 300 GB。
- 本次：prefill `i-03a3d5f11d86e5db3`（172.31.42.175），decode `i-0aa05f3deb2f888f1`（172.31.37.223）。

### 2.4 公网出口：Elastic IP

**多网卡的实例不能自动分配公网 IP**。子网只有 IGW、没有 NAT，SSM agent 连不上 AWS。给每台主网卡（card 0,
device 0）挂一个 Elastic IP：

```bash
for eni in <prefill 主网卡 eni> <decode 主网卡 eni>; do
  A=$(aws ec2 allocate-address --region $R --domain vpc --query AllocationId --output text)
  aws ec2 associate-address --region $R --allocation-id $A --network-interface-id $eni
done
aws ec2 wait instance-running --region $R --instance-ids <两个 id>
# 等 SSM 在线
aws ssm describe-instance-information --region $R --filters Key=InstanceIds,Values=<id1>,<id2> \
  --query 'InstanceInformationList[].PingStatus'
```

### 2.5 远程执行：pd.sh

`pd.sh <prefill|decode|both> '<命令>' [超时秒]`：通过 SSM 在节点上以 root 执行。命令先 base64 编码再在节点上
解码执行，否则 SSM 外层 shell 会提前展开 `$(...)`、`$VAR`（第一版就因此出错）。

注意：SSM 命令结束时会杀掉它启动的后台进程，长任务一律用 `systemd-run --unit=<名字> ...` 启动。

## 3. 网络与 EFA 配置

**安全组必须同时允许"到自身"的入站和出站全部流量。** 默认安全组通常只有入站自引用规则，出站是 0.0.0.0/0。
EFA 的 SRD 流量不是 IP 流量，0.0.0.0/0 的出站规则不覆盖它。缺这条规则时 TCP / ping 都正常，但 EFA 不通
（`fi_pingpong`："Unreachable remote - never received a response"；serving："Decode transfer failed"）。

```bash
aws ec2 authorize-security-group-egress --region $R --group-id sg-019c6da2b1582a74d \
  --ip-permissions '[{"IpProtocol":"-1","UserIdGroupPairs":[{"GroupId":"sg-019c6da2b1582a74d","Description":"EFA"}]}]'
```

## 4. 测试 EFA

```bash
# 两台：EFA 设备（trn2.48xlarge 上 rdm domain 应为 32）、Neuron 设备 16
./pd.sh both 'export PATH=/opt/amazon/efa/bin:$PATH; fi_info -p efa -t FI_EP_RDM | grep -c "domain:"; ls /dev/neuron* | wc -l'
# 跨节点 ping-pong：server 用 systemd-run 启动（不然 SSM 结束就被杀），输出用 stdbuf 取消缓冲
./pd.sh prefill 'systemd-run --unit=pingpong --property=StandardOutput=file:/mnt/nvme/pp.log \
  stdbuf -o0 /opt/amazon/efa/bin/fi_pingpong -p efa -e rdm -I 2000 -S all'
./pd.sh decode 'timeout 100 stdbuf -o0 /opt/amazon/efa/bin/fi_pingpong -p efa -e rdm -I 2000 -S all <prefill ip> | tail -12'
```

本次结果：小消息单向约 10 µs，8 KB 单 QP ping-pong 658 MB/s（`results_qwen06_1p1d/efa_pingpong_client.txt`）。

## 5. 分发环境和模型

开发机在私网 IP 上起一个文件服务（只有同安全组成员能访问），节点按相同路径拉取：

```bash
# 开发机
nohup python3 -m http.server 8090 --bind <开发机私网 ip> --directory /mnt/nvme &
cd / && tar cf /mnt/nvme/work/dist/optvenv.tar opt/aws_neuronx_venv_torch_neuronx_2_14_0
tar cf /mnt/nvme/work/dist/source_repos.tar source-repos
cd /mnt/nvme && tar cf work/dist/code.tar --exclude='work/hy4/cache_*' --exclude='*/__pycache__' \
  work/venv work/sglang work/nixl_overlay work/hy4 sglang-neuron
find models/Hy4-FP8 -type f -printf '%s %p\n' > work/dist/hy4_files.txt
# 节点（都用 systemd-run，可重跑：大小一致的文件跳过）
./pd.sh both 'curl -s http://<开发机 ip>:8090/work/dist/fetch_env.sh -o /mnt/nvme/fetch_env.sh; chmod +x /mnt/nvme/fetch_env.sh;
  systemd-run --unit=fetch-env --property=StandardOutput=file:/mnt/nvme/fetch_env.log /mnt/nvme/fetch_env.sh'
./pd.sh both 'curl -s http://<开发机 ip>:8090/work/dist/fetch_model.sh -o /mnt/nvme/fetch_model.sh; chmod +x /mnt/nvme/fetch_model.sh;
  systemd-run --unit=fetch-model --property=StandardOutput=file:/mnt/nvme/fetch_model.log /mnt/nvme/fetch_model.sh'
```

- 环境必须完整复制：`/opt` 的 Neuron venv（9 GB，可能被改过）、`/source-repos`（torch-neuronx 是 editable
  安装）、`/mnt/nvme/work/{venv,sglang}`、`sglang-neuron`（editable）。
- 模型 733 GiB，两台同时拉，32 路并发每台约 0.6 GB/s，约 20 分钟。
- 代码有改动时，只打包改动的文件（例如 `tar cf code_n.tar sglang-neuron/sglang_neuron/models/hy4/kv.py`）再拉。

## 6. 传输后端

sglang-neuron 支持两个 PD KV 传输后端，都是 EFA 上 HBM → HBM 直写：

### 6.1 Mooncake（sgl-project/sglang-neuron#1）

- 本分支 cherry-pick 了 #1 的两个提交，并把 Hy4 加入 `disaggregation/mooncake.py` 的 `SERVED_PACKAGES`（Hy4 两边
  attention TP 相同，entry 按 `kv_layer_ids` 配对）。
- 安装 transfer engine（开发机下载 wheel，节点从文件服务装）：
  ```bash
  python3 -m pip download "mooncake-transfer-engine-non-cuda>=0.3.14" --no-deps -d /mnt/nvme/work/dist
  ./pd.sh both '/mnt/nvme/work/venv/bin/python -m pip install --no-deps <wheel>; python3 -c "import mooncake.engine"'
  ```
  依赖只有 aiohttp / requests / msgpack（venv 里已有）；和 torch-neuronx 同进程加载没有 protobuf 冲突。
- 启动要求（`launch_problems` 会检查）：`LD_LIBRARY_PATH` 含 `/opt/aws/neuron/lib`（否则 libfabric 注册 HBM 报
  `fi_mr_regattr -> ENOSYS`）；不能设 `MC_FORCE_TCP=1`；`MOONCAKE_PROTOCOL` 由插件设为 `efa`。
  `pd_hy4.sh` 在 `PD_BACKEND=mooncake` 时设置好这些。

### 6.2 NIXL（LIBFABRIC）

- 不要 pip 装 PyPI 的 nixl（它的 libnixl 链了 etcd/gRPC，与 torch-neuronx 的 protobuf 冲突）。用仓库脚本构建
  带补丁的 overlay：
  ```bash
  W=/mnt/nvme/work/nixl_wheels; python3 -m pip download --no-deps --require-hashes -r <TOOL_WHEELS+META_WHEEL> -d $W
  NIXL_WHEELS=$W bash tools/build_nixl.sh /mnt/nvme/work/nixl_overlay
  ```
  运行时 `PYTHONPATH=/mnt/nvme/work/nixl_overlay:$PYTHONPATH`，`unset NIXL_PLUGIN_DIR`，
  `SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC`。

## 7. 启动 prefill、decode、balancer

`pd_hy4.sh <prefill|decode|balancer> <prefill ip> <decode ip>`：两边模型参数完全一致（page 1024、context 512K、
pool 3.1M tokens、TP64 EP64、开发机的最优 decode flags），decode 额外开 DFly。

```bash
B=mooncake   # 或 nixl
./pd.sh decode  "systemd-run --unit=pdh-decode  --setenv=PD_BACKEND=$B --property=StandardOutput=file:/mnt/nvme/decode.log \
  /mnt/nvme/work/hy4/pd_hy4.sh decode  172.31.42.175 172.31.37.223"
./pd.sh prefill "systemd-run --unit=pdh-prefill --setenv=PD_BACKEND=$B --property=StandardOutput=file:/mnt/nvme/prefill.log \
  /mnt/nvme/work/hy4/pd_hy4.sh prefill 172.31.42.175 172.31.37.223"
# 两边日志都出现 "The server is fired up and ready to roll!" 之后：
./pd.sh prefill "systemd-run --unit=pdh-balancer --setenv=BALANCER_HOST=172.31.42.175 \
  --property=StandardOutput=file:/mnt/nvme/balancer.log /mnt/nvme/work/hy4/pd_hy4.sh balancer 172.31.42.175 172.31.37.223"
```

- `AGENTX=1`：加 `--served-model-name tencent/Hy4-FP8 --enable-metrics --enable-cache-report`，**两边**的设备
  sampler 都用 `topk_topp`（prefill 也会采样每个请求的第一个 token，用 greedy 会拒绝 top-k/top-p 请求，见 §10）。
- 启动时脚本会给编译缓存重新打戳（`compile_cache stamp`），否则同步了插件其他文件后缓存 guard 会拒绝启动（§10）。
- balancer 默认只监听 127.0.0.1；AgentX 客户端在开发机上跑时设 `BALANCER_HOST=<prefill ip>`。

## 8. 正确性检查

```bash
# 6 个提示词，128 token，greedy；与同 flags 的单机服务逐字比较
./pd.sh prefill 'source /mnt/nvme/work/hy4/env.sh; cd /mnt/nvme/work/hy4;
  python tests/dfly_check.py 8000 /mnt/nvme/check.json 128'
```

本次：NIXL 与 Mooncake 两个后端的 6 条输出都与单机逐字相同，DFly 接受长度相同（2.61 3.87 2.78 6.73 2.56 6.4），
请求耗时一致（`results_hy4_1p1d/`、`results_hy4_1p1d_mooncake/`）。

确认确实走 Mooncake 的 EFA transport（而不是退回 TCP）：

```bash
./pd.sh both 'L=/mnt/nvme/prefill.log; grep -o "disaggregation_transfer_backend.\{0,14\}" $L | head -1;
  grep "protocol: efa" $L | head -1;                       # transfer engine 日志
  p=$(pgrep -f "sglang::scheduler" | head -1); tr "\0" "\n" < /proc/$p/environ | grep -E "MOONCAKE_PROTOCOL|NEURON_RT_MAP_HBM"'
# EFA 硬件计数器：每张 EFA 卡都有 RDMA write 字节数
./pd.sh prefill 'cat /sys/class/infiniband/*/ports/1/hw_counters/rdma_write_bytes'
```

小模型验证（先确认链路）：Qwen3-0.6B 1P1D（`pd_qwen.sh`，examples/qwen3_0p6b），17×23 得 `\boxed{391}`，GSM8K 前
100 题 0.44（`results_qwen06_1p1d/`）。

## 9. 跑 AgentX

按 agentx-bench 的规范（`CONTRIBUTING.md`）：

- **每个并发点重新部署服务**：停掉上一轮的 balancer / prefill / decode（以及残留的 `sglang::` 进程），重新启动、
  等 `/health`，再起 balancer、跑客户端。不要在热服务上连跑多个并发，也不要手动清缓存。
- 每个点固定 3600 s 测量（另有 warmup 和 drain，单点 1–1.5 小时），种子 42，用 `client/run_point.sh` 的默认值。
- 错误率 ≤ 10%（客户端会校验）。
- **TP 字段填芯片数**：trn2.48xlarge 是 16 颗芯片，LNC=2 下 Neuron TP64 填 `TP=16`。真实的 Neuron TP / EP / LNC
  写在 metadata 的 `notes` 里。填错了不必重跑：改好 point env，用 `client/import_point.sh <point.env>
  <run>/results/aiperf_artifacts <run>/client.log` 重新导入。

`agentx_pd2.sh`（在开发机上运行，客户端也在开发机上；每个点按上面的流程重部署两边服务）：

```bash
PD_BACKEND=mooncake CONCS="1 4" nohup ./agentx_pd2.sh > agentx_pd.out 2>&1 &
# 单机对照（同一 trace）：开发机上 trn2_agentx_dfly.sh，同样每个点重启服务
```

PD point env 的关键字段：

```
SERVER_URL=http://<prefill ip>:8000                     # balancer
SERVER_METRICS_URLS=http://<prefill ip>:30000/metrics,http://<decode ip>:30001/metrics   # 逗号分隔，两边都抓
DISAGG=true
KV_P2P_TRANSFER=mooncake
PREFILL_NUM_WORKERS=1  PREFILL_TP=16  PREFILL_EP=16
DECODE_NUM_WORKERS=1   DECODE_TP=16   DECODE_EP=16
TP=16 EP=16 PP=1 DCP=1 PCP=1                            # DISAGG=true 时 PP/DCP/PCP 必须为 1
```

进度监控：客户端日志里 `Phase warmup ... started/complete`、`Phase profiling ... started`（开始 3600 s）、
`Validated aiperf request error rate`、`Point complete`；服务端看 decode 日志的 `Hy4 DFly: ... last 64 steps`、
prefill 日志的 `Prefill batch`，以及两边的 `Decode transfer failed` / `Traceback`。KV 传输耗时在 metrics 里：
`sglang:kv_transfer_latency_ms`、`kv_transfer_total_mb`、`kv_transfer_speed_gb_s`（aiperf 的
`server_metrics_export.json`）。

结果（`results_agentx/`，每个点 0% 错误）：

| AgentX | 单机 c1 | PD c1 | 单机 c4 | PD c4 |
|---|---|---|---|---|
| output tok/s | 24.7 | 24.7 | 32.9 | **36.9** |
| ITL avg / p50 / p90 (ms) | 33.5 / 28.2 / 42.3 | 29.8 / 26.4 / 37.5 | 102.3 / 88.0 / 126.6 | **73.1 / 73.9 / 101.1** |
| TTFT avg / p50 / p90 (s) | 8.0 / 3.6 / 16.8 | 8.2 / 4.2 / 15.7 | 22.1 / 4.8 / 62.8 | **16.6 / 4.6 / 39.7** |
| request latency avg / p90 (s) | 39.5 / 95.8 | 36.7 / 83.7 | 96.1 / 293.5 | **78.2 / 248.0** |
| 芯片数 | 16 | 32 | 16 | 32 |

- conc1 时 PD 几乎没有收益（只有一个会话，prefill 很少和 decode 争抢）。
- conc4 时 decode 不再被其他会话的长 prefill 阻塞：ITL 降 20–29%，TTFT p90 降 37%，请求延迟降 19%；但用了两倍
  芯片，每芯片吞吐更低。
- KV 传输不是瓶颈：c1 平均 145 ms（每 rank 约 273 MB，64 个 rank 合计约 160 GB/s），c4 平均 328 ms；TTFT 主要是
  prefill 算力（每个 1024-token chunk 约 1.2 s）。

## 10. 问题排查

| 现象 | 原因 | 处理 |
|---|---|---|
| SSM 执行报 `sudo: unknown user ssm-user` | user-data 还没跑完 | 等 `cloud-init status: done`；pd.sh 用 root 执行 |
| `fi_info: not found`、awk 引号报错 | SSM 外层 shell 提前展开了 `$(...)` | pd.sh 把命令 base64 编码后在节点解码执行 |
| 后台下载/服务在 SSM 命令结束后消失 | SSM 杀掉命令的进程组 | 长任务用 `systemd-run` |
| `fi_pingpong` "Unreachable remote"、serving "Decode transfer failed" | 安全组缺"到自身"的出站规则 | §3 |
| `fi_pingpong` 无输出 | stdout 被缓冲 | `stdbuf -o0` |
| Hy4 注册被拒："Layer metadata length must match transfer entries: src metadata=2 entries=99" | accounting pool 只上报了 1 个假层的 layer id | `Hy4AccountingPool.get_kv_layer_ids`（kv.py） |
| AgentX warmup 全部 400："requires temperature=0 or top_k=1 ... SAMPLER=greedy" | prefill 用了 greedy sampler | 两边都 `SGLANG_NEURON_SAMPLER=topk_topp`（AGENTX=1） |
| 启动报 "the compile cache ... was built from other NKI kernel sources" | 同步了插件文件，缓存戳不匹配（Hy4 kernel 没变） | 启动前 `python -m sglang_neuron.backend.compile_cache stamp <dir>` |
| prefill chunk 4096 / 2048 OOM（`NRT_RESOURCE`） | 512K context + 3.1M pool 占满 HBM | 保持 1024（更大 chunk 也不更快，prefill 是算力瓶颈） |
| decode："Request ... timed out after 300.0s in KVPoll.WaitingForInput" | 并发时多个冷会话在单个 prefill 上排队（4 × 128–380K token，约 9 分钟），超过 SGLang 默认 300 s | `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` / `..._BOOTSTRAP_TIMEOUT` 设为 3600（pd_hy4.sh 已设） |
| Mooncake 启动被拒："the model (...) moves its KV with NIXL only" | #1 只放行 Qwen3 / Qwen3-MoE / GLM-4 | 把模型包加入 `disaggregation/mooncake.py` 的 `SERVED_PACKAGES`（Hy4 已加） |
| `pkill -f <模式>` 把自己的 shell 也杀了（exit 144） | 模式出现在自己的命令行里 | 按 PID 杀，或用 `ps ... \| awk` 过滤掉自己 |
| AgentX 点名 / 每芯片指标不对 | TP 字段填了软件 TP（64） | 填芯片数（16），用 `import_point.sh` 重新导入 |

## 11. 清理

```bash
aws ec2 terminate-instances --region $R --instance-ids i-03a3d5f11d86e5db3 i-0aa05f3deb2f888f1
aws ec2 release-address --region $R --allocation-id eipalloc-0f7bb8f0dbfaf8fdd
aws ec2 release-address --region $R --allocation-id eipalloc-03ad86de243cb1a97
# 可选：删除 §3 加的出站规则
aws ec2 revoke-security-group-egress --region $R --group-id sg-019c6da2b1582a74d --security-group-rule-ids sgr-0070bd40e7cd928b5
```
