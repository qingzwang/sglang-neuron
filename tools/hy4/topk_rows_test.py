"""topk_rows_test.py: kernels/select_topk.topk_rows against torch.topk (as sets; bf16 ties aside)."""
import sys, time, torch, torch_neuronx  # noqa
sys.path.insert(0, "/mnt/nvme/sglang-neuron")
from sglang_neuron.backend.stack import wrap_nki
sys.path.insert(0, "/mnt/nvme/work/hy4/wip"); from select_topk import topk_rows
from sglang_neuron.utils.hardware_config import get_logical_nc_config

ok = True
for R, K, k in ((7, 8208, 256), (28, 2064, 256), (16, 16384, 2048), (1024, 16400, 256)):
    torch.manual_seed(R + K)
    s = torch.randn(R, K).to(torch.bfloat16)
    kern = wrap_nki(topk_rows)[get_logical_nc_config()]
    f = torch.compile(lambda x: kern(x, k=k), backend="neuron", fullgraph=True, dynamic=False)
    v, i = (t.cpu() for t in f(s.to("neuron")))
    rv, ri = torch.topk(s.float(), k, dim=-1)
    # the kernel's values are the scores at its indices, and its k-th value is torch's
    gathered = s.gather(1, i.long().clamp(0, K - 1))
    a = torch.equal(gathered, v) and bool((i >= 0).all() and (i < K).all())
    b = torch.equal(v.float().sort(-1).values, rv.sort(-1).values)
    uniq = all(len(set(row.tolist())) == k for row in i)
    t = time.time()
    for _ in range(10): r = f(s.to("neuron"))
    r[0].cpu()
    print(f"TOPK R={R} K={K} k={k}: values at indices {a}, same top-k values {b}, distinct {uniq}, "
          f"{(time.time() - t) / 10 * 1e3:.2f} ms wall", flush=True)
    ok &= a and b and uniq
print("TOPK", "PASS" if ok else "FAIL")
