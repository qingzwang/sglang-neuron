"""kv_fp8_test.py: kernels/kv_fp8.py pack-write and unpack-gather against the host's packing (OCP e4m3fn == trn2
e4m3 below 240), for the MLA rows (512 fp8 + 64 bf16) and the index keys (128 fp8)."""
import sys, torch, torch_neuronx  # noqa
sys.path.insert(0, "/mnt/nvme/sglang-neuron")
from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.models.hy4.kernels.kv_fp8 import gather_order, kv_pack_write, kv_unpack_gather
from sglang_neuron.utils.hardware_config import get_logical_nc_config


def pack(x, n8):
    return torch.cat([x[:, :n8].to(torch.float8_e4m3fn).view(torch.uint8), x[:, n8:].contiguous().view(torch.uint8)], 1)


def unpack(c, n8):
    return torch.cat([c[:, :n8].contiguous().view(torch.float8_e4m3fn).to(torch.bfloat16),
                      c[:, n8:].contiguous().view(torch.bfloat16)], 1)


ok = True
for D, n8, R, N in ((576, 512, 8192, 4096), (128, 128, 4096, 384), (576, 512, 2048, 128)):
    torch.manual_seed(D + N)
    x = torch.randn(N, D).to(torch.bfloat16)
    x[:, :n8] = (x[:, :n8] * 30).clamp(-224, 224)
    rows = torch.randperm(R)[:N].to(torch.int32)
    rows[::7] = R  # skipped slots
    W = n8 + 2 * (D - n8)
    pw = wrap_nki(kv_pack_write, alias_outputs_in_place=True)[get_logical_nc_config()]
    ug = wrap_nki(kv_unpack_gather)[get_logical_nc_config()]
    c0 = torch.zeros(R, W, dtype=torch.uint8)
    cdev = c0.to("neuron")
    f = torch.compile(lambda c, r, v: pw(c, r, v, n8=n8), backend="neuron", fullgraph=True, dynamic=False)
    f(cdev, rows.to("neuron"), x.to("neuron"))
    got = cdev.cpu()
    want = c0.clone(); keep = rows < R; want[rows[keep].long()] = pack(x[keep], n8)
    a = torch.equal(got, want)
    rq = torch.randperm(R)[:N].to(torch.int32); rq[::5] = R
    g = torch.compile(lambda c, r: ug(c, r, n8=n8), backend="neuron", fullgraph=True, dynamic=False)
    M = N // 128
    o = g(want.to("neuron"), gather_order(rq).to("neuron")).cpu()
    ref = torch.zeros(N, D, dtype=torch.bfloat16); k = rq < R; ref[k] = unpack(want[rq[k].long()], n8)
    b = torch.equal(o, ref)
    print(f"KVFP8 D={D} n8={n8} N={N}: pack_write exact {a} ({int((got != want).sum())} bytes differ), "
          f"unpack_gather exact {b} ({int((o != ref).any(1).sum())} rows differ)", flush=True)
    bad = (o != ref).any(1).nonzero().flatten()
    print("  bad slots p,m:", sorted(set((int(i) % 128, int(i) // 128) for i in bad[:20])), "of", len(bad))
    if len(bad):
        i = int(bad[0]); full = unpack(want, n8)
        hit = [j for j in range(R) if torch.equal(full[j], o[i])]
        print("  slot", i, "row wanted", int(rq[i]), "got row", hit[:3], "zero" if o[i].abs().sum() == 0 else "",
              "max diff", (o[i].float() - ref[i].float()).abs().max().item())
    ok &= a and b
print("KVFP8", "PASS" if ok else "FAIL")
