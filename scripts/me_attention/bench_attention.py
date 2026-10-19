"""Param counts + fwd/bwd step time for the attention family (not a test). Usage: PYTHONPATH=src python this.py [device]"""
import sys, time
import torch
from me_attention.models import attention as A

dev = sys.argv[1] if len(sys.argv) > 1 else "mps"
meta = A.synthetic_meta(n_features=192, n_ex=8, seed=0, step=10.0)
F, E, M = meta["n_features"], len(meta["excitations"]), len(meta["emissions"])
print(f"F={F} E={E} M={M} device={dev}")
variants = {
    "smartattention_v0 (default)": ("smartattention_v0", {}),
    "smartattention_v0 discard": ("smartattention_v0", dict(edge_mode="discard")),
    "meas_tokens (default)": ("meas_tokens", {}),
    "meas_tokens relbias": ("meas_tokens", dict(relbias=True)),
    "meas_tokens gated": ("meas_tokens", dict(gated=True)),
    "meas_tokens d128 L4 r4": ("meas_tokens", dict(dim=128, depth=4, mlp_ratio=4)),
    "axial (default)": ("axial", {}),
    "axial relbias": ("axial", dict(relbias=True)),
    "axial d128 L3 r3": ("axial", dict(dim=128, depth=3, mlp_ratio=3)),
    "ex_tokens (default)": ("ex_tokens", {}),
    "ex_tokens conv sub4": ("ex_tokens", dict(ex_embed="conv", sub_tokens=4)),
    "ex_tokens d128 L4 r4": ("ex_tokens", dict(dim=128, depth=4, mlp_ratio=4)),
    "spatial_attn mlp P5": ("spatial_attn", dict(spec_encoder="mlp")),
    "spatial_attn ex P5": ("spatial_attn", dict(spec_encoder="ex")),
    "spatial_attn meas P5": ("spatial_attn", dict(spec_encoder="meas")),
}
B = 256
for label, (name, cfg) in variants.items():
    m = A._FACTORIES[name](meta, cfg).to(dev).train()
    n = m.n_params()
    x = torch.randn(B, 5, 5, F, device=dev) if name == "spatial_attn" else torch.randn(B, F, device=dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    def step():
        loss = m(x).logsumexp(1).mean() + m.extra_loss()
        opt.zero_grad(); loss.backward(); opt.step()
    for _ in range(3): step()
    if dev == "mps": torch.mps.synchronize()
    t = time.time()
    for _ in range(10): step()
    if dev == "mps": torch.mps.synchronize()
    print(f"{label:30s} params={n/1e3:8.1f}k  step(B={B})={1e3*(time.time()-t)/10:8.1f} ms")
