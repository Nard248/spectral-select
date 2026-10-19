"""Config lists per stage (dev_explore.py)."""
import itertools


def pp_name(p):
    return f"{p['norm']}-{p['norm_scope'][:2]}-{'log' if p['log1p'] else 'lin'}"


def pp_grid():
    out = []
    for log in (False, True):
        out.append(dict(norm="none", norm_scope="global", log1p=log))
    for norm, sc, log in itertools.product(("l2", "area"), ("global", "per_ex"), (False, True)):
        out.append(dict(norm=norm, norm_scope=sc, log1p=log))
    return out


PP_FEW = [dict(norm="none", norm_scope="global", log1p=False), dict(norm="area", norm_scope="global", log1p=False),
          dict(norm="area", norm_scope="global", log1p=True)]


def s1():
    out = []
    for m in ("lr", "hgb", "mlp"):
        for p in pp_grid():
            kind = ("nn:" if m == "mlp" else "sk:") + m
            out.append(dict(name=f"{m}|{pp_name(p)}", kind=kind, cfg=dict(p)))
    for m, extra in (("rbfsvm", dict(max_train=10000)), ("knn", {}), ("cnn1d", {}),
                     ("spectralformer", dict(dim=32, depth=3)), ("bandvit", {})):
        for p in (PP_FEW[:2] if m in ('spectralformer', 'bandvit') else PP_FEW):
            kind = ("nn:" if m in ("cnn1d", "spectralformer", "bandvit") else "sk:") + m
            out.append(dict(name=f"{m}|{pp_name(p)}", kind=kind, cfg={**p, **extra}))
    return out


BP = dict(norm="area", norm_scope="global", log1p=True)          # best dev preprocessing (stage 1)
SM = dict(dim=32, depth=2, heads=2)                              # reduced size for the expensive 149-token models


def s2():
    """Existing attention variants (area-gl-log)."""
    out = []
    def add(n, kind, **kw):
        out.append(dict(name=f"{n}|bp", kind=kind, cfg={**BP, **kw}))
    add("v0", "att:smartattention_v0")
    add("v0+sn", "att:smartattention_v0", aug="scale+noise")
    add("ex_tokens", "att:ex_tokens")
    add("ex_tokens+sn", "att:ex_tokens", aug="scale+noise")
    add("ex_tokens+sub2conv", "att:ex_tokens", sub_tokens=2, ex_embed="conv")
    add("ex_tokens+mean", "att:ex_tokens", pool="mean")
    add("axial_small", "att:axial", dim=32, depth=1, heads=2)
    add("meas_small", "att:meas_tokens", **SM)
    add("meas_small+sn", "att:meas_tokens", aug="scale+noise", **SM)
    add("meas_small+relbias", "att:meas_tokens", relbias=True, **SM)
    add("meas_small+meanpool", "att:meas_tokens", pool="mean", **SM)
    add("meas_small+drop3", "att:meas_tokens", token_drop=0.3, **SM)
    return out


def _hyb(n, mode="inner", kind="v2:hybrid", **kw):
    return dict(name=f"{kind.split(':')[1]}{'' if mode=='inner' else '[full]'}|{n}", kind=kind, mode=mode, cfg={**BP, **kw})


LOGC = dict(norm="none", norm_scope="global", log1p=True, standardize=False)   # raw log-intensity (for centring)


def s3():
    """New mechanisms on the hybrid conv-attention model and, as objective ablation, on the MLP."""
    out = []
    for kind in ("v2:hybrid", "nn:mlp"):
        for mode in ("inner", "full"):
            out.append(_hyb("base", mode, kind))
            out.append(_hyb("aug_sn", mode, kind, aug="scale+noise"))
            out.append(_hyb("exscale0.4", mode, kind, aug_ex_scale=0.4))
            out.append(_hyb("exscale0.8", mode, kind, aug_ex_scale=0.8))
            out.append(_hyb("exscale0.4+smooth0.5", mode, kind, aug_ex_scale=0.4, aug_smooth=0.5))
            out.append(_hyb("exdrop0.25", mode, kind, ex_drop=0.25))
            out.append(_hyb("supcon0.3", mode, kind, supcon_w=0.3))
            for c in ("pixel", "ex", "both"):
                out.append(dict(name=f"{kind.split(':')[1]}{'' if mode=='inner' else '[full]'}|center-{c}", kind=kind, mode=mode,
                                cfg={**LOGC, "center": c}))
        for n, kw in (("mixup0.4", dict(mixup=0.4)), ("adv1.0", dict(adv_w=1.0)), ("supcon0.3+mixup0.4", dict(supcon_w=0.3, mixup=0.4))):
            out.append(_hyb(n, "full", kind, **kw))
    return out


def s3a():   # MLP objective ablation (cheap)
    return [c for c in s3() if c["kind"] == "nn:mlp"]


def s3b():   # hybrid, prioritised for the time box
    h = {c["name"]: c for c in s3() if c["kind"] == "v2:hybrid"}
    order = ["hybrid|base", "hybrid|aug_sn", "hybrid|center-pixel", "hybrid|supcon0.3", "hybrid|exdrop0.25",
             "hybrid[full]|base", "hybrid[full]|aug_sn", "hybrid[full]|supcon0.3", "hybrid[full]|adv1.0",
             "hybrid[full]|mixup0.4", "hybrid|exscale0.4", "hybrid|center-ex", "hybrid|exscale0.4+smooth0.5",
             "hybrid[full]|center-pixel", "hybrid[full]|exdrop0.25", "hybrid[full]|supcon0.3+mixup0.4"]
    return [h[n] for n in order]
