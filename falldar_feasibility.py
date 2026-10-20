import argparse, io, os, zipfile
import numpy as np
import scipy.io
from scipy.fft import dctn, idctn

ROOT = os.path.dirname(os.path.abspath(__file__))
ZIP = os.path.join(ROOT, "FallDar.zip")
CACHE = os.path.join(ROOT, "falldar_cache.npz")
OUT = os.path.join(ROOT, "falldar_results")
POOL = 4
RATES = [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128]


def build_cache():
    X, y, groups, users = [], [], [], []
    outer = zipfile.ZipFile(ZIP)
    inner = zipfile.ZipFile(outer.open("falldar/denoisemat.zip"))
    infos = sorted((i for i in inner.infolist() if i.filename.endswith(".mat")), key=lambda i: i.header_offset)
    for k, info in enumerate(infos):
        _, cls, date, fname = info.filename.split("/")
        m = scipy.io.loadmat(io.BytesIO(inner.read(info)))
        csi = m["csi_data"]
        if csi.shape != (2000, 30, 3):
            print("  skipping file with unexpected shape", info.filename, csi.shape)
            continue
        X.append(csi.reshape(2000 // POOL, POOL, 30, 3).mean(1).astype(np.complex64))
        y.append(1 if cls == "fall" else 0)
        stem = fname[:-4]
        groups.append(f"{date}/{stem.rsplit('-', 1)[0] if cls == 'nonfall' else stem}")
        users.append(fname.split("-")[0])
        if k % 200 == 0:
            print(f"  loaded {k}/{len(infos)}")
    np.savez(CACHE, X=np.stack(X), y=np.array(y), groups=np.array(groups), users=np.array(users))


def zone(r, T):
    s = np.sqrt(r)
    return max(1, round(T * s)), max(1, round(30 * s))


def reconstruct(C_re, C_im, r):
    kt, ks = zone(r, C_re.shape[1])
    out = []
    for C in (C_re, C_im):
        Z = np.zeros_like(C)
        Z[:, :kt, :ks] = C[:, :kt, :ks]
        out.append(idctn(Z, axes=(1, 2), norm="ortho"))
    return out[0] + 1j * out[1]


def true_rate(r, T):
    kt, ks = zone(r, T)
    return kt * ks / (T * 30)


def features(Xc):
    a = np.abs(Xc).reshape(len(Xc), Xc.shape[1], -1)
    a = a - a.mean(1, keepdims=True)
    a = a / (a.std((1, 2), keepdims=True) + 1e-6)
    return a.transpose(0, 2, 1).astype(np.float32)


def train_eval(F, y, tr, va, te, seed, epochs):
    import torch, torch.nn as nn

    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Conv1d(90, 64, 7, 2, 3), nn.BatchNorm1d(64), nn.ReLU(),
        nn.Conv1d(64, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.ReLU(),
        nn.Conv1d(64, 128, 5, 2, 2), nn.BatchNorm1d(128), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 2),
    )
    w = torch.tensor(len(tr) / (2 * np.bincount(y[tr], minlength=2)), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    Ft, yt = torch.from_numpy(F), torch.from_numpy(y).long()

    def probs(idx):
        net.eval()
        with torch.no_grad():
            return torch.softmax(net(Ft[idx]), 1)[:, 1].numpy()

    best, best_state = 1e9, None
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        net.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 32)):
            opt.zero_grad()
            lossf(net(Ft[b]), yt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(Ft[va]), yt[va]).item()
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return probs(te)


def metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1)
    return dict(recall=rec, precision=prec, f1=2 * rec * prec / max(rec + prec, 1e-9),
                acc=float((pred == y).mean()), missed_falls=fn, false_alarms=fp)


def split(y, groups, seed):
    from sklearn.model_selection import StratifiedGroupKFold

    idx = np.arange(len(y))
    rest, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(idx, y, groups))
    a, b = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed + 100).split(rest, y[rest], groups[rest]))
    tr, va = rest[a], rest[b]
    assert not (set(groups[tr]) & set(groups[te])) and not (set(groups[va]) & set(groups[te]))
    return tr, va, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    if not os.path.exists(CACHE):
        print(">> First run: building cache from FallDar.zip ...")
        build_cache()
    d = np.load(CACHE)
    X, y, groups = d["X"], d["y"], d["groups"]
    T = X.shape[1]
    print(f">> Data {X.shape}  fall={int(y.sum())}  nonfall={int((1 - y).sum())}  recording groups={len(set(groups))}")

    C_re = dctn(X.real, axes=(1, 2), norm="ortho")
    C_im = dctn(X.imag, axes=(1, 2), norm="ortho")
    assert np.allclose(reconstruct(C_re, C_im, 1), X, atol=1e-4), "r=1 must be lossless"
    for a, b in zip(RATES[1:], RATES):
        assert all(p <= q for p, q in zip(zone(a, T), zone(b, T)))

    feats = {r: features(reconstruct(C_re, C_im, r)) for r in RATES}
    rate = {r: true_rate(r, T) for r in RATES}
    del C_re, C_im

    os.makedirs(OUT, exist_ok=True)
    fixed_rows, oracle_rows = [], []
    for seed in range(args.seeds):
        tr, va, te = split(y, groups, seed)
        yte = y[te]
        pred, pred2 = {}, {}
        for r in RATES:
            pred[r] = (train_eval(feats[r], y, tr, va, te, seed, args.epochs) > 0.5).astype(int)
            pred2[r] = (train_eval(feats[r], y, tr, va, te, seed + 1000, args.epochs) > 0.5).astype(int)
            m = metrics(yte, pred[r])
            fixed_rows.append(dict(seed=seed, rate=rate[r], CR=1 / rate[r], **m))
            print(f"  seed{seed} CR={1 / rate[r]:7.1f}  recall={m['recall']:.3f}  prec={m['precision']:.3f}  F1={m['f1']:.3f}")
        for rb in RATES:
            for rh in RATES:
                if rh < rb:
                    continue
                second = pred2[rb] if rh == rb else pred[rh]
                wrong = pred[rb] != yte
                final = np.where(wrong, second, pred[rb])
                base_fn = (pred[rb] == 0) & (yte == 1)
                oracle_rows.append(dict(
                    seed=seed, base_rate=rate[rb], refine_rate=rate[rh], control=rh == rb,
                    refined_frac=float(wrong.mean()),
                    avg_rate=rate[rb] + wrong.mean() * (rate[rh] - rate[rb]),
                    base_recall=metrics(yte, pred[rb])["recall"],
                    missed_falls_fixed=float((second[base_fn] == 1).mean()) if base_fn.any() else np.nan,
                    **metrics(yte, final)))

    import pandas as pd
    fx = pd.DataFrame(fixed_rows); orc = pd.DataFrame(oracle_rows)
    fx.to_csv(os.path.join(OUT, "fixed_rate.csv"), index=False)
    orc.to_csv(os.path.join(OUT, "oracle.csv"), index=False)
    fxm = fx.groupby("rate").mean(numeric_only=True).sort_index(ascending=False)
    orm = orc.groupby(["base_rate", "refine_rate"]).mean(numeric_only=True).reset_index()
    ctl = orm[orm.control.astype(bool)].set_index("base_rate")
    full = orm[(orm.refine_rate == 1) & ~orm.control.astype(bool)].set_index("base_rate")
    print("\n=== Fixed-rate (mean over seeds) ===")
    print(fxm[["CR", "recall", "precision", "f1", "missed_falls", "false_alarms"]].round(3).to_string())
    print("\n=== Oracle refinement (refine to full rate r_h=1) ===")
    print(full.reset_index()[["base_rate", "refined_frac", "avg_rate", "base_recall", "recall",
                              "missed_falls_fixed", "f1"]].round(3).to_string(index=False))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(fxm.index, fxm.recall, "o-", color="k", label="Fixed-rate")
    ax.scatter(ctl.index, ctl.recall, marker="x", color="k", s=40, label="Control: same rate, 2nd seed")
    for rh, g in orm[~orm.control.astype(bool)].groupby("refine_rate"):
        ax.scatter(g.avg_rate, g.recall, s=22, label=f"Oracle refine → r_h={rh:.3g}")
    ax.set_xscale("log"); ax.set_xlabel("Average transmitted rate (fraction of full CSI)")
    ax.set_ylabel("Fall recall"); ax.set_title("FallDar: fixed-rate vs oracle selective refinement")
    ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "rate_vs_recall.png"), dpi=150)
    cmp = pd.DataFrame({"CR": 1 / ctl.index, "base_recall": ctl.base_recall,
                        "oracle_same_rate": ctl.recall, "oracle_refine_to_full": full.recall,
                        "fixed_by_ctl": ctl.missed_falls_fixed, "fixed_by_refine": full.missed_falls_fixed})
    cmp["info_gain"] = cmp["oracle_refine_to_full"] - cmp["oracle_same_rate"]
    print("\n=== Control: how much of the gain comes from extra information? (info_gain = refine to full - same-rate 2nd seed) ===")
    print(cmp.round(3).to_string())
    cmp.to_csv(os.path.join(OUT, "control_comparison.csv"))
    print(f"\n>> Results written to {OUT}")


if __name__ == "__main__":
    main()
