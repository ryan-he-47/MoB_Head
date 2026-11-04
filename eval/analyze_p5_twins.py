# P5 twin mid-run comparison (CPU-only analysis, no GPU on this machine).
# Reads the two pulled train_log.jsonl files, aligns by STEP (not wall time),
# prints matched-step comparison + health metrics, saves docs/p5_twin_curves.png.
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = r".\runs"


def load(path):
    train, panel = [], []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        (panel if d.get("panel") else train).append(d)
    return train, panel


nat_tr, nat_pa = load(RUNS + r"\p5_twin_native_log.jsonl")
mob_tr, mob_pa = load(RUNS + r"\p5_twin_mob_log.jsonl")

print(f"native: {len(nat_tr)} log pts -> step {nat_tr[-1]['step']}, {nat_tr[-1]['elapsed_s']/3600:.1f}h train time")
print(f"mob:    {len(mob_tr)} log pts -> step {mob_tr[-1]['step']}, {mob_tr[-1]['elapsed_s']/3600:.1f}h train time")
print(f"step rate: native {nat_tr[-1]['elapsed_s']/nat_tr[-1]['step']:.2f}s/step, "
      f"mob {mob_tr[-1]['elapsed_s']/mob_tr[-1]['step']:.2f}s/step "
      f"({nat_tr[-1]['elapsed_s']/nat_tr[-1]['step'] / (mob_tr[-1]['elapsed_s']/mob_tr[-1]['step']):.1f}x faster per step)")

# windowed train-loss means for matched steps
def win(tr, lo, hi):
    v = [d["loss"] for d in tr if lo <= d["step"] < hi]
    return float(np.mean(v)) if v else float("nan")

print("\n== windowed train loss (matched steps) ==")
print(f"{'window':>14} | {'native':>8} | {'mob':>8} | {'gap':>6}")
for lo in range(0, max(nat_tr[-1]["step"], mob_tr[-1]["step"]), 4000):
    n, m = win(nat_tr, lo, lo + 4000), win(mob_tr, lo, lo + 4000)
    if not (np.isnan(n) or np.isnan(m)):
        print(f"{lo:>7}-{lo+4000:<6} | {n:8.4f} | {m:8.4f} | {m-n:+6.3f}")

print("\n== dev panels ==")
nat_d = {d["step"]: d["dev_nll"] for d in nat_pa}
print(f"{'step':>7} | {'native dev':>10} | {'mob dev':>8} | {'gap':>6} | mob n_eff | dead | logZ")
for d in mob_pa:
    s = d["step"]
    nn = nat_d.get(s, float("nan"))
    print(f"{s:>7} | {nn:10.4f} | {d['dev_nll']:8.4f} | {d['dev_nll']-nn:+6.3f} | "
          f"{d['n_eff']:9.1f} | {d['dead']:4d} | {d['logZ']:7.4f}")

# PPL-ratio trajectory at matched panel steps
print("\n== implied PPL ratio (mob/native) at matched panels ==")
for d in mob_pa:
    s = d["step"]
    if s in nat_d:
        print(f"step {s:>6}: {np.exp(d['dev_nll'] - nat_d[s]):.3f}x")

# grad-norm tails
for name, tr in [("native", nat_tr), ("mob", mob_tr)]:
    g = np.array([d["gnorm"] for d in tr])
    print(f"\n{name} gnorm: mean {g.mean():.2f}, p95 {np.percentile(g, 95):.2f}, max {g.max():.2f}")

# figure
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot([d["step"] / 1e3 for d in nat_tr], [d["loss"] for d in nat_tr],
           color="#888", lw=0.6, alpha=0.5)
ax[0].plot([d["step"] / 1e3 for d in mob_tr], [d["loss"] for d in mob_tr],
           color="#c0392b", lw=0.6, alpha=0.5)
# smoothed
for tr, c, lab in [(nat_tr, "#333", "native"), (mob_tr, "#c0392b", "mob")]:
    s = np.array([d["step"] for d in tr]) / 1e3
    l = np.array([d["loss"] for d in tr])
    k = max(1, len(l) // 60)
    ax[0].plot(s[: len(l) // k * k].reshape(-1, k).mean(1),
               l[: len(l) // k * k].reshape(-1, k).mean(1), color=c, lw=1.8, label=lab)
ax[0].set_title("train loss (window-smoothed)")
ax[0].set_xlabel("step (k)"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot([d["step"] / 1e3 for d in nat_pa], [d["dev_nll"] for d in nat_pa],
           "o-", color="#333", label="native")
ax[1].plot([d["step"] / 1e3 for d in mob_pa], [d["dev_nll"] for d in mob_pa],
           "o-", color="#c0392b", label="mob")
ax[1].set_title("dev NLL panels")
ax[1].set_xlabel("step (k)"); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.suptitle("P5 from-scratch twins (212M, fp32, FineWeb) — mid-run")
fig.tight_layout()
out = r".\docs\p5_twin_curves.png"
fig.savefig(out, dpi=150)
print("\nsaved", out)
