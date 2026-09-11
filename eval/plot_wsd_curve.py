# Plot dev_top1_agree / dev_nll trajectories: WSD (job 520825) vs cosine (519759),
# with WSD LR overlay and decay-onset marker. User observation 2026-09-03:
# agree plateaued ~0.72 through 100k-150k, climbed from decay start, still
# rising at the end. Output: runs/wsd_vs_cosine_curve.png
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    steps, agree, devnll = [], [], []
    lr_s, lr_v = [], []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if d.get("panel"):
            steps.append(d["step"])
            agree.append(d["dev_top1_agree"])
            devnll.append(d["dev_nll"])
        elif "lr" in d and "nll" in d:
            lr_s.append(d["step"])
            lr_v.append(d["lr"])
    return (np.array(steps), np.array(agree), np.array(devnll),
            np.array(lr_s), np.array(lr_v))


wsd = load(r"E:\Trash_things\llm_codec\runs\train_log_wsd.jsonl")
cos = load(r"E:\Trash_things\llm_codec\runs\p2_fineweb_full\train_log.jsonl")

# decay onset from the LR trace: first step AFTER the peak where lr < 0.999*peak
peak = wsd[4].max()
peak_i = int(np.argmax(wsd[4]))
below = np.nonzero(wsd[4][peak_i:] < 0.999 * peak)[0]
onset = wsd[3][peak_i + below[0]] if len(below) else wsd[3][-1]

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                         gridspec_kw={"height_ratios": [3, 2]})
ax = axes[0]
ax.plot(cos[0] / 1e3, cos[1], color="#888", lw=1.2, label="cosine 2e-3 (job 519759)")
ax.plot(wsd[0] / 1e3, wsd[1], color="#c0392b", lw=1.6, label="WSD 7e-4 (job 520825)")
ax.axvline(onset / 1e3, color="#c0392b", ls="--", lw=1, alpha=0.6)
ax.annotate(f"WSD decay onset\n~{onset/1e3:.0f}k", (onset / 1e3, 0.60),
            textcoords="offset points", xytext=(8, -30), fontsize=9, color="#c0392b")
ax.axhline(wsd[1][-1], color="#c0392b", ls=":", lw=0.8, alpha=0.5)
ax.set_ylabel("dev top-1 agreement")
ax.set_title("MoB K=128 distillation: dev agreement & NLL vs step (FineWeb-Edu 1B tokens)")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
ax2 = ax.twinx()
ax2.plot(wsd[3][::10] / 1e3, wsd[4][::10], color="#2980b9", lw=0.8, alpha=0.5)
ax2.set_ylabel("WSD LR", color="#2980b9")
ax2.tick_params(axis="y", labelcolor="#2980b9")

ax = axes[1]
ax.plot(cos[0] / 1e3, cos[2], color="#888", lw=1.2, label="cosine")
ax.plot(wsd[0] / 1e3, wsd[2], color="#c0392b", lw=1.6, label="WSD")
ax.axvline(onset / 1e3, color="#c0392b", ls="--", lw=1, alpha=0.6)
ax.set_ylabel("dev NLL (nats)")
ax.set_xlabel("step (k)")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
fig.tight_layout()
out = r"E:\Trash_things\llm_codec\runs\wsd_vs_cosine_curve.png"
fig.savefig(out, dpi=150)
print("saved", out)

# numeric summary: still-climbing check
s, a = wsd[0], wsd[1]
for lo, hi in [(100_000, 150_000), (150_000, 159_250), (159_250, 200_000), (200_000, 245_000)]:
    m = (s >= lo) & (s < hi)
    if m.any():
        print(f"{lo//1000}k-{hi//1000}k: agree {a[m][0]:.4f} -> {a[m][-1]:.4f} "
              f"(delta {a[m][-1]-a[m][0]:+.4f})")
last = a[-10:]
slope = np.polyfit(np.arange(10), last, 1)[0]
print(f"last-10-panel agree slope: {slope:+.5f}/kstep, final {a[-1]:.4f}")
n = wsd[2]
print(f"last-10-panel dev_nll slope: {np.polyfit(np.arange(10), n[-10:], 1)[0]:+.6f}/kstep, final {n[-1]:.4f}")
print(f"cosine final agree {cos[1][-1]:.4f} | WSD final agree {a[-1]:.4f}")
