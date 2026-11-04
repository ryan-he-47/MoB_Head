"""Render paired pretraining likelihoods from released numerical measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "results/native-pretraining.json").read_text())
    curves = {name: {row["step"]: row["dev_nll"] for row in rows}
              for name, rows in data.items()}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for name, color in (("native", "#334155"), ("mob", "#b45309")):
        steps = sorted(curves[name])
        axes[0].plot(np.array(steps) * 16 * 512 / 1e6,
                     [curves[name][step] for step in steps], label=name, color=color)
    shared = sorted(set(curves["native"]) & set(curves["mob"]))
    ratio = [np.exp(curves["mob"][step] - curves["native"][step]) for step in shared]
    axes[1].plot(np.array(shared) * 16 * 512 / 1e6, ratio, color="#0369a1")
    axes[1].annotate(f"{ratio[-1]:.3f}×", (shared[-1] * 16 * 512 / 1e6, ratio[-1]),
                     xytext=(-50, 16), textcoords="offset points", color="#0369a1")
    axes[0].set_ylabel("Development NLL (nats)")
    axes[1].set_ylabel("MoB / native perplexity")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.set_xlabel("Training tokens (millions)")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(root / "docs/figures/native-pretraining.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
