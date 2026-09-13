# Assemble the race3 fidelity + adaptive suite into one comparison table.
import json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

arms = ["control", "resmlp", "swiglu"]
F = {a: json.load(open(f"runs/race3_fidelity_{a}.json")) for a in arms}
A = {a: json.load(open(f"runs/race3_adaptive_{a}.json")) for a in arms}

print("=== fidelity (16,384 pos, SmolLM2 fineweb dev) ===")
print(f"{'arm':9s} {'recall@1':>8s} {'recall@32':>9s} {'rerank':>7s} {'round_rr':>8s} {'hit@8':>7s} {'mass32':>7s} {'PPLx':>6s}")
for a in arms:
    f = F[a]
    print(f"{a:9s} {f['topk_recall_vs_native_top1']['top-1']:8.4f} {f['topk_recall_vs_native_top1']['top-32']:9.4f} "
          f"{f['rerank_top1_agreement_vs_native']:7.4f} {f['round_proposal_32comp']['rerank_agreement']:8.4f} "
          f"{f['hitrate_mob_top32_catches_native']['top8']:7.4f} {f['cascade32_native_mass_captured']:7.4f} "
          f"{f['ppl_ratio_mob_over_native']:6.3f}")

print("\n=== adaptive (2,048 pos): round32 / roundflip(v3) / best adaptive grid ===")
for a in arms:
    d = A[a]
    b = d["round32_baseline"]
    rf = d["round32_plus_flips"]
    best = max(d["grid"], key=lambda g: g["recall"])
    print(f"{a:9s} round32: recall {b['recall']:.4f} hit8 {b['hit8']:.4f} rerank {b['rerank']:.4f}")
    print(f"{'':9s} roundflip: recall {rf['recall']:.4f} hit8 {rf['hit8']:.4f} rerank {rf['rerank']:.4f} "
          f"cand mean {rf['cand_mean']:.0f} p95 {rf['cand_p95']:.0f}")
    print(f"{'':9s} best-adaptive({best['mode']}): recall {best['recall']:.4f} rerank {best['rerank']:.4f} "
          f"cand mean {best['cand_mean']:.0f} p95 {best['cand_p95']:.0f}")
