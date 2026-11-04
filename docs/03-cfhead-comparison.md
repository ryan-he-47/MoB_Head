# Compact Learned Density Heads versus Coarse–Fine Output Approximation

## Abstract

We compare a learned mixture-of-Bernoullis head with the training-free CFHead
baseline on Qwen3.5-2B. At 32 candidates, mixture proposals exceed CFHead-T4
containment by 2.5 and 3.1 percentage points on English and Chinese, respectively.
At 256 candidates, CFHead-T4 is stronger across all four sources and captures
more native probability mass. Both methods take approximately 4.4 ms for the
measured GPU/CPU offload head path. MoB therefore approaches CFHead's measured
performance without surpassing its overall accuracy–cost tradeoff.

## 1. Experimental protocol

The backbone is the instruction-tuned Qwen3.5-2B with hidden size 2048 and
vocabulary 248,320. The learned head is a 49.84M-parameter SwiGLU MoB with K=128,
M=18, distilled for 224k updates. CFHead caches use seed 99 and ten k-means
iterations. Baseline code is available from the
[official CFHead repository](https://github.com/HigherIdeal/CFHead).

| Baseline setting | Tile width | Centroids | Candidate budget |
|---|---:|---:|---:|
| CFHead-T4 | 4 | 64 | 32 or 256 |
| CFHead-Q | 8 | 64 | 256 |
| CFHead-M | 8 | 4 | 8192 |

Teacher-forced evaluations use English, Chinese, code, and instruction development
text, with 16,384 paired positions per source. Containment is the fraction of
positions whose native argmax belongs to the candidate set. It differs from
fine-stage agreement when low-precision ties change the selected argmax.

Budget-256 proposal and distribution-shape studies use 2,048 paired positions
per source. Their values are not interchangeable with the full-slice figures.

## 2. Candidate recall and density

### 2.1 Fixed budget of 32 candidates

| Source | MoB mixture containment@32 | CFHead-T4 containment@32 | MoB normalized PPL ratio |
|---|---:|---:|---:|
| English | **0.9667** | 0.9415 | 2.17× |
| Chinese | **0.9420** | 0.9114 | 3.66× |
| Code | 0.9503 | **0.9623** | 2.30× |
| Instruction | 0.9845 | **0.9907** | 1.47× |

The advantage at low budget is domain-dependent. On the most diffuse quartile
of native predictions, MoB containment exceeds T4 by 5.8 points in English and
5.1 points in Chinese. These measurements do not imply native-density parity.

### 2.2 Fixed budget of 256 candidates

| Source | Exact MoB mixture top-256 | MoB scoreflip, at most 256 | CFHead-T4@256 |
|---|---:|---:|---:|
| English | 0.9810 | 0.9746 | **0.9976** |
| Chinese | 0.9683 | 0.9604 | **0.9902** |
| Code | 0.9741 | 0.9634 | **0.9966** |
| Instruction | 0.9932 | 0.9893 | **0.9990** |

Scoreflip emits a mean of 90–216 candidates depending on source. Exact mixture
ranking also trails CFHead, so candidate enumeration alone does not explain
the measured gap. The learned distribution itself ranks some native targets
below the cutoff.

## 3. Distribution coverage

For candidate set $C$, retained native mass is $P(C)=\sum_{t\in C}P(t)$.
Renormalizing the exact native distribution inside $C$ has total-variation
distance $1-P(C)$ from the original distribution.

| Source | CFHead-T4 retained mass | Exact MoB top-256 retained mass | MoB scoreflip retained mass |
|---|---:|---:|---:|
| English | 0.913 | 0.881 | 0.840 |
| Chinese | 0.904 | 0.855 | 0.824 |
| Code | 0.974 | 0.930 | 0.905 |
| Instruction | 0.988 | 0.969 | 0.951 |

These rounded T=1 measurements show a persistent coverage advantage for CFHead.
Largest omitted-token mass and complete native nucleus inclusion favor it as
well. Lowering temperature to 0.7 narrows but does not eliminate the differences.
The recorded study concerns candidate distribution shape, not a claim of
equivalent free-running stochastic generations.

## 4. Downstream and free-running evaluation

| Task | Native | CFHead-Q (256 candidates) | MoB cascade (about 86 candidates) |
|---|---:|---:|---:|
| MGSM English, n=250 | 67.2 | 66.4 | 62.0 |
| MGSM Chinese, n=250 | 48.0 | 52.4 | 43.2 |
| MMLU, n=100 | 60.0 | 60.0 | 57.0 |
| GSM8K, n=128 | 60.9 | 63.3 | 59.4 |
| ARC-Challenge, n=128 | 79.7 | 79.7 | 78.1 |

Entries are accuracy percentages. This downstream baseline is **Q**, not T4;
the table must not be labeled as a T4 benchmark. Sample sizes are small and
candidate budgets differ. Paired correctness flips show that small apparent
improvements over native are insufficient to establish a systematic quality gain.

Long-form greedy evaluation uses six prompts and 8,192 steps per prompt. MoB
candidate containment is 99.87%, but the native trajectory spends 90.3% of its
steps in repetition loops. Near-perfect self-conditioned containment therefore
does not establish equivalent held-out prediction quality.

## 5. Bandwidth and latency

The MoB head stores approximately 95.0 MiB of fp16 weights. CFHead-T4's index map
would occupy 95.3 MiB with ideal 6-bit packing, while the evaluated uint8 map
occupies approximately 127 MiB. These are distinct theoretical and runtime
storage quantities; an equal-bit-budget statement does not imply identical
implemented memory traffic.

On the RTX 4060 Laptop offload path, the GPU computes coarse information, a
small transfer crosses PCIe, and the CPU proposes/rescores candidates using
RAM-resident output rows:

| Head path | GPU | Transfer | CPU | Sum |
|---|---:|---:|---:|---:|
| MoB | 1.803 ms | 0.155 ms | 2.442 ms | **4.400 ms** |
| CFHead | 3.614 ms | 0.159 ms | 0.625 ms | **4.398 ms** |

End-to-end rates are 9.96 tokens/s for MoB offload, 10.27 for CFHead offload,
10.43 for native GPU, and 5.06 for naive native CPU offload. The split measurement
supports comparable head latency, not an end-to-end win over CFHead.

## 6. Conclusion

MoB achieves performance close to CFHead under several measured decoding
conditions, with low-budget advantages on English and Chinese. CFHead remains
the stronger overall acceleration baseline: it requires no distillation and
offers better wide-budget containment and distribution coverage. This project
concluded without exceeding that tradeoff. The archived implementation and
checkpoints document a viable learned density-head alternative, including a
normalized likelihood and a mode of operation that does not use native output
rows for decoding.
