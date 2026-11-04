# Technical reports

MoB_Head studies a semantic binary-code representation of the language-model output
space. The reports describe the implemented method, completed experiments, and
the comparison with CFHead.

| Report | Contents |
|---|---|
| [Semantic codes and mixture density heads](01-method.md) | Mathematical formulation, balanced codebook construction, distillation, decoding, normalization |
| [Distillation and architecture study](02-distillation.md) | Models, corpora, optimization, head ablations, normalized perplexity and efficiency |
| [Comparison with CFHead](03-cfhead-comparison.md) | Candidate budgets, distribution coverage, downstream tasks, offload latency, conclusions |
| [Native pretraining](04-native-pretraining.md) | Controlled small-model experiment establishing end-to-end trainability |
| [Reproduction and model artifacts](05-reproduction.md) | Software, commands, checkpoint format, loading and evaluation |

Machine-readable measurements are in [`results/`](../results/).
