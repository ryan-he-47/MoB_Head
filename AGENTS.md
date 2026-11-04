# MoB_Head repository conventions

This is an archived research implementation of semantic-code mixture output
heads. Read `docs/01-method.md` and `docs/05-reproduction.md` before changing code.
Document corrections in the relevant technical report; preserve reported
checkpoint identities and evaluation conditions.

- Run entry points as modules from the repository root.
- Preserve head/checkpoint compatibility, token-ID code mappings, and fp32
  log-space likelihood calculations.
- Keep weights, data caches, credentials, local archives, and run directories
  out of Git. Model artifacts are published separately using explicit manifests.
- Compare normalized perplexity on matched positions. Distinguish candidate
  recall, reranking fidelity, native-head likelihood, and independent decoding.
- Do not alter architecture or experimental hyperparameters as part of packaging.
- Source and Markdown use UTF-8; Python and Slurm files use LF line endings.
- Use CPU-only tests for artifact export and loading; never start training during
  documentation or packaging work. Cluster computation belongs in Slurm jobs.
- Check `git status`, preserve unrelated local assets, and use explicit file
  allowlists for publication. Do not rewrite repository history.
- Original source is MIT-licensed. Keep third-party and model-specific licenses
  separate and retain required attribution.
