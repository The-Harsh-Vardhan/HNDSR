# Audit 02: Notebook Comparison Against `HNDSR_Original.ipynb`

Date: 2026-03-30

Baseline: `HNDSR_Original.ipynb`

Compared notebooks:

- `HNDSR_Corrected.ipynb`
- `HNDSR_Kaggle.ipynb`
- `HNDSR_Databricks_MLflow.ipynb`
- `MLFlow/HNDSR_MLflow.ipynb`
- `HNDSR in Production/training/HNDSR_Kaggle_Updated.ipynb`

## Executive Verdict

The notebook lineage improves presentation much faster than it improves epistemic honesty.

- `HNDSR_Original.ipynb` is the broken, Colab-bound, synthetic-metric baseline.
- `HNDSR_Corrected.ipynb` is too small and too unevidenced to be trusted as a replacement.
- `HNDSR_Kaggle.ipynb` and `HNDSR in Production/training/HNDSR_Kaggle_Updated.ipynb` clean up the structure but keep the paper-level performance fantasy.
- `HNDSR_Databricks_MLflow.ipynb` improves engineering hygiene and documents more of the pipeline, but it still repeats the same inflated Stage 3 claims.
- `MLFlow/HNDSR_MLflow.ipynb` is the only notebook that seriously confronts the real bugs and reports measured results that match the rest of the repo reality. It is the least-bad source of truth, not a pristine one.

Blunt version: the notebook family evolves from "broken and fake-confident" to "better instrumented but still narratively conflicted." Only the MLflow notebook materially closes the gap between claim and evidence.

## Structural Comparison

| Notebook | Cells | Code | Markdown | Headings | Executed code cells | Output cells | Saved errors | High-level read |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `HNDSR_Original.ipynb` | 46 | 32 | 14 | 0 | 0 | 17 | 4 | Broken exploratory dump with stale outputs |
| `HNDSR_Corrected.ipynb` | 5 | 4 | 1 | 1 | 0 | 0 | 0 | Tiny rewrite, no evidence, no execution trail |
| `HNDSR_Kaggle.ipynb` | 34 | 20 | 14 | 24 | 0 | 0 | 0 | Cleaner Kaggle-oriented training notebook |
| `HNDSR_Databricks_MLflow.ipynb` | 47 | 23 | 24 | 38 | 0 | 0 | 0 | Best documented static notebook, still unevidenced |
| `MLFlow/HNDSR_MLflow.ipynb` | 47 | 24 | 23 | 26 | 23 | 10 | 1 | Only notebook with real execution evidence and bug-fix history |
| `HNDSR in Production/training/HNDSR_Kaggle_Updated.ipynb` | 34 | 20 | 14 | 24 | 0 | 0 | 0 | Near-duplicate of the Kaggle notebook |

## Comparison Matrix

| Notebook | What it fixes versus the original | What it still inherits or gets wrong | Credibility level |
|---|---|---|---|
| `HNDSR_Corrected.ipynb` | Removes output clutter, removes obvious Colab debris, compresses the pipeline | No execution evidence, no metrics, no artifact trail, too small to validate claims | Low |
| `HNDSR_Kaggle.ipynb` | Better structure, headings, dataset detection, cleaner training narrative | Still carries the `29.40 / 0.87 / 0.16` result table; no executed evidence; still environment-coupled to Kaggle | Low |
| `HNDSR_Databricks_MLflow.ipynb` | Adds stronger documentation, includes `implicit_amp` in the described optimizer path, documents checkpoint composition, adds MLflow framing | Still repeats the inflated Stage 3 metrics; no executed trail; still more narrative than evidence | Medium-low |
| `MLFlow/HNDSR_MLflow.ipynb` | Explicitly documents the `implicit_amp` bug, fixes diffusion-context mismatch, adds SDEdit-style reasoning, includes actual execution traces and measured metrics | Still contains the paper metrics table, still contains a kernel crash, still mixes retrospective explanation with live notebook evidence | Medium |
| `HNDSR in Production/training/HNDSR_Kaggle_Updated.ipynb` | Packages the Kaggle notebook into the production tree | Inherits Kaggle notebook problems almost verbatim, including the inflated paper metrics | Low |

## Findings

### 1. The notebook family cleans up formatting before it cleans up truth

Severity: Critical

Evidence:

- The original notebook has zero markdown headings, stale outputs, and saved failures.
- The later notebooks add headings, sectioning, and explanatory markdown.
- But `HNDSR_Kaggle.ipynb`, `HNDSR_Databricks_MLflow.ipynb`, and `HNDSR in Production/training/HNDSR_Kaggle_Updated.ipynb` still preserve the paper-style claim that full HNDSR reaches `29.40 / 0.87 / 0.16`.

Impact:

- Readers can easily mistake improved writing for improved validity.
- The polished notebooks are safer to present, but not automatically safer to trust.

### 2. `HNDSR_Corrected.ipynb` is a rewrite, not evidence

Severity: High

Evidence:

- It contains only 5 cells total.
- It has no saved outputs, no executed cells, no metrics, and no artifact record.
- Its opening markdown claims a "complete pipeline," but the notebook itself is just a compact code rewrite.

Impact:

- This file can help as a code sketch.
- It cannot serve as proof that the pipeline trains, evaluates, or reproduces results.

### 3. The Kaggle notebooks are cleaner but still sell the paper result, not the measured result

Severity: Critical

Evidence:

- `HNDSR_Kaggle.ipynb` includes `| **HNDSR (Ours)** | **29.40** | **0.87** | **0.16** |`.
- `HNDSR in Production/training/HNDSR_Kaggle_Updated.ipynb` contains the same result line.
- Neither notebook preserves executed evidence showing those numbers being produced.

Impact:

- These notebooks inherit the original project's core reporting problem: the headline numbers are asserted, not demonstrated.
- They are presentation artifacts, not reliable experiment records.

### 4. The Databricks notebook improves the engineering story but not the empirical honesty

Severity: High

Evidence:

- `HNDSR_Databricks_MLflow.ipynb` documents that Stage 2 optimizes both `neural_op` and `implicit_amp`.
- It documents that checkpoints save both `neural_operator_best.pth` and `implicit_amp`.
- It still states `| Stage 3 — + Diffusion (full HNDSR) | **29.40** | **0.87** | **0.16** |`.
- It contains no executed outputs to validate those claims.

Impact:

- This notebook is better as an implementation blueprint than the Kaggle notebook.
- It is still not a trustworthy experiment log because the performance narrative outruns the evidence.

### 5. The MLflow notebook is the only one that admits the real bugs

Severity: Critical

Evidence:

- It explicitly says `implicit_amp` was previously never included in the optimizer and therefore retained random weights.
- It explicitly says diffusion context previously did not match inference.
- It adds SDEdit-style diffusion initialization and states that the ablation found `diffusion_strength=0.0` to be best.
- It reports measured evaluation results of `PSNR: 23.48 ± 3.25 dB`, `SSIM: 0.5700 ± 0.1224`, and `LPIPS: 0.4809 ± 0.1053`.

Impact:

- This is the first notebook in the chain that behaves like someone finally started debugging the actual system instead of only narrating the intended one.
- It materially undermines the earlier Stage 3 superiority claim.

### 6. The MLflow notebook is still not clean enough to be called a definitive research artifact

Severity: High

Evidence:

- It still includes the same Stage 3 table with `29.40 / 0.87 / 0.16`.
- It contains one saved failure: a VS Code Jupyter kernel crash.
- It mixes design notes, fixes, training traces, ablation, and reporting in one long notebook.

Impact:

- It is the most useful notebook here.
- It is still a notebook, not a rigorously isolated experiment package.

### 7. The production training notebook is mostly a relocation, not a new truth source

Severity: Medium

Evidence:

- Its structure and opening sections closely mirror `HNDSR_Kaggle.ipynb`.
- It retains the same headline metrics claim.
- It adds no new execution evidence.

Impact:

- Moving a questionable notebook under `HNDSR in Production/` does not upgrade its evidentiary value.
- It should not be treated as the canonical training history.

## Which Notebook Is the Best Source of Truth?

Short answer: `MLFlow/HNDSR_MLflow.ipynb`.

Why:

- It is the only notebook with substantial saved execution traces.
- It is the only notebook that documents the concrete v3 bug fixes.
- It is the only notebook that reports the measured metrics that align with the repo's more candid evaluation story.
- It is the only notebook that records the uncomfortable but important result that diffusion strength `0.0` performed best in the tested ablation.

Why not "high confidence":

- It still carries contradictory narrative baggage from the paper storyline.
- It still lives as an all-in-one notebook rather than a controlled training/evaluation pipeline with immutable configs and separate reports.

## Recommended Canonical Ordering

If someone insists on reading this project through notebooks, the least misleading order is:

1. `HNDSR_Original.ipynb` as the failure baseline.
2. `HNDSR_Databricks_MLflow.ipynb` as the best static explanation of the intended pipeline.
3. `MLFlow/HNDSR_MLflow.ipynb` as the best available evidence of what was actually fixed and measured.

The Kaggle notebooks should be treated as convenience-training variants, not as authoritative result sources.

## Power of 10 Lens

Relevant Power of 10 pressure points:

- Rule 1, simple control flow: none of these notebooks are simple enough to audit confidently as a single execution artifact.
- Rule 2, fixed loop bounds: notebook training code and evaluation sweeps are parameterized, but the real problem here is not loop bounds, it is hidden state and environment coupling.
- Rule 5, assertions: the notebooks are assertion-poor in the engineering sense. They explain intent in markdown instead of enforcing invariants in code.
- Rule 6, limited scope: the later notebooks still bundle setup, training, evaluation, logging, and reporting into one mutable document.
- Rule 10, compile with warnings and static analysis: this does not map cleanly to notebooks, which is exactly part of the problem. They evade the feedback loops that would normally catch drift between claim and implementation.

Important limitation:

Power of 10 does not directly grade scientific honesty, experiment reporting, or notebook rhetoric. It does, however, explain why these notebooks are hard to trust: too much hidden state, too many responsibilities, too little executable discipline.

## Bottom Line

The comparison is not "original bad, later notebooks good."

The real pattern is:

- original notebook: operationally broken
- corrected and Kaggle notebooks: cosmetically improved but still weak evidence
- Databricks notebook: better implementation narrative, still inflated on results
- MLflow notebook: first artifact that seriously admits bugs and reports grounded metrics

If this repo needs one notebook to cite while the codebase is being corrected, cite `MLFlow/HNDSR_MLflow.ipynb` and explicitly say it supersedes the earlier performance story.
