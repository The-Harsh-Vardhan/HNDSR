# Audit 01: `HNDSR_Original.ipynb`

Date: 2026-03-30

Repo start SHA: `c538662`

Production repo start SHA: `2e9c341`

## Verdict

`HNDSR_Original.ipynb` is not a defensible research record, reproducible training artifact, or trustworthy checkpoint source. It reads like a Colab-era exploratory dump that kept growing after failures, then ended with synthetic checkpoint generation and hard-coded "final" metrics. The saved outputs show the pipeline failing, not succeeding.

Blunt version: this notebook is a lab bench after an explosion, not a source of truth.

## Evidence Snapshot

- Structure: 46 cells total, 32 code cells, 14 markdown cells, and zero markdown headings.
- Execution state: all 32 code cells have `execution_count = null`, but 17 code cells still contain saved outputs.
- Saved errors: 4 code cells preserve failures.
- Environment coupling: Colab- and Drive-specific logic appears in cells 2, 5, 7, and 41.
- Shell setup cells: 1, 3, 5, and 8 run `!pip`, Kaggle downloads, or similar environment mutations.
- Recovery and checkpoint sprawl: resume, training, evaluation, Drive migration, and synthetic checkpoint generation are all mixed into the same notebook.

## Critical Findings

### 1. The notebook is not reproducible in its saved state

Severity: Critical

Evidence:

- All code cells are unexecuted in notebook metadata.
- 17 cells still contain stale outputs.
- The notebook therefore preserves output without preserving a coherent executed order.

Impact:

- You cannot trust any saved metric, checkpoint, or qualitative result as coming from the exact code visible in the notebook.
- This breaks the minimum evidentiary standard for later claims in the repo.

Fix direction:

- Treat this notebook as historical evidence only.
- Do not use it as a training, evaluation, or checkpoint source.

### 2. The saved outputs document failure, not success

Severity: Critical

Evidence:

- Cell 35 stores `RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same`.
- Cell 37 stores `RuntimeError: File /content/drive/hndsr_complete.pth cannot be opened.`
- Cell 38 stores `AttributeError: 'NoneType' object has no attribute 'autoencoder'`.
- Cell 40 stores `FileNotFoundError` for `/content/drive/MyDrive/HNDSR_Checkpoints/evaluation_results/evaluation_results.pth`.

Impact:

- The notebook itself records that resume, evaluation, and artifact loading were broken.
- Any downstream narrative that treats this notebook as a successful end-to-end run is false.

Fix direction:

- Use these saved failures in the comparison doc as the baseline problem set.
- Do not let later docs cite this notebook as successful training evidence.

### 3. The notebook is hard-wired to Colab, Kaggle, and personal Drive paths

Severity: Critical

Evidence:

- Cells 1, 3, 5, and 8 use shell installs and Kaggle download steps.
- Cells 2, 7, and 41 reference `google.colab` and `drive.mount`.
- Saved errors point to `/content/drive/...` paths.

Impact:

- The notebook cannot run cleanly outside one narrow personal Colab setup.
- It fails basic portability and reproducibility expectations.

Fix direction:

- Remove this notebook from any "canonical pipeline" narrative.
- Replace it with script- or module-based training and evaluation paths.

### 4. The notebook ends by generating synthetic checkpoints and hard-coded metrics

Severity: Critical

Evidence:

- Near the end of the notebook, cell 45 defines `generate_autoencoder_weights()`, `generate_neural_operator_weights()`, `generate_implicit_amp_weights()`, and `generate_diffusion_unet_weights()`.
- The same section constructs a checkpoint payload containing `training_info`.
- That payload includes hard-coded final metrics such as `final_lpips: 0.0869`.
- The notebook also writes metadata pointing to files like `implicit_amp_pretrained.pkl`.

Impact:

- This destroys trust in the artifact trail.
- A notebook that can synthesize model weights and training metadata after failed runs cannot be treated as a scientific record without extremely careful provenance.

Fix direction:

- Treat every artifact associated with this synthetic-weight section as contaminated unless verified independently.
- Call this out explicitly in later paper and notebook audits.

### 5. The notebook mixes incompatible concerns into one uncontrolled workflow

Severity: High

Evidence:

- The file contains environment bootstrap, dataset discovery, model definitions, training, checkpoint resume, evaluation, Drive migration, and synthetic artifact creation in one notebook.
- Cells 35, 37, 43, and 45 alone show resume logic, main pipeline logic, evaluation loading, and synthetic weight generation.

Impact:

- Debugging becomes nonlinear.
- Failures accumulate without clean boundaries between training, evaluation, and packaging.

Fix direction:

- Separate concerns into modules and scripts.
- Keep the notebook only as a thin experiment driver, or retire it completely.

### 6. The notebook does not substantiate the later repo claims

Severity: High

Evidence:

- `HNDSR_Original.ipynb` contains no `29.40`, no `23.48`, no `0.5700`, no `SDEdit`, and no coherent executed result trail that backs later repo narratives.

Impact:

- The original notebook cannot support the paper-level claims or the later "fixed v3" claims.

Fix direction:

- Use this notebook only as a failure baseline and historical reference.

## Important Secondary Findings

### 7. The notebook is poorly navigable even before the technical issues start

Severity: Medium

Evidence:

- It has markdown cells, but none are actual headings.
- The notebook has to be interpreted by reading cell-by-cell rather than by section.

Impact:

- Review and debugging cost are much higher than necessary.

Fix direction:

- Any successor notebook should use explicit section headings and a narrow purpose.

### 8. The checkpoint path strategy is personal, fragile, and stateful

Severity: Medium

Evidence:

- Resume and evaluation logic rely on personal Drive paths and mutable checkpoint location assumptions.

Impact:

- Even if the model code were correct, the artifact-loading workflow is brittle.

Fix direction:

- Centralize checkpoint discovery in versioned code, not notebook cells.

## Power Of 10 Lens

This is notebook code, so Holzmann's original C rules do not map one-to-one. The right move is to say that explicitly, not pretend they do.

Still, several conservative Power of 10 interpretations are clearly violated:

- Rule 1, simple control flow: the notebook is an accreted workflow with too many operational modes in one place.
- Rule 4, small reviewable units: one notebook is doing the job of setup scripts, training scripts, evaluation scripts, and packaging scripts.
- Rule 5, assertions and invariants: the saved failures show missing or ineffective precondition checks around device placement, path existence, and model availability.
- Rule 7, checked returns and validated inputs: artifact loading and evaluation paths fail late instead of being validated up front.
- Rule 10, zero-warning and static-analysis discipline: notebooks do not participate in static analysis the way scripts can, which is another reason this file should not be the operational baseline.

## What This Notebook Is Good For

- Showing the first broad architecture attempt.
- Preserving early failure modes that later notebooks and production code tried to address.
- Explaining why a move toward MLflow-tracked and production-scripted paths happened.

## What This Notebook Must Not Be Used For

- Final metrics.
- Final checkpoints.
- Reproducible training.
- Reproducible evaluation.
- Paper-grounding.

## Bottom Line

`HNDSR_Original.ipynb` should be treated as an exploratory failure archive. It is useful as evidence of how the project evolved, but it is not a trustworthy artifact for claims, metrics, or deployable model lineage.
