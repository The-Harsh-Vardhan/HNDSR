# Audit 03: Paper Feasibility and Ground Reality

Date: 2026-03-30

Primary paper artifact: `docs/HNDSR_Paper_Final.md`

Cross-checked against:

- `README.md`
- `HNDSR in Production/README.md`
- `MLFlow/HNDSR_MLflow.ipynb`
- `HNDSR in Production/scripts/run_sr.py`
- `HNDSR in Production/backend/app.py`
- `HNDSR in Production/backend/inference/model_loader.py`
- `HNDSR in Production/backend/model/model_stubs.py`
- `HNDSR in Production/checkpoints/manifest.json`
- `HNDSR in Production/MLFlow/evaluation_results/`

## Executive Verdict

The paper describes a strong research story, but the repo reality is a weaker, split-brain implementation:

- the **idea** is plausible,
- the **measured v3 system** is materially worse than the headline paper metrics,
- the **production deployment** does not fully line up with the v3 training narrative,
- and the **continuous-scale claim** is only partially true in code and not true in the public API.

Blunt version: the paper writes up the best-case imagined HNDSR. The repo shows a more complicated truth: a promising hybrid pipeline, real bug history, middling measured performance, and a production stack that still contains mismatched assumptions.

## Feasibility Snapshot

| Subject | Paper claim | Repo reality | Assessment |
|---|---|---|---|
| Core hybrid idea | Neural operator prior + latent diffusion is effective | The hybrid pipeline exists in notebooks and production code | Feasible in principle |
| Stage 3 benefit | Diffusion refinement improves the system to `29.40 / 0.87 / 0.16` | Measured v3 results are `23.48 / 0.5700 / 0.4809`; ablation says `diffusion_strength=0.0` is best | Not supported |
| Continuous-scale SR | Supports arbitrary, non-integer scales | Script accepts float scale; production API only accepts integer `scale_factor` from 2 to 8 | Only partially true |
| Unified optimisation | Joint optimisation strategy balances all losses | Repo training story is stage-wise: autoencoder, then neural operator, then diffusion | Paper overstates integration |
| Production faithfulness | Real-world viable deployment of the trained architecture | Deployment uses reverse-engineered stubs, random-init `implicit_amp`, and an inconsistent manifest | Operationally fragile |

## Subject 1: Metrics and Headline Claims

### Finding 1. The paper's headline numbers are contradicted by the repo's own measured results

Severity: Critical

Paper claim:

- `docs/HNDSR_Paper_Final.md` states Stage 3 achieves `PSNR 29.40 dB`, `SSIM 0.87`, and `LPIPS 0.16`.
- The paper also says HNDSR outperforms existing methods on 4× satellite SR.

Repo reality:

- `README.md` keeps those figures only as `HNDSR (Target)`.
- `README.md` also reports measured v3 results of `23.48 ± 3.25`, `0.5700 ± 0.1224`, and `0.4809 ± 0.1053`.
- `HNDSR in Production/README.md` repeats the same measured v3 results.
- `MLFlow/HNDSR_MLflow.ipynb` prints the same measured evaluation values.

Impact:

- The paper's main quantitative claim is not supported by the repo's best available evidence.
- At minimum, the paper is mixing aspirational metrics with measured metrics.

Assessment:

- The paper's numbers should be treated as unsupported target values unless independent evidence is produced.

## Subject 2: Diffusion Contribution

### Finding 2. The paper says diffusion is the winning refinement stage, but the measured ablation says otherwise

Severity: Critical

Paper claim:

- The paper says the diffusion refinement stage "yielded the highest PSNR, SSIM, and lowest LPIPS values."
- It frames Stage 3 as the reason the system becomes superior.

Repo reality:

- `MLFlow/HNDSR_MLflow.ipynb` records a diffusion-strength sweep and reports `BEST: diffusion_strength=0.0`.
- `README.md` and `HNDSR in Production/README.md` both show the same v3 ablation pattern:
  - `0.0`: best
  - `0.1`: worse
  - `0.3`: worse
  - `1.0`: much worse
- The repo's own explanation is that pure-noise diffusion was broken enough to require an SDEdit-style workaround, and even then skipping diffusion often performs best.

Impact:

- The paper's core narrative about diffusion-driven superiority is upside down relative to the measured v3 evidence.
- What the repo currently supports is: "hybrid architecture exists, but the diffusion component is not yet a reliable improvement over direct decoding."

Assessment:

- The diffusion module is feasible as a research direction.
- Its claimed empirical advantage is not established by this repo.

## Subject 3: Continuous-Scale Super-Resolution

### Finding 3. The paper and READMEs market arbitrary-scale SR harder than the production interface actually supports

Severity: High

Paper claim:

- `docs/HNDSR_Paper_Final.md` repeatedly describes HNDSR as a continuous-scale model with non-integer upscaling support.
- `HNDSR in Production/README.md` explicitly advertises arbitrary scale factors such as `3.14×`.

Repo reality:

- `HNDSR in Production/scripts/run_sr.py` accepts `--scale` as a float, so the CLI path is at least nominally scale-continuous.
- `HNDSR in Production/backend/app.py` defines `scale_factor: int = Field(default=4, ge=2, le=8)`.
- The public production API therefore exposes only integer scales, not arbitrary continuous ones.
- The paper itself quietly admits that the current implementation focuses on a fixed 4× dataset due to training constraints.

Impact:

- The production story and the research story are not aligned.
- "Continuous-scale" is currently an architectural aspiration plus a CLI possibility, not a convincingly validated deployed capability.

Assessment:

- It is fair to say the architecture is scale-conditioned.
- It is not fair to present the deployed system as already proven for arbitrary non-integer operational use.

## Subject 4: Training Strategy and Pipeline Description

### Finding 4. The paper sells "joint optimisation," but the repo is actually a sequential three-stage pipeline

Severity: High

Paper claim:

- The paper lists a "Joint Optimisation Strategy" that balances diffusion, perceptual, and content losses in a unified training scheme.

Repo reality:

- `HNDSR in Production/README.md` describes the training pipeline as:
  - Stage 1: autoencoder
  - Stage 2: neural operator
  - Stage 3: diffusion UNet
- The notebooks follow the same staged narrative.
- The repo's bug history is also stage-specific: `implicit_amp` was not optimised in Stage 2, and diffusion conditioning was inconsistent in Stage 3.

Impact:

- The paper's wording overstates the coherence of the actual training process.
- This matters because stage boundaries are exactly where the real implementation bugs appeared.

Assessment:

- The codebase implements staged training, not a tightly unified optimisation process.
- The paper should describe that honestly.

## Subject 5: Production Faithfulness

### Finding 5. The production stack is not a clean deployment of the measured v3 story

Severity: Critical

Evidence:

- `HNDSR in Production/backend/model/model_stubs.py` explicitly says the architecture is reverse-engineered from checkpoint tensor shapes.
- `HNDSR in Production/backend/inference/model_loader.py` states that `ImplicitAmplification` has no saved checkpoint and uses fresh random initialization.
- `HNDSR in Production/scripts/run_sr.py` also instantiates `implicit_amp` but does not load any saved state for it.
- This conflicts with `MLFlow/HNDSR_MLflow.ipynb`, which says v3 fixed the bug by jointly training `implicit_amp` and saving it with the neural-operator checkpoint.

Impact:

- There is a serious alignment problem between the "best measured" notebook narrative and the deployed inference stack.
- Even if the production system runs, it may not faithfully represent the v3 model that produced the measured notebook results.

Assessment:

- The project has a deployable super-resolution stack.
- It does not yet have a fully trustworthy claim that production equals the measured v3 research artifact.

### Finding 6. The production checkpoint manifest is currently inconsistent

Severity: High

Evidence:

- `HNDSR in Production/checkpoints/manifest.json` lists SHA-256 hashes for the three production checkpoints.
- Local hash verification shows:
  - `neural_operator_best.pth`: matches manifest
  - `diffusion_best.pth`: matches manifest
  - `autoencoder_best.pth`: does **not** match manifest

Impact:

- The repo's integrity-control mechanism is already compromised for at least one production artifact.
- This directly weakens any "enterprise-grade" or "production-trustworthy" framing until corrected.

Assessment:

- Deployment is feasible.
- Production validation hygiene is not yet reliable.

## Subject 6: Artifact Consistency Across Checkpoint Sets

### Finding 7. The repo contains at least two materially different checkpoint sets

Severity: High

Evidence:

- `HNDSR in Production/checkpoints/` and `HNDSR in Production/MLFlow/` contain overlapping checkpoint filenames.
- Hash comparison shows:
  - `autoencoder_best.pth`: same in both locations
  - `neural_operator_best.pth`: different
  - `diffusion_best.pth`: different

Impact:

- "Which model are we talking about?" is not rhetorical here. It is a real operational problem.
- The paper presents one HNDSR result story, but the repo contains multiple materially distinct artifact sets.

Assessment:

- The repo does not currently present a single unambiguous canonical model state.

## Subject 7: Grounded Strengths

The project is not empty hype. A few things are genuinely defensible:

- The hybrid formulation is technically coherent enough to implement.
- The repo preserves a real debugging history rather than a magically perfect one.
- The MLflow notebook at least admits the important bugs and records grounded metrics.
- The production repo includes manifest validation, fallback handling, tests, and an inference stack substantial enough to verify.

Those are real strengths. They just do not justify the current paper-level confidence.

## Practical Feasibility Assessment

### Research feasibility

Status: Feasible

Reason:

- The idea of combining a scale-conditioned neural operator prior with latent-space refinement is technically plausible and already partially implemented.

### Performance feasibility

Status: Not yet demonstrated at the paper's claimed level

Reason:

- The best measured numbers in the repo are materially below the paper's target table.

### Continuous-scale feasibility

Status: Partial

Reason:

- The model code and CLI suggest float-scale support.
- The deployed API is still integer-only.
- The training/evaluation evidence is centered on a fixed 4× dataset.

### Production feasibility

Status: Feasible but fragile

Reason:

- The service architecture exists and can likely run inference.
- But it is undermined by checkpoint inconsistency, reverse-engineered stubs, and mismatch between notebook claims and production loading assumptions.

## Power of 10 Lens

Where Power of 10 helps:

- Rule 1, simple control flow: the staged pipeline is understandable, but the repo still has too many parallel stories for one system.
- Rule 5, assertions: checkpoint integrity and interface invariants should be enforced more aggressively; the manifest mismatch shows this plainly.
- Rule 6, limited scope: paper, notebooks, training artifacts, and production deployment are not separated cleanly enough.
- Rule 8, pointer/data discipline analogue: here the analogue is artifact discipline. The repo currently allows semantic drift between documents, checkpoints, and serving code.

Where it does not map directly:

- Power of 10 does not tell you whether a paper metric is honest.
- It does help explain why this repo drifted: insufficiently constrained interfaces, mutable notebooks, and too many responsibilities bundled together.

## Bottom Line

The paper is directionally plausible and empirically overstated.

What the repo supports today is:

- a feasible hybrid HNDSR concept,
- a real v3 bug-fix story,
- measured results around `23.48 / 0.5700 / 0.4809`,
- an inference/deployment stack that is usable but not fully aligned with the measured v3 narrative.

What the repo does **not** support today is:

- confidence in the `29.40 / 0.87 / 0.16` headline,
- confidence that diffusion is the decisive winning component,
- confidence that production currently represents a fully faithful continuous-scale v3 deployment.

If this project is going to be corrected honestly, the paper should be reframed as:

- "proposed architecture plus measured current status,"

not

- "already validated state-of-the-art result."
