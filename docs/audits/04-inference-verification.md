# Audit 04: Inference Verification

Date: 2026-03-30

Scope:

- bootstrap a local `.venv`
- verify runtime imports
- run targeted production tests
- add `--checkpoint-dir` support to `HNDSR in Production/scripts/run_sr.py`
- compare inference on the same sample using:
  - `HNDSR in Production/checkpoints/`
  - `HNDSR in Production/MLFlow/`

Sample used:

- LR input: `HNDSR in Production/MLFlow/evaluation_results/lr/sample_000.png`
- HR reference: `HNDSR in Production/MLFlow/evaluation_results/hr/sample_000.png`
- Existing saved SR reference: `HNDSR in Production/MLFlow/evaluation_results/sr/sample_000.png`

## Executive Verdict

Inference verification succeeded, but it split the two checkpoint sets cleanly:

- the **MLFlow checkpoint set** is the only one that behaves like the latest, believable model,
- the **production `checkpoints/` set** runs but produces much worse output on the verified sample,
- and the production checkpoint manifest is in a failed state because the saved hash for `autoencoder_best.pth` does not match the actual file.

Blunt version: the repo can run inference, but the trustworthy artifact is the MLFlow checkpoint set, not the currently published production checkpoint bundle.

## Subject 1: Environment

### Local runtime used

- Python: `3.13.12`
- Torch: `2.11.0+cpu`
- FastAPI: `0.135.2`
- pytest: `9.0.2`
- scikit-image: `0.26.0`
- Device: CPU only (`torch.cuda.is_available() == False`)

### Important environment deviation

The repo pins `torch==2.1.2`, but that version does not publish wheels for the usable interpreter on this machine (`Python 3.13`). To complete verification honestly, the local `.venv` used a Python-3.13-compatible CPU Torch stack instead:

- `torch 2.11.0+cpu`
- `torchvision 0.26.0+cpu`

This deviation affects environment fidelity, not the core checkpoint-format findings below.

## Subject 2: Verification Checks

### Import verification

Verified successfully inside the local `.venv`:

- `torch`
- `fastapi`
- `pytest`
- `skimage`

### Targeted tests

Passed:

- `HNDSR in Production/tests/test_infer_fallback_mode.py`
  - `2 passed in 3.65s`
- `HNDSR in Production/tests/test_checkpoint_manifest_validation.py`
  - `3 passed in 0.07s`

Important note:

- The manifest test initially failed inside the sandbox because pytest temporary-directory cleanup hit Windows/OneDrive permission issues.
- Rerunning that same test file outside the sandbox resolved the harness problem and the code-level assertions passed.

### Script smoke check

The new checkpoint-directory CLI path worked:

- command path: `scripts/run_sr.py --checkpoint-dir checkpoints ...`
- result: successful model load and sample inference on CPU

## Subject 3: Code Change Applied

Two minimal script-level changes were required in `HNDSR in Production/scripts/run_sr.py`:

1. Public CLI change requested in the plan:
   - added `--checkpoint-dir`
   - default remains `PROJECT_ROOT/checkpoints`

2. Minimal blocker fix needed to complete the second inference pass:
   - extended the Stage 2 loader so `neural_operator_best.pth` can read the MLFlow checkpoint format:
     - `{"neural_operator": ..., "implicit_amp": ...}`
   - when present, the script now loads the saved `implicit_amp` weights instead of discarding them

No HTTP/API behavior was changed.

## Subject 4: Production Checkpoint Verification

### Manifest status

Actual manifest validation result on `HNDSR in Production/checkpoints/`:

- `manifest_ok=False`
- `manifest_match=False`
- failure reason:
  - `autoencoder_best.pth: expected=399b72... actual=b63377...`

Per-file status:

- `autoencoder_best.pth`: hash mismatch
- `neural_operator_best.pth`: hash match
- `diffusion_best.pth`: hash match

### Artifact consistency across checkpoint sets

Hash comparison between `checkpoints/` and `MLFlow/`:

- `autoencoder_best.pth`: same
- `neural_operator_best.pth`: different
- `diffusion_best.pth`: different

Interpretation:

- the repo currently ships at least two materially different model bundles
- they are not interchangeable
- only one of them appears to line up with the saved MLFlow evaluation outputs

## Subject 5: Inference Runs

### Run A: production checkpoint directory

Command shape:

- `--checkpoint-dir checkpoints`
- `--scale 4`
- `--steps 50`
- `--strength 0.0`
- `--device cpu`

Outcome:

- load: success
- output size: `64x64`
- runtime: about `0.04s`

Output files:

- `HNDSR in Production/.inference-prod/sample_000_sr.png`
- `HNDSR in Production/.inference-prod/sample_000_compare.png`

### Run B: MLFlow checkpoint directory

Command shape:

- `--checkpoint-dir MLFlow`
- `--scale 4`
- `--steps 50`
- `--strength 0.0`
- `--device cpu`

Outcome:

- load: success after the minimal loader-format fix
- output size: `64x64`
- runtime: about `0.04s`

Output files:

- `HNDSR in Production/.inference-mlflow/sample_000_sr.png`
- `HNDSR in Production/.inference-mlflow/sample_000_compare.png`

## Subject 6: Quantitative Sample Comparison

Single-sample quick metrics against the paired HR image:

| Output | PSNR vs HR | SSIM vs HR |
|---|---:|---:|
| `production checkpoints` | `12.4986` | `0.127641` |
| `MLFlow checkpoints` | `19.6291` | `0.457314` |
| `existing saved MLFlow SR` | `19.6271` | `0.457581` |

Direct output-to-output comparison:

| Pair | Mean absolute error | Max absolute pixel diff |
|---|---:|---:|
| `production` vs `MLFlow` | `42.393719` | `134` |
| `production` vs `saved MLFlow SR` | `42.266357` | `135` |
| `MLFlow` vs `saved MLFlow SR` | `0.465251` | `2` |

Interpretation:

- The MLFlow checkpoint run almost reproduces the existing saved MLFlow SR output exactly.
- The production checkpoint run is dramatically different and dramatically worse on this sample.
- The production artifact bundle therefore does not currently represent the latest verified model behavior.

## Subject 7: Ground-Reality Conclusion

What is now verified:

- the script can run inference end to end in a local `.venv`
- the new `--checkpoint-dir` path works
- the MLFlow checkpoint set is loadable and produces output aligned with the saved evaluation artifact
- the production checkpoint bundle is runnable but does not validate cleanly and performs much worse on the checked sample

What this means in practice:

- If the question is "which checkpoint set reflects the latest believable HNDSR model state?", the answer is `HNDSR in Production/MLFlow/`, not `HNDSR in Production/checkpoints/`.
- If the question is "does the current production checkpoint bundle deserve trust as-is?", the answer is no, not until the manifest mismatch and artifact drift are corrected.

## Power of 10 Lens

Relevant lessons:

- Rule 5, assertions: the manifest mismatch is exactly the kind of artifact-integrity failure that should block trust automatically.
- Rule 6, limited scope: inference code should not need ad hoc reasoning about multiple checkpoint formats in the first place.
- Rule 10 analogue, strong verification feedback: targeted tests plus direct sample comparison caught a real model-bundle split that documentation alone obscured.

This is a good example of why the repo needed this phase. The code "runs" was not the right question. The right question was "which artifact run is the one we should trust?" and the answer turned out not to be the production bundle.
