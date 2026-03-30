# HNDSR Rebuild Kickoff

## Intent

- Rebuild the research lane in an isolated folder.
- Keep production untouched until a new baseline is verified.
- Use W&B as the only experiment tracker in the new lane.

## Phase Ladder

1. Bicubic bootstrap and dataset parity checks
2. SR3-style conditional diffusion baseline
3. Latent diffusion baseline
4. Neural-operator baseline
5. Hybrid merge with bounded ablations

## Gate Policy

- Do not promote a phase unless it beats bicubic on the tracked validation pack.
- Every promoted phase must export metrics JSON and sample strips for the fixed smoke pack.
- Notebook outputs are summaries only; scripts remain the source of truth.
