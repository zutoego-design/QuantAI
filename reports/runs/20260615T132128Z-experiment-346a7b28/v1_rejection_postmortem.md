# multifactor-rule-confirmatory-v1 Rejection Postmortem

Final status: `rejected_final`

Final run: `20260615T132128Z-experiment-346a7b28`

Finalized at: `2026-06-15`

## Decision

`multifactor-rule-confirmatory-v1` is closed as final rejected evidence. The run remains a valid reproducible artifact, but it must not be described as confirmed alpha and the 2024-08-01 to 2025-12-31 holdout must not be reused for v1 tuning or confirmatory reruns.

## What Passed

- Artifact acceptance passed for the saved run.
- Holdout net total return was positive: `33.69%`.
- Holdout Sharpe was positive: `1.4321`.
- Bootstrap one-sided lower bound for Sharpe exceeded zero: `0.3070`.
- Fama-French style regression was available with `100%` holdout coverage.

## Why This Is Rejected

- Deflated Sharpe probability was `0.6954`, below the preregistered `0.95` requirement.
- Trial count reached `5`; this is now the v1 hard budget recorded for this study.
- All `13` preregistered factor-level evidence checks were unsupported after expected direction and FDR controls.
- The report was generated from a dirty git workspace, which is acceptable for artifact traceability but below the preferred standard for final confirmatory evidence.

## Governance Actions

- `experiments/confirmatory_rule_score.yaml` is marked `study_status: rejected_final`.
- `experiments/study_closures.json` blocks the v1 study and its current holdout window from confirmatory reuse.
- Current run `research_decision.json` and manifest `evidence_status` are marked `rejected_final`.
- Future confirmatory runs now hard fail when `trial_budget` is exceeded or a closed holdout is reused.

## V2 Guidance

- Do not require all 13 factors to explain the result simultaneously.
- Split value, low-risk, momentum, and quality into separate hypothesis families.
- Pre-register expected direction, score orientation, trial budget, and a fresh forward holdout or walk-forward window before any v2 confirmatory run.
