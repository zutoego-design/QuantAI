# multifactor-rule-confirmatory-v1 Rejection Postmortem

Canonical run postmortem: `reports/runs/20260615T132128Z-experiment-346a7b28/v1_rejection_postmortem.md`

Final status: `rejected_final`

Final run: `20260615T132128Z-experiment-346a7b28`

The v1 confirmatory study is closed. The saved artifact remains valid and reproducible, but the evidence is finally rejected and the 2024-08-01 to 2025-12-31 holdout may not be reused for v1 tuning or confirmatory reruns.

Key rejection drivers:

- Deflated Sharpe probability was `0.6954`, below the preregistered `0.95` requirement.
- Trial count reached the now-recorded v1 budget of `5`.
- All `13` preregistered factor-level evidence checks were unsupported after expected-direction and FDR controls.
- Final reporting used a dirty git workspace, which weakens confirmatory traceability even though the artifact identity was captured.

Governance updates:

- `experiments/confirmatory_rule_score.yaml` is marked `study_status: rejected_final`.
- `experiments/study_closures.json` records the closed study and holdout window.
- Current run decision and manifest evidence status are marked `rejected_final`.
- Future confirmatory runs hard fail on closed holdout reuse or exceeded `trial_budget`.
