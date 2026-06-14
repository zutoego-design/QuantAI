# Construction Report Application Audit

Last updated: `2026-06-13`

## Executive conclusion

The actionable gaps identified in the original audit are now implemented and covered
by automated tests. The repository is ML-ready, PIT-aware, cost-aware, agent-auditable,
registry-backed, and protected by a human approval gate.

The project remains a research and paper-trading preparation platform. It does not
route live orders.

## Current status matrix

| Report area | Status | Evidence |
| --- | --- | --- |
| PIT-aware research data | Applied | membership reconstruction, SEC available dates, strict data gates |
| Feature metadata | Applied | `src/qss/factors/metadata.py`, run snapshots |
| Labels | Applied | `src/qss/labels/`, versioned Parquet artifacts |
| Walk-forward validation | Applied | purged and embargo-aware split engine |
| Linear baseline | Applied | Ridge and ElasticNet |
| Tree baseline | Applied | LightGBM |
| Robustness matrix | Applied | subperiod, cost, Top-N, rebalance shift, delisting |
| Critic audit | Applied | deterministic Markdown and JSON bias review |
| Experiment registry | Applied | DuckDB registry and query/refresh CLI |
| Human approval | Applied | review-required packet and explicit transition |
| SEC text/events | Applied | event metadata, text cache, risk-disclosure factor, event labels |
| Operating jobs | Applied | owned daily/monthly/registry job definitions |

## Acceptance artifacts

Every new backtest writes:

- `feature_snapshot.parquet`
- `factor_metadata.json`
- `label_config.json`
- label Parquet files and `label_validation.csv`
- `cost_sensitivity.csv`
- `bias_review.md` and `bias_review.json`
- `final_report.md`
- ML fold artifacts when `ml.enabled=true`

The built-in acceptance command verifies these artifacts in addition to the original
ledger, report, metric, benchmark, holding-count, and delisting checks.

## Deliberate implementation choices

- The storage layout remains `raw/silver/gold` rather than renaming stable paths.
- The deterministic orchestrator remains the control boundary; separate "agents" are
  represented by explicit spec, QA, critic, and report artifacts instead of autonomous
  code-modifying processes.
- The default scheduler is a documented job runner. Prefect can be selected later
  without changing job ownership or command contracts.
- Free S&P 500, Yahoo/Stooq, SEC, and FRED data retain the limitations documented in
  the archived v0.2 materials.

## Residual institutional upgrades

These are not acceptance blockers:

- licensed historical index membership and delisting data
- institutional corporate-action and point-in-time fundamentals feeds
- FRED vintage reconstruction
- production secret management and external scheduler deployment
- broker/paper-trading integration after a separate approval process
