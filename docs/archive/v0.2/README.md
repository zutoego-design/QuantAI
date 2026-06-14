# Archived v0.2 Docs

Archived on: `2026-06-13`

## Purpose

This folder contains completed `v0.2` delivery documents that describe the shipped
research foundation and its validation record. They were moved out of the root
`docs/` directory so the root can focus on active planning and pending development.

## Archived files

- [PROJECT_STATUS.md](./PROJECT_STATUS.md)
- [RESEARCH_METHODOLOGY.md](./RESEARCH_METHODOLOGY.md)
- [DATA_LIMITATIONS.md](./DATA_LIMITATIONS.md)
- [CODE_REVIEW.md](./CODE_REVIEW.md)
- [REPORT_APPLICATION_AUDIT.md](./REPORT_APPLICATION_AUDIT.md)
- [DEVELOPMENT_ACCEPTANCE_CHECKLIST.md](./DEVELOPMENT_ACCEPTANCE_CHECKLIST.md)
- [AI辅助美股量化研究平台_改进版系统建设报告.md](./AI辅助美股量化研究平台_改进版系统建设报告.md)

## When to read these

- Use `PROJECT_STATUS.md` for the last validated scope and run status.
- Use `RESEARCH_METHODOLOGY.md` for the current PIT and research-gate rules.
- Use `DATA_LIMITATIONS.md` for free-data caveats.
- Use `CODE_REVIEW.md` for the final review snapshot of the completed `v0.2` work.
- Use `REPORT_APPLICATION_AUDIT.md` and
  `DEVELOPMENT_ACCEPTANCE_CHECKLIST.md` as historical code-delivery records.
- Use the Chinese construction report as the original design blueprint.

## Evidence context

The original v0.2 delivery record was validated against:

- `reports/runs/20260612T153324Z-backtest-8f5a55b3`
- 51 passing tests, compileall, Ruff, strict data validation, acceptance checks, and
  Streamlit AppTest as recorded in `PROJECT_STATUS.md`

The archived completion audit and checklist were archived after the repository gained
labels, ML baselines, metadata, registry, approval, text/event, and job-definition
capabilities. The later run
`reports/runs/20260613T103622Z-backtest-90ef83a1` confirms the strict rule-score path,
but it did not enable ML or text factors and has not yet completed the v0.3 operating
validation.

For the current truth, use the active [documentation index](../../README.md).
