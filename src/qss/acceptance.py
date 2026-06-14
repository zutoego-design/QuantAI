from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qss.backtest.metrics import compounded_monthly_returns
from qss.config.schema import AppConfig
from qss.data.storage import resolve_path, write_csv
from qss.reporting.service import report_bundle
from qss.runs.manifest import create_run_context


def _frames_equivalent(
    saved: pd.DataFrame,
    recomputed: pd.DataFrame,
    *,
    atol: float = 1e-12,
) -> bool:
    if saved.columns.tolist() != recomputed.columns.tolist() or len(saved) != len(
        recomputed
    ):
        return False
    for column in saved.columns:
        if pd.api.types.is_numeric_dtype(saved[column]):
            if not np.allclose(
                saved[column],
                recomputed[column],
                atol=atol,
                equal_nan=True,
            ):
                return False
        elif not saved[column].equals(recomputed[column]):
            return False
    return True


def _mappings_equivalent(
    saved: dict,
    recomputed: dict,
    *,
    atol: float = 1e-12,
) -> bool:
    if saved.keys() != recomputed.keys():
        return False
    for key, value in recomputed.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not np.isclose(saved[key], value, atol=atol, equal_nan=True):
                return False
        elif saved[key] != value:
            return False
    return True


def _experiment_acceptance_checks(
    config: AppConfig,
    root: Path,
    manifest: dict,
) -> list[dict]:
    from qss.ingestion.fama_french import load_fama_french_daily
    from qss.research.decision import research_evidence_decision
    from qss.research.orchestrator import ExperimentSpec, _factor_evidence
    from qss.research.statistics import (
        block_bootstrap_summary,
        deflated_sharpe_probability,
        fama_french_style_regression,
    )

    checks: list[dict] = []
    required = [
        "experiment_spec.json",
        "research_protocol.json",
        "data_snapshot.json",
        "child_runs.json",
        "robustness_matrix.csv",
        "holdout_evaluation/portfolio_metrics.csv",
        "bootstrap_summary.csv",
        "deflated_sharpe.json",
        "style_factor_exposures.csv",
        "style_factor_summary.json",
        "factor_evidence.csv",
        "research_decision.json",
        "research_decision.md",
    ]
    missing = [name for name in required if not (root / name).exists()]
    checks.append(
        {
            "check": "confirmatory_artifacts_complete",
            "passed": not missing,
            "details": str(missing),
        }
    )
    checks.append(
        {
            "check": "source_run_valid",
            "passed": manifest.get("status") == "valid",
            "details": manifest.get("status"),
        }
    )
    if missing:
        return checks

    protocol = json.loads(
        (root / "research_protocol.json").read_text(encoding="utf-8")
    )
    spec = ExperimentSpec.model_validate_json(
        (root / "experiment_spec.json").read_text(encoding="utf-8")
    )
    checks.append(
        {
            "check": "preregistered_protocol_identity",
            "passed": protocol == manifest.get("research_protocol")
            and spec.spec_hash == manifest.get("spec_hash")
            and protocol.get("stage") == "confirmatory",
            "details": f"stage={protocol.get('stage')}",
        }
    )
    gap_days = len(
        pd.bdate_range(
            pd.Timestamp(protocol["development_end"]) + pd.offsets.BDay(1),
            pd.Timestamp(protocol["holdout_start"]) - pd.offsets.BDay(1),
        )
    )
    checks.append(
        {
            "check": "holdout_label_gap",
            "passed": gap_days >= config.labels.horizon_days,
            "details": f"business_days={gap_days}",
        }
    )

    snapshot = json.loads((root / "data_snapshot.json").read_text(encoding="utf-8"))
    identity_payload = json.dumps(
        snapshot.get("files", []),
        sort_keys=True,
        separators=(",", ":"),
    )
    recomputed_snapshot_id = hashlib.sha256(
        identity_payload.encode("utf-8")
    ).hexdigest()
    checks.append(
        {
            "check": "data_snapshot_identity",
            "passed": recomputed_snapshot_id == snapshot.get("snapshot_id")
            == manifest.get("data_snapshot_id"),
            "details": snapshot.get("snapshot_id"),
        }
    )

    child_runs = json.loads((root / "child_runs.json").read_text(encoding="utf-8"))
    child_identity_ok = bool(child_runs)
    for child in child_runs:
        child_manifest_path = (
            resolve_path(config.paths.reports)
            / "runs"
            / child["run_id"]
            / "manifest.json"
        )
        if not child_manifest_path.exists():
            child_identity_ok = False
            break
        child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
        child_identity_ok = child_identity_ok and all(
            [
                child_manifest.get("status") == "valid",
                child_manifest.get("data_snapshot_id")
                == manifest.get("data_snapshot_id"),
                child_manifest.get("spec_hash") == manifest.get("spec_hash"),
                child_manifest.get("trial_number") == manifest.get("trial_number"),
            ]
        )
    checks.append(
        {
            "check": "child_run_identity",
            "passed": child_identity_ok,
            "details": f"children={len(child_runs)}",
        }
    )

    robustness = pd.read_csv(root / "robustness_matrix.csv")
    robustness_ok = (
        not robustness.empty
        and not robustness["status"].eq("skipped").any()
    )
    for row in robustness.itertuples(index=False):
        difference = json.loads(row.config_diff or "{}")
        if row.test == "top_n_sensitivity":
            robustness_ok = robustness_ok and set(difference) == {
                "optimizer.constraints.target_num_holdings"
            }
        elif difference:
            robustness_ok = False
    checks.append(
        {
            "check": "robustness_config_diffs",
            "passed": robustness_ok,
            "details": f"rows={len(robustness)}",
        }
    )

    decision = json.loads(
        (root / "research_decision.json").read_text(encoding="utf-8")
    )
    model = decision.get("selected_model", "rule_score")
    holdout_root = root / "holdout_evaluation" / model
    holdout_required = [
        "daily_returns.csv",
        "metrics.csv",
        "holdings.csv",
        "rebalances.csv",
        "trades.csv",
    ]
    missing_holdout = [
        name for name in holdout_required if not (holdout_root / name).exists()
    ]
    checks.append(
        {
            "check": "shared_holdout_ledger_complete",
            "passed": not missing_holdout,
            "details": str(missing_holdout),
        }
    )
    if missing_holdout:
        return checks

    daily = pd.read_csv(holdout_root / "daily_returns.csv")
    portfolio_metrics = pd.read_csv(
        root / "holdout_evaluation" / "portfolio_metrics.csv"
    )
    validation = config.research_validation
    saved_bootstrap = pd.read_csv(root / "bootstrap_summary.csv")
    recomputed_bootstrap = block_bootstrap_summary(
        daily,
        primary_metric=protocol["primary_metric"],
        block_size=validation.bootstrap_block_days,
        samples=validation.bootstrap_samples,
        seed=validation.bootstrap_seed,
        confidence_level=validation.confidence_level,
    )
    bootstrap_ok = (
        saved_bootstrap["metric"].tolist()
        == recomputed_bootstrap["metric"].tolist()
        and np.allclose(
            saved_bootstrap.select_dtypes(include=np.number),
            recomputed_bootstrap.select_dtypes(include=np.number),
            atol=1e-12,
            equal_nan=True,
        )
    )
    checks.append(
        {
            "check": "bootstrap_reproducible",
            "passed": bootstrap_ok,
            "details": f"samples={validation.bootstrap_samples}",
        }
    )

    saved_deflated = json.loads(
        (root / "deflated_sharpe.json").read_text(encoding="utf-8")
    )
    recomputed_deflated = deflated_sharpe_probability(
        daily["portfolio_return"],
        int(manifest.get("trial_number") or 1),
    )
    deflated_ok = all(
        np.isclose(
            saved_deflated[key],
            recomputed_deflated[key],
            atol=1e-12,
            equal_nan=True,
        )
        for key in recomputed_deflated
    )
    checks.append(
        {
            "check": "deflated_sharpe_reproducible",
            "passed": deflated_ok,
            "details": f"trials={manifest.get('trial_number')}",
        }
    )

    style_factors = load_fama_french_daily(validation.style_factor_cache)
    recomputed_exposures, recomputed_style = fama_french_style_regression(
        daily,
        style_factors,
    )
    recomputed_style["coverage"] = float(
        recomputed_style.get("observations", 0.0)
    ) / max(len(daily), 1)
    saved_exposures = pd.read_csv(root / "style_factor_exposures.csv")
    saved_style = json.loads(
        (root / "style_factor_summary.json").read_text(encoding="utf-8")
    )
    style_ok = (
        saved_exposures["factor"].tolist()
        == recomputed_exposures["factor"].tolist()
        and np.allclose(
            saved_exposures.select_dtypes(include=np.number),
            recomputed_exposures.select_dtypes(include=np.number),
            atol=1e-12,
            equal_nan=True,
        )
        and all(
            np.isclose(saved_style[key], value, atol=1e-12, equal_nan=True)
            for key, value in recomputed_style.items()
        )
    )
    checks.append(
        {
            "check": "style_regression_reproducible",
            "passed": style_ok and saved_style.get("coverage", 0.0) >= 0.95,
            "details": f"coverage={saved_style.get('coverage')}",
        }
    )

    full_child = next(
        child for child in child_runs if child.get("period") == "full"
    )
    diagnostics = pd.read_csv(
        resolve_path(config.paths.reports)
        / "runs"
        / full_child["run_id"]
        / "factor_diagnostics.csv"
    )
    recomputed_evidence, factor_blockers = _factor_evidence(
        diagnostics,
        spec.research_protocol(),
        validation.fdr_alpha,
    )
    saved_evidence = pd.read_csv(root / "factor_evidence.csv")
    evidence_ok = _frames_equivalent(saved_evidence, recomputed_evidence)
    checks.append(
        {
            "check": "factor_evidence_reproducible",
            "passed": evidence_ok,
            "details": f"factors={len(saved_evidence)}",
        }
    )

    blockers = list(factor_blockers)
    if recomputed_exposures.empty:
        blockers.append(
            "Fama-French 5-factor plus Momentum regression had insufficient overlap."
        )
    elif recomputed_style["coverage"] < 0.95:
        blockers.append(
            "Fama-French style-factor coverage is below 95% of holdout days."
        )
    recomputed_decision = research_evidence_decision(
        stage=protocol["stage"],
        primary_metric=protocol["primary_metric"],
        threshold=protocol["primary_metric_threshold"],
        bootstrap_summary=recomputed_bootstrap,
        deflated_sharpe=recomputed_deflated,
        net_total_return=float(portfolio_metrics.iloc[0]["net_total_return"]),
        required_probability=validation.deflated_sharpe_probability,
        blockers=blockers,
    )
    recomputed_decision["selected_model"] = model
    checks.append(
        {
            "check": "research_decision_reproducible",
            "passed": _mappings_equivalent(decision, recomputed_decision)
            and decision.get("status") == manifest.get("evidence_status"),
            "details": decision.get("status"),
        }
    )
    return checks


def run_acceptance_checks(config: AppConfig, run_path: str | Path | None = None):
    if run_path is None:
        latest_path = resolve_path(config.paths.reports) / "latest_run.json"
        if not latest_path.exists():
            raise ValueError("No latest run pointer exists.")
        run_path = json.loads(latest_path.read_text(encoding="utf-8"))["path"]
    source_root = Path(run_path)
    context = create_run_context(config, "acceptance")
    source_manifest_path = source_root / "manifest.json"
    if not source_manifest_path.exists():
        raise ValueError(f"Run manifest does not exist: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("run_type") == "experiment":
        checks = _experiment_acceptance_checks(
            config,
            source_root,
            source_manifest,
        )
        frame = pd.DataFrame(checks)
        status = "valid" if bool(frame["passed"].all()) else "invalid"
        write_csv(frame, context.path("acceptance_checks.csv"))
        context.update(
            status=status,
            quality_gates={
                row["check"]: bool(row["passed"]) for row in checks
            },
            notes=[f"Validated source run {source_manifest['run_id']}."],
        )
        return frame, context

    bundle = report_bundle(source_root)
    checks: list[dict] = []

    missing = bundle.validate()
    checks.append({"check": "report_bundle_complete", "passed": not missing, "details": str(missing)})
    manifest = json.loads(bundle.manifest.read_text(encoding="utf-8"))
    structured = json.loads(bundle.structured_report.read_text(encoding="utf-8"))
    source_config = manifest.get("config", {})
    protocol = manifest.get("research_protocol") or {}
    checks.append(
        {
            "check": "source_run_valid",
            "passed": manifest.get("status") == "valid",
            "details": manifest.get("status"),
        }
    )
    checks.append(
        {
            "check": "report_schema_version",
            "passed": structured.get("schema_version")
            == manifest.get("report_schema_version"),
            "details": structured.get("schema_version"),
        }
    )
    daily = pd.read_csv(bundle.daily_returns)
    metrics = pd.read_csv(bundle.metrics)
    rebalances = pd.read_csv(bundle.root / "rebalances.csv")
    sensitivity = pd.read_csv(bundle.root / "delisting_sensitivity.csv")
    saved_monthly = pd.read_csv(bundle.root / "monthly_returns.csv")
    recalculated = compounded_monthly_returns(daily)
    monthly_match = np.allclose(
        saved_monthly["portfolio_return"],
        recalculated["portfolio_return"],
        atol=1e-12,
        equal_nan=True,
    )
    checks.append({"check": "monthly_returns_compounded", "passed": monthly_match, "details": ""})
    checks.append(
        {
            "check": "daily_returns_complete",
            "passed": not daily[["portfolio_return", "benchmark_return"]].isna().any().any(),
            "details": str(daily[["portfolio_return", "benchmark_return"]].isna().sum().to_dict()),
        }
    )
    target_holdings = config.optimizer.constraints.target_num_holdings
    minimum_holdings = min(target_holdings, max(1, int(target_holdings * 0.5)))
    maximum_holdings = max(target_holdings, config.optimizer.candidate_count)
    holdings_ok = (
        not rebalances.empty
        and bool(
            rebalances["holding_count"]
            .between(minimum_holdings, maximum_holdings)
            .all()
        )
    )
    checks.append(
        {
            "check": "target_holding_count",
            "passed": holdings_ok,
            "details": f"range={minimum_holdings}-{maximum_holdings}",
        }
    )
    if protocol.get("stage") == "confirmatory":
        snapshot_path = bundle.root / "data_snapshot.json"
        snapshot = (
            json.loads(snapshot_path.read_text(encoding="utf-8"))
            if snapshot_path.exists()
            else {}
        )
        checks.append(
            {
                "check": "confirmatory_data_snapshot",
                "passed": bool(snapshot)
                and snapshot.get("snapshot_id")
                == manifest.get("data_snapshot_id"),
                "details": snapshot.get("snapshot_id", "missing"),
            }
        )
    required_research_artifacts = {
        "feature_snapshot.parquet",
        "factor_metadata.json",
        "label_config.json",
        "label_validation.csv",
        "cost_sensitivity.csv",
        "sector_return_attribution.csv",
        "sector_return_attribution_summary.csv",
        "bias_review.md",
        "bias_review.json",
        "final_report.md",
    }
    missing_research_artifacts = sorted(
        name for name in required_research_artifacts if not (bundle.root / name).exists()
    )
    checks.append(
        {
            "check": "ml_ready_artifacts_complete",
            "passed": not missing_research_artifacts,
            "details": str(missing_research_artifacts),
        }
    )
    if not missing_research_artifacts:
        label_checks = pd.read_csv(bundle.root / "label_validation.csv")
        checks.append(
            {
                "check": "label_validation_passed",
                "passed": not label_checks.empty and bool(label_checks["passed"].all()),
                "details": str(
                    label_checks.loc[~label_checks["passed"], "check"].tolist()
                ),
            }
        )
        factor_metadata = json.loads(
            (bundle.root / "factor_metadata.json").read_text(encoding="utf-8")
        )
        checks.append(
            {
                "check": "factor_metadata_complete",
                "passed": bool(factor_metadata)
                and all(
                    item.get("description")
                    and item.get("inputs")
                    and item.get("leakage_checks")
                    for item in factor_metadata
                ),
                "details": f"factors={len(factor_metadata)}",
            }
        )
        factor_diagnostics = pd.read_csv(bundle.root / "factor_diagnostics.csv")
        configured_factors = {
            factor_name
            for group in source_config.get("factor_groups", {}).values()
            for factor_name in group.get("factors", {})
        }
        configured_diagnostics = factor_diagnostics.loc[
            factor_diagnostics["factor_name"].isin(configured_factors)
        ]
        minimum_factor_coverage = float(
            source_config.get("strategy", {}).get("min_factor_coverage", 0.80)
        )
        configured_diagnostics = configured_diagnostics.assign(
            input_coverage=1.0
            - pd.to_numeric(
                configured_diagnostics["missing_rate"],
                errors="coerce",
            )
        )
        weak_factors = configured_diagnostics.loc[
            configured_diagnostics["input_coverage"] < minimum_factor_coverage,
            "factor_name",
        ].tolist()
        checks.append(
            {
                "check": "configured_factor_coverage",
                "passed": set(configured_diagnostics["factor_name"])
                == configured_factors
                and not weak_factors,
                "details": (
                    f"threshold={minimum_factor_coverage}; weak={weak_factors}; "
                    f"configured={len(configured_factors)}"
                ),
            }
        )
        cost_sensitivity = pd.read_csv(bundle.root / "cost_sensitivity.csv")
        checks.append(
            {
                "check": "cost_sensitivity_complete",
                "passed": len(cost_sensitivity) >= 3,
                "details": f"scenarios={len(cost_sensitivity)}",
            }
        )
        bias_review = json.loads(
            (bundle.root / "bias_review.json").read_text(encoding="utf-8")
        )
        checks.append(
            {
                "check": "critic_audit_structured",
                "passed": {
                    "blocking_issues",
                    "major_concerns",
                    "required_follow_up_tests",
                    "recommendation",
                }.issubset(bias_review),
                "details": bias_review.get("recommendation", ""),
            }
        )
        attribution = pd.read_csv(bundle.root / "sector_return_attribution.csv")
        attribution["date"] = pd.to_datetime(attribution["date"])
        daily_dates = pd.to_datetime(daily["date"])
        portfolio_attribution = (
            attribution.groupby("date")["portfolio_contribution"]
            .sum()
            .reindex(daily_dates, fill_value=0.0)
            .to_numpy()
        )
        benchmark_attribution = (
            attribution.groupby("date")["internal_cap_benchmark_contribution"]
            .sum()
            .reindex(daily_dates, fill_value=0.0)
            .to_numpy()
        )
        daily_reconciles = np.allclose(
            portfolio_attribution,
            daily["portfolio_return"],
            atol=1e-10,
            equal_nan=True,
        ) and np.allclose(
            benchmark_attribution,
            daily["internal_cap_weight_return"],
            atol=1e-10,
            equal_nan=True,
        )
        summary = pd.read_csv(
            bundle.root / "sector_return_attribution_summary.csv"
        )
        portfolio_total = float((1.0 + daily["portfolio_return"]).prod() - 1.0)
        benchmark_total = float(
            (1.0 + daily["internal_cap_weight_return"]).prod() - 1.0
        )
        linked_reconciles = np.isclose(
            summary["portfolio_linked_contribution"].sum(),
            portfolio_total,
            atol=1e-10,
        ) and np.isclose(
            summary["benchmark_linked_contribution"].sum(),
            benchmark_total,
            atol=1e-10,
        )
        checks.append(
            {
                "check": "sector_attribution_reconciles",
                "passed": bool(daily_reconciles and linked_reconciles),
                "details": (
                    f"daily={daily_reconciles}; linked={linked_reconciles}"
                ),
            }
        )
    if source_config.get("ml", {}).get("enabled"):
        fold_metrics_path = bundle.root / "ml_evaluation" / "fold_metrics.csv"
        split_manifest_path = bundle.root / "ml_evaluation" / "split_manifest.csv"
        portfolio_metrics_path = bundle.root / "ml_evaluation" / "portfolio_metrics.csv"
        ml_complete = (
            fold_metrics_path.exists()
            and split_manifest_path.exists()
            and portfolio_metrics_path.exists()
        )
        if ml_complete:
            fold_metrics = pd.read_csv(fold_metrics_path)
            portfolio_metrics = pd.read_csv(portfolio_metrics_path)
            ml_complete = (
                not fold_metrics.empty
                and not portfolio_metrics.empty
                and int(portfolio_metrics.iloc[0]["periods"]) > 0
            )
        checks.append(
            {
                "check": "walk_forward_metrics_complete",
                "passed": ml_complete,
                "details": str(fold_metrics_path),
            }
        )
    scenarios = set(np.round(sensitivity["delisting_return"], 2))
    checks.append(
        {
            "check": "delisting_sensitivity_complete",
            "passed": scenarios == {0.0, -0.3, -1.0},
            "details": str(sorted(scenarios)),
        }
    )
    required_metrics = {
        "cagr",
        "annualized_volatility",
        "downside_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "omega_ratio",
        "var_95_daily",
        "cvar_95_daily",
        "alpha_annualized",
        "beta",
        "correlation",
        "r_squared",
        "tracking_error",
        "information_ratio",
        "up_capture",
        "down_capture",
    }
    missing_metrics = required_metrics - set(metrics["metric"])
    checks.append(
        {
            "check": "professional_metrics_complete",
            "passed": not missing_metrics,
            "details": str(sorted(missing_metrics)),
        }
    )
    frame = pd.DataFrame(checks)
    status = "valid" if bool(frame["passed"].all()) else "invalid"
    write_csv(frame, context.path("acceptance_checks.csv"))
    context.update(
        status=status,
        quality_gates={row["check"]: bool(row["passed"]) for row in checks},
        notes=[f"Validated source run {bundle.run_id}."],
    )
    return frame, context
