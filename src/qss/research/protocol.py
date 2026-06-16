from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

StudyStatus = Literal["active", "closed", "superseded", "rejected_final"]


class ResearchProtocol(BaseModel):
    study_id: str
    stage: Literal["exploratory", "confirmatory"] = "exploratory"
    study_status: StudyStatus = "active"
    development_start: str
    development_end: str
    holdout_start: str | None = None
    holdout_end: str | None = None
    primary_metric: Literal[
        "total_return",
        "cagr",
        "sharpe_ratio",
        "alpha_annualized",
        "max_drawdown",
    ] = "sharpe_ratio"
    primary_metric_threshold: float = 0.0
    null_hypothesis: str = "The strategy has no positive out-of-sample investment value."
    trial_family: str
    trial_budget: int | None = Field(default=None, ge=1)
    clean_git_required: bool = False
    factor_evidence_mode: Literal["factor_level", "family_level"] = "factor_level"
    hypothesis_families: dict[str, dict[str, Any]] = Field(default_factory=dict)
    factor_directions: dict[str, Literal[-1, 1]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_windows(self) -> "ResearchProtocol":
        development_start = pd.Timestamp(self.development_start)
        development_end = pd.Timestamp(self.development_end)
        if development_end <= development_start:
            raise ValueError("development_end must be after development_start")
        if self.stage == "confirmatory":
            if not self.holdout_start or not self.holdout_end:
                raise ValueError(
                    "Confirmatory protocols require holdout_start and holdout_end."
                )
            holdout_start = pd.Timestamp(self.holdout_start)
            holdout_end = pd.Timestamp(self.holdout_end)
            if holdout_end <= holdout_start:
                raise ValueError("holdout_end must be after holdout_start")
            if holdout_start <= development_end:
                raise ValueError("The holdout period must start after development_end.")
        return self

    @property
    def spec_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_label_gap(
    protocol: ResearchProtocol,
    trading_dates: pd.DatetimeIndex,
    label_horizon_days: int,
) -> None:
    if protocol.stage != "confirmatory":
        return
    assert protocol.holdout_start is not None
    dates = pd.DatetimeIndex(pd.to_datetime(trading_dates)).sort_values().unique()
    development_end = pd.Timestamp(protocol.development_end)
    holdout_start = pd.Timestamp(protocol.holdout_start)
    development_position = int(dates.searchsorted(development_end, side="right") - 1)
    holdout_position = int(dates.searchsorted(holdout_start, side="left"))
    if development_position < 0 or holdout_position >= len(dates):
        raise ValueError("Protocol windows fall outside the available trading calendar.")
    gap = holdout_position - development_position - 1
    if gap < label_horizon_days:
        raise ValueError(
            "Development and holdout periods must be separated by at least "
            f"{label_horizon_days} trading days; observed {gap}."
        )


def exploratory_protocol(
    *,
    study_id: str,
    start_date: str,
    end_date: str,
    factors: list[str],
) -> ResearchProtocol:
    return ResearchProtocol(
        study_id=study_id,
        stage="exploratory",
        development_start=start_date,
        development_end=end_date,
        trial_family=study_id,
        factor_directions={factor: 1 for factor in factors},
    )
