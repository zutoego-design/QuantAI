from types import SimpleNamespace

import pandas as pd

from qss import cli


def test_monthly_pipeline_syncs_universe_before_two_year_price_warmup(
    monkeypatch,
):
    config = SimpleNamespace(
        backtest=SimpleNamespace(start_date="2016-01-01"),
        universe=SimpleNamespace(validation_provider="disabled"),
        paths=SimpleNamespace(silver_data="unused"),
    )
    calls = []

    monkeypatch.setattr(cli, "_load_app_config", lambda paths: config)
    monkeypatch.setattr(
        cli,
        "sync_universe",
        lambda *args, **kwargs: calls.append(("sync", kwargs)),
    )
    monkeypatch.setattr(
        cli,
        "_research_tickers",
        lambda *args, **kwargs: calls.append(
            ("tickers", args[1:])
        )
        or ["AAA"],
    )
    monkeypatch.setattr(
        cli,
        "ingest_prices",
        lambda *args, **kwargs: calls.append(("prices", kwargs)),
    )
    monkeypatch.setattr(
        cli,
        "ingest_fundamentals",
        lambda *args, **kwargs: calls.append(("fundamentals", kwargs)),
    )
    monkeypatch.setattr(
        cli,
        "enrich_sector_metadata",
        lambda *args, **kwargs: calls.append(("sectors", kwargs)),
    )
    monkeypatch.setattr(
        cli,
        "ingest_macro",
        lambda *args, **kwargs: calls.append(("macro", {})),
    )
    monkeypatch.setattr(
        cli,
        "build_and_store_universe",
        lambda *args, **kwargs: calls.append(("build", {})),
    )
    monkeypatch.setattr(
        cli,
        "_require_valid_research_data",
        lambda *args, **kwargs: calls.append(("validate", args[1:])),
    )
    monkeypatch.setattr(
        cli,
        "compute_and_store_factor_values",
        lambda *args, **kwargs: calls.append(("factors", {})),
    )
    monkeypatch.setattr(
        cli,
        "compute_and_store_scores",
        lambda *args, **kwargs: calls.append(("scores", {})),
    )
    monkeypatch.setattr(
        cli,
        "run_rebalance",
        lambda *args, **kwargs: calls.append(("rebalance", kwargs)),
    )
    monkeypatch.setattr(cli, "read_parquet", lambda path: pd.DataFrame())
    monkeypatch.setattr(
        cli,
        "compute_macro_regime",
        lambda *args, **kwargs: calls.append(("regime", {})),
    )

    cli.run_monthly_pipeline_cmd(
        config=["configs/default.yaml"],
        date="2026-06-12",
        start="2016-01-01",
    )

    assert [name for name, _ in calls] == [
        "sync",
        "tickers",
        "prices",
        "fundamentals",
        "sectors",
        "macro",
        "build",
        "validate",
        "factors",
        "scores",
        "rebalance",
        "regime",
    ]
    assert calls[0][1] == {
        "start_date": "2016-01-01",
        "end_date": "2026-06-12",
        "validate_recent": False,
    }
    assert calls[2][1]["start_date"] == "2014-01-01"
    assert calls[2][1]["end_date"] == "2026-06-12"
    assert calls[2][1]["tickers"] == ["AAA"]
    assert calls[7][1] == ("2016-01-01", "2026-06-12")
