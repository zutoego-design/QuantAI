import pandas as pd

from qss.config.loader import get_config
from qss.quickstart import _load_quickstart_master, _seed_membership


def test_quickstart_membership_covers_each_month(tmp_path):
    config = get_config(["configs/quickstart.yaml"])
    config.paths.silver_data = str(tmp_path / "silver")
    config.quickstart.universe_source = "seed"
    membership = _seed_membership(config, "2024-01-01", "2024-03-31")
    assert pd.to_datetime(membership["date"]).dt.to_period("M").nunique() == 3
    assert membership["included"].all()
    assert set(membership["source"]) == {"quickstart_current_membership"}


def test_quickstart_can_build_large_sp500_style_master(monkeypatch):
    config = get_config(["configs/quickstart.yaml"])
    config.quickstart.prefer_seed_symbols = False

    rows = 520
    table = pd.DataFrame(
        {
            "Symbol": [f"T{i:03d}" for i in range(rows)],
            "Security": [f"Company {i}" for i in range(rows)],
            "GICS Sector": ["Information Technology"] * rows,
            "GICS Sub-Industry": ["Software"] * rows,
        }
    )
    monkeypatch.setattr(pd, "read_html", lambda *_args, **_kwargs: [table])

    master = _load_quickstart_master(config, target_symbols=500)

    assert len(master) == 500
    assert master["symbol"].nunique() == 500
    assert set(master["source"]) == {"quickstart_sp500_wikipedia"}
