from qss.config.loader import get_config


def test_config_loading_and_validation():
    config = get_config(["configs/default.yaml"])
    assert config.project.name == "quant_stock_system"
    assert config.strategy.name == "multifactor_balanced_us"
    assert "value" in config.factor_groups
    assert config.optimizer.constraints.max_weight == 0.05
    assert config.universe.membership_mode == "point_in_time"
    assert config.universe.long_history_provider == "sp500_wikipedia"
    assert config.universe.start_date == "2016-01-01"
    assert config.universe.min_long_price_coverage == 0.80
    assert config.universe.min_recent_price_coverage == 0.98
    assert config.backtest.start_date == "2016-01-01"
    assert config.backtest.primary_benchmark == "^GSPC"
    assert config.robustness.parallel_workers == 2


def test_quickstart_config_is_isolated_and_non_research():
    config = get_config(["configs/quickstart.yaml"])
    assert config.runtime.research_mode is False
    assert config.runtime.allow_synthetic is True
    assert config.paths.silver_data == "data/quickstart/silver"
    assert config.quickstart.target_symbols == 500
    assert config.quickstart.max_symbols == 1000
    assert config.optimizer.constraints.target_num_holdings == 50
