from qss.config.loader import get_config


def test_config_loading_and_validation():
    config = get_config(["configs/default.yaml"])
    assert config.project.name == "quant_stock_system"
    assert config.strategy.name == "multifactor_balanced_us"
    assert "value" in config.factor_groups
    assert config.optimizer.constraints.max_weight == 0.05
