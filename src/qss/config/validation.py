from __future__ import annotations

from qss.config.schema import AppConfig


def validate_config(config: dict) -> AppConfig:
    return AppConfig.model_validate(config)
