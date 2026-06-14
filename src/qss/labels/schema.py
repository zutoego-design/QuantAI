from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class LabelDefinition(BaseModel):
    name: Literal["forward_return", "cross_sectional_rank", "event_window_return"]
    horizon_days: int = Field(gt=0)
    start_offset_days: int = Field(default=0, ge=0)
    embargo_days: int = Field(default=0, ge=0)
    version: str = "v1"

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("version must not be empty")
        return value.strip()
