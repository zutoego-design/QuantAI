from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReportBundle:
    run_id: str
    root: Path
    manifest: Path
    html_report: Path
    metrics: Path
    daily_returns: Path
    structured_report: Path

    def validate(self) -> list[str]:
        missing = [
            str(path)
            for path in [
                self.manifest,
                self.html_report,
                self.metrics,
                self.daily_returns,
                self.structured_report,
            ]
            if not path.exists()
        ]
        return missing
