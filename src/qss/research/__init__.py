from typing import Any

__all__ = ["ExperimentSpec", "ResearchOrchestrator"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from qss.research.orchestrator import ExperimentSpec, ResearchOrchestrator

        return {
            "ExperimentSpec": ExperimentSpec,
            "ResearchOrchestrator": ResearchOrchestrator,
        }[name]
    raise AttributeError(name)
