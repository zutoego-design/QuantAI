import re
from pathlib import Path


def test_markdown_local_links_resolve():
    root = Path(__file__).resolve().parents[1]
    files = [
        root / "README.md",
        *sorted((root / "docs").rglob("*.md")),
        *sorted((root / "reports" / "research").glob("*.md")),
    ]
    missing = []
    for source in files:
        text = source.read_text(encoding="utf-8")
        for raw_target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            target = raw_target.strip().split()[0].strip("<>")
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            if target and not (source.parent / target).resolve().exists():
                missing.append(f"{source.relative_to(root)} -> {raw_target}")
    assert not missing, "\n".join(missing)


def test_docs_root_contains_only_active_documents():
    root = Path(__file__).resolve().parents[1] / "docs"
    assert {path.name for path in root.glob("*.md")} == {
        "README.md",
        "RESEARCH_METHODOLOGY.md",
        "RESEARCH_CREDIBILITY_PLAN.md",
        "OPERATIONS.md",
    }
