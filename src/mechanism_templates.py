from __future__ import annotations

from pathlib import Path

import yaml


def load_mechanism_template(project_root: Path, family: str) -> dict:
    """Load a mechanism template without silently falling back to ALDH rules."""
    if not family:
        family = "aldh"
    path = project_root / "config" / "mechanisms" / f"{family}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Unknown mechanism family '{family}'. Expected a template at {path}."
        )
    with path.open(encoding="utf-8") as handle:
        template = yaml.safe_load(handle) or {}
    template["path"] = str(path)
    template["family"] = family
    return template


def mechanism_capability(template: dict) -> dict:
    status = template.get("status", "audit_only")
    return {
        "family": template.get("family"),
        "display_name": template.get("display_name", template.get("family")),
        "status": status,
        "implemented_scoring": status == "implemented",
        "box_strategy": template.get("box_strategy"),
        "scoring_strategy": template.get("scoring_strategy"),
        "interpretation_limits": template.get("interpretation_limits", []),
    }
