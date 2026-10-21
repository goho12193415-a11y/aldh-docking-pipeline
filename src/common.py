from __future__ import annotations

import logging
import shutil
import sys
import sysconfig
import site
from pathlib import Path
from typing import Any

import numpy as np


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def require_executable(name_or_path: str) -> str:
    resolved = shutil.which(name_or_path)
    if resolved:
        return resolved
    path = Path(name_or_path)
    if path.exists():
        return str(path)
    if not path.is_absolute():
        candidates = [
            Path(sysconfig.get_path("scripts")) / name_or_path,
            Path(site.getuserbase()) / "Scripts" / name_or_path,
            Path(sys.executable).parent / name_or_path,
            Path(sys.executable).parent / "Scripts" / name_or_path,
        ]
        candidates.extend(
            base / "Scripts" / name_or_path
            for base in Path(site.getuserbase()).glob("Python*")
        )
        for candidate in candidates:
            for actual in (candidate, candidate.with_suffix(".exe")):
                if actual.exists():
                    return str(actual)
    raise FileNotFoundError(
        f"Required executable was not found: {name_or_path}. "
        "Install the dependency or update pipeline_config.yaml."
    )


def atom_dict(line: str) -> dict[str, Any]:
    element = line[76:78].strip() if len(line) >= 78 else ""
    return {
        "record": line[0:6].strip(),
        "serial": int(line[6:11].strip()),
        "name": line[12:16].strip(),
        "resname": line[17:20].strip(),
        "chain": line[21].strip(),
        "resseq": int(line[22:26].strip()),
        "coord": np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=float,
        ),
        "element": element or line[12:16].strip()[0],
        "line": line.rstrip("\n"),
    }
