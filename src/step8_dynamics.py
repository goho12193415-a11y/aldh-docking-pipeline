from __future__ import annotations

import json
from pathlib import Path


def write_md_setup(complex_pdb: Path, output_dir: Path, md_config: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "complex_pdb": str(complex_pdb),
        "engine": md_config.get("engine", "gromacs"),
        "steps": md_config.get("steps", 5000000),
        "temperature_K": md_config.get("temperature_K", 298.15),
        "note": "Preparation manifest only; run topology, solvation, equilibration, and production with validated force fields.",
    }
    path = output_dir / f"{complex_pdb.stem}_md_setup.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / f"{complex_pdb.stem}_md_checklist.txt").write_text(
        "1. Select protein/cofactor/ligand force fields.\n"
        "2. Parameterize ligand and verify NAD chemistry.\n"
        "3. Solvate, add ions, minimize, equilibrate NVT/NPT.\n"
        "4. Run replicate production trajectories.\n"
        "5. Analyze ligand RMSD, contacts, catalytic distances, and NAD geometry.\n",
        encoding="utf-8",
    )
    return path
