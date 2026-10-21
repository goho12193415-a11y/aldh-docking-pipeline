from __future__ import annotations

import subprocess
from pathlib import Path

from .common import require_executable


def run_vina(vina_path: str, receptor_pdbqt: Path, ligand_pdbqt: Path, box: dict[str, float], output_dir: Path, ligand_name: str, exhaustiveness: int, num_modes: int, energy_range: float, seed: int | None = None) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{ligand_name}_out.pdbqt"
    log_path = output_dir / f"{ligand_name}.log"
    executable = require_executable(vina_path)
    command = [
        executable, "--receptor", str(receptor_pdbqt), "--ligand", str(ligand_pdbqt),
        "--center_x", str(box["center_x"]), "--center_y", str(box["center_y"]), "--center_z", str(box["center_z"]),
        "--size_x", str(box["size_x"]), "--size_y", str(box["size_y"]), "--size_z", str(box["size_z"]),
        "--exhaustiveness", str(exhaustiveness), "--num_modes", str(num_modes),
        "--energy_range", str(energy_range), "--out", str(out_path),
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    result = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Vina failed for {ligand_name}; see {log_path}")
    return out_path, log_path
