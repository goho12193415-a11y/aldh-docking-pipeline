from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path

from .common import require_executable

LOGGER = logging.getLogger(__name__)


def _compute_run_fingerprint(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    box: dict[str, float],
    exhaustiveness: int,
    num_modes: int,
    energy_range: float,
    seed: int | None,
    engine_name: str = "vina",
    flex_pdbqt: Path | None = None,
) -> str:
    hasher = hashlib.sha256()
    if receptor_pdbqt.is_file():
        hasher.update(receptor_pdbqt.read_bytes())
    if ligand_pdbqt.is_file():
        hasher.update(ligand_pdbqt.read_bytes())
    if flex_pdbqt and flex_pdbqt.is_file():
        hasher.update(flex_pdbqt.read_bytes())
    payload = json.dumps(
        {
            "engine": engine_name,
            "box": {k: round(float(v), 3) for k, v in sorted(box.items()) if isinstance(v, (int, float))},
            "exhaustiveness": int(exhaustiveness),
            "num_modes": int(num_modes),
            "energy_range": float(energy_range),
            "seed": seed,
            "has_flex": bool(flex_pdbqt and flex_pdbqt.is_file()),
        },
        sort_keys=True,
    )
    hasher.update(payload.encode("utf-8"))
    return hasher.hexdigest()


def run_vina(
    vina_path: str,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    box: dict[str, float],
    output_dir: Path,
    ligand_name: str,
    exhaustiveness: int,
    num_modes: int,
    energy_range: float,
    seed: int | None = None,
    use_cache: bool = True,
    max_retries: int = 2,
    flex_pdbqt: Path | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{ligand_name}_out.pdbqt"
    log_path = output_dir / f"{ligand_name}.log"
    cache_meta_path = output_dir / f".{ligand_name}_cache.json"

    fingerprint = _compute_run_fingerprint(
        receptor_pdbqt,
        ligand_pdbqt,
        box,
        exhaustiveness,
        num_modes,
        energy_range,
        seed,
        engine_name=Path(vina_path).stem.lower(),
        flex_pdbqt=flex_pdbqt,
    )
    if use_cache and out_path.is_file() and out_path.stat().st_size > 0:
        if cache_meta_path.is_file():
            try:
                cached = json.loads(cache_meta_path.read_text(encoding="utf-8"))
                if cached.get("fingerprint") == fingerprint:
                    LOGGER.info("Checkpoint cache hit for %s (%s); reusing %s", ligand_name, fingerprint[:10], out_path.name)
                    return out_path, log_path
            except json.JSONDecodeError:
                pass
        elif "REMARK VINA RESULT:" in out_path.read_text(encoding="utf-8", errors="replace"):
            try:
                require_executable(vina_path)
            except FileNotFoundError:
                LOGGER.info("Reusing existing validated Vina output for %s at %s", ligand_name, out_path)
                return out_path, log_path

    executable = require_executable(vina_path)
    command = [
        executable, "--receptor", str(receptor_pdbqt), "--ligand", str(ligand_pdbqt),
        "--center_x", str(box["center_x"]), "--center_y", str(box["center_y"]), "--center_z", str(box["center_z"]),
        "--size_x", str(box["size_x"]), "--size_y", str(box["size_y"]), "--size_z", str(box["size_z"]),
        "--exhaustiveness", str(exhaustiveness), "--num_modes", str(num_modes),
        "--energy_range", str(energy_range), "--out", str(out_path),
    ]
    if flex_pdbqt and flex_pdbqt.is_file():
        command.extend(["--flex", str(flex_pdbqt)])
    if seed is not None:
        command.extend(["--seed", str(seed)])

    for attempt in range(1, max(1, int(max_retries)) + 1):
        result = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
        log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        if result.returncode == 0 and out_path.is_file() and out_path.stat().st_size > 0:
            cache_meta_path.write_text(
                json.dumps({"ligand": ligand_name, "fingerprint": fingerprint, "seed": seed}, indent=2),
                encoding="utf-8",
            )
            return out_path, log_path
        if attempt < max_retries:
            LOGGER.warning("Vina attempt %d/%d failed for %s; retrying...", attempt, max_retries, ligand_name)
            time.sleep(0.5)

    raise RuntimeError(f"Vina failed for {ligand_name}; see {log_path}")
