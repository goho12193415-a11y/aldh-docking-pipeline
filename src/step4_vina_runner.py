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

def merge_conformer_pdbqt_outputs(
    conformer_outputs: list[tuple[str, Path]],
    merged_out_path: Path,
    max_models: int = 20,
) -> Path:
    """Merge multi-conformer Vina output PDBQTs into a unified energy-sorted multi-MODEL PDBQT file for mechanism scoring."""
    collected_models: list[tuple[float, str, list[str]]] = []

    for conf_label, pdbqt_path in conformer_outputs:
        if not pdbqt_path.is_file():
            continue
        lines = pdbqt_path.read_text(encoding="utf-8", errors="replace").splitlines()
        cur_block: list[str] = []
        cur_score = 999.0
        in_model = False
        pending_remark_score: float | None = None

        for line in lines:
            if line.startswith("REMARK VINA RESULT:"):
                parts = line.split()
                try:
                    val = float(parts[3])
                except (IndexError, ValueError):
                    val = 999.0
                if in_model:
                    cur_score = val
                    cur_block.append(line)
                else:
                    pending_remark_score = val
            elif line.startswith("MODEL"):
                in_model = True
                cur_block = []
                if pending_remark_score is not None:
                    cur_score = pending_remark_score
                    cur_block.append(f"REMARK VINA RESULT:   {cur_score:6.2f}      0.000      0.000")
                    pending_remark_score = None
                else:
                    cur_score = 999.0
                cur_block.append(f"REMARK CONFORMER_SOURCE: {conf_label}")
            elif line.startswith("ENDMDL"):
                if in_model and cur_block:
                    collected_models.append((cur_score, conf_label, cur_block))
                in_model = False
                cur_block = []
            elif in_model:
                cur_block.append(line)

        if not collected_models and lines:
            # Single-model file without explicit MODEL/ENDMDL wrapper
            score_val = pending_remark_score if pending_remark_score is not None else -5.0
            body = [f"REMARK VINA RESULT:   {score_val:6.2f}      0.000      0.000", f"REMARK CONFORMER_SOURCE: {conf_label}"]
            body.extend(l for l in lines if not l.startswith(("MODEL", "ENDMDL", "REMARK VINA RESULT:")))
            collected_models.append((score_val, conf_label, body))

    collected_models.sort(key=lambda item: item[0])
    selected = collected_models[:max(1, int(max_models))]

    out_lines: list[str] = []
    for idx, (_, _, block_lines) in enumerate(selected, 1):
        out_lines.append(f"MODEL {idx:8d}")
        out_lines.extend(block_lines)
        out_lines.append("ENDMDL")

    merged_out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return merged_out_path


def run_vina_ensemble(
    vina_path: str,
    receptor_pdbqt: Path,
    conformer_pdbqts: list[Path],
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
    """Dock a list of ring/stereoisomer conformer PDBQT files in Vina and merge all resulting poses into a single multi-MODEL output."""
    if not conformer_pdbqts:
        raise ValueError(f"No conformer PDBQT files provided for {ligand_name}")
    if len(conformer_pdbqts) == 1:
        return run_vina(
            vina_path, receptor_pdbqt, conformer_pdbqts[0], box, output_dir, ligand_name,
            exhaustiveness, num_modes, energy_range, seed, use_cache=use_cache,
            max_retries=max_retries, flex_pdbqt=flex_pdbqt,
        )

    conf_out_dir = output_dir / "conformer_runs" / ligand_name
    conf_outputs: list[tuple[str, Path]] = []
    logs: list[str] = []

    for conf_pdbqt in conformer_pdbqts:
        conf_name = conf_pdbqt.stem
        out_p, log_p = run_vina(
            vina_path, receptor_pdbqt, conf_pdbqt, box, conf_out_dir, conf_name,
            exhaustiveness, num_modes, energy_range, seed, use_cache=use_cache,
            max_retries=max_retries, flex_pdbqt=flex_pdbqt,
        )
        conf_outputs.append((conf_name, out_p))
        if log_p.is_file():
            logs.append(f"=== {conf_name} ===\n" + log_p.read_text(encoding="utf-8", errors="replace"))

    merged_out = output_dir / f"{ligand_name}_out.pdbqt"
    merged_log = output_dir / f"{ligand_name}.log"
    merge_conformer_pdbqt_outputs(conf_outputs, merged_out, max_models=max(num_modes * 2, 18))
    merged_log.write_text("\n".join(logs) + "\n", encoding="utf-8")
    return merged_out, merged_log
