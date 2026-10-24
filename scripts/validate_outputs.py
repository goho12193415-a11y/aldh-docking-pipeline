from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml


def read_atoms(path: Path) -> list[str]:
    return [
        line for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]


def _manifest_config(root: Path, manifest: dict) -> Path:
    raw = str(manifest.get("config_path") or "")
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            return candidate
    return root / "config" / "pipeline_config.yaml"


def validate(root: Path) -> int:
    manifest_path = root / "reports" / "run_manifest.json"
    manifest: dict = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = _manifest_config(root, manifest)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    expected = [str(item["name"]) for item in config.get("ligands", [])]
    summary_path = root / "reports" / "docking_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    actual = [row.get("Ligand", "") for row in rows]
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ValueError(f"Configured ligands missing from summary: {missing} (summary has {actual})")
    run_id = str(manifest.get("run_id") or "")
    if run_id:
        mismatched = [row.get("Ligand") for row in rows if row.get("Run_ID") and str(row.get("Run_ID")) != run_id]
        if mismatched:
            raise ValueError(f"Summary rows from a different run_id: {mismatched}")
    cofactor_names = set(config.get("receptor", {}).get("keep_cofactors", []) or [])
    render_png = bool(config.get("visualization", {}).get("render_png", True))
    for row in rows:
        ligand = row["Ligand"]
        for field in ("Binding_Affinity_kcal_mol", "d_nuc_A"):
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"{ligand}: {field} is not finite")
        parsed_modes = int(float(row.get("Parsed_Mode_Count", "0")))
        if parsed_modes <= 0:
            raise ValueError(f"{ligand}: no docking modes were parsed")
        competent_modes = int(float(row.get("Catalytically_Competent_Mode_Count", "0")))
        if competent_modes < 0 or competent_modes > parsed_modes:
            raise ValueError(f"{ligand}: invalid competent mode count")
        if row["Catalytically_Competent"] not in {"True", "False"}:
            raise ValueError(f"{ligand}: invalid Catalytically_Competent value")
        if row.get("Selection_Status") not in {
            "Catalytically_Competent",
            "Affinity_Fallback_No_Catalytic_Pose",
        }:
            raise ValueError(f"{ligand}: invalid Selection_Status")
        active = root / "reports" / "active_poses" / f"{ligand}_active.pdb"
        atoms = read_atoms(active)
        if not any(line[17:20].strip() == "LIG" for line in atoms):
            raise ValueError(f"{ligand}: active pose has no LIG residue")
        if cofactor_names and not any(line[17:20].strip() in cofactor_names for line in atoms):
            raise ValueError(f"{ligand}: active pose has no retained cofactor from {sorted(cofactor_names)}")
        interaction = root / "reports" / "interactions" / f"{ligand}_interactions.csv"
        if not interaction.is_file():
            raise FileNotFoundError(f"{ligand}: missing interaction report")
        figure_dir = root / "reports" / "figures"
        render_script = figure_dir / f"{ligand}_render.pml"
        if not render_script.is_file() or render_script.stat().st_size == 0:
            raise ValueError(f"{ligand}: missing or empty {render_script.name}")
        status_path = figure_dir / f"{ligand}_render_status.json"
        pymol_unavailable = False
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            pymol_unavailable = status.get("pymol_available") is False
        pse = figure_dir / f"{ligand}_session.pse"
        if not pymol_unavailable and (not pse.is_file() or pse.stat().st_size == 0):
            raise ValueError(f"{ligand}: missing or empty {pse.name}")
        if render_png:
            png = figure_dir / f"{ligand}_300dpi.png"
            if not png.is_file() or png.stat().st_size == 0:
                raise ValueError(f"{ligand}: missing or empty {png.name}")
            try:
                from PIL import Image
            except ImportError:
                continue
            with Image.open(png) as image:
                if image.size != (config["visualization"]["png_width"], config["visualization"]["png_height"]):
                    raise ValueError(f"{ligand}: unexpected PNG size {image.size}")
    print(f"Validated {len(rows)} ligand result(s) against {config_path.name}, including active poses and interaction tables.")
    quality = root / "reports" / "receptor_quality.json"
    if not quality.is_file():
        raise FileNotFoundError(f"Missing receptor quality report: {quality}")
    compatibility = root / "reports" / "ligand_compatibility.json"
    if not compatibility.is_file():
        raise FileNotFoundError(f"Missing ligand compatibility report: {compatibility}")
    chemical_state = root / "reports" / "chemical_state.json"
    if not chemical_state.is_file():
        raise FileNotFoundError(f"Missing chemical state report: {chemical_state}")
    capability = root / "reports" / "capability_report.json"
    if not capability.is_file():
        raise FileNotFoundError(f"Missing capability report: {capability}")
    enzyme = root / "reports" / "enzyme_readiness.json"
    if not enzyme.is_file():
        raise FileNotFoundError(f"Missing enzyme readiness report: {enzyme}")
    gate = root / "reports" / "capability_gate.json"
    if not gate.is_file():
        raise FileNotFoundError(f"Missing capability gate report: {gate}")
    receptor_state = root / "reports" / "receptor_state.json"
    if not receptor_state.is_file():
        raise FileNotFoundError(f"Missing receptor state report: {receptor_state}")
    box_audit = root / "reports" / "docking_box_audit.json"
    if not box_audit.is_file():
        raise FileNotFoundError(f"Missing docking box audit: {box_audit}")
    execution_plan = root / "reports" / "execution_plan.json"
    if not execution_plan.is_file():
        raise FileNotFoundError(f"Missing execution plan: {execution_plan}")
    manifest = root / "reports" / "run_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing run manifest: {manifest}")
    for filename in (
        "structure_repair_audit.json",
        "multi_box_plan.json",
        "flexibility_plan.json",
        "engine_consensus_plan.json",
        "workflow_readiness.json",
        "system_coverage.json",
        "run_history.sqlite3",
    ):
        if not (root / "reports" / filename).is_file():
            raise FileNotFoundError(f"Missing system upgrade artifact: {filename}")
    for item in config.get("ligands", []):
        if item.get("name") and any(row.get("name") == item["name"] and row.get("sugar_like") for row in json.loads(compatibility.read_text(encoding="utf-8"))):
            manifest = root / "data" / "04_prepped_ligands" / "polysaccharides" / f"{item['name']}__sugar_conformers.json"
            if not manifest.is_file():
                raise FileNotFoundError(f"Missing polysaccharide conformer manifest: {manifest}")
    replicate_count = int(config.get("advanced", {}).get("replicates", {}).get("count", 1))
    if replicate_count > 1:
        consensus = root / "reports" / "consensus_summary.csv"
        if not consensus.is_file():
            raise FileNotFoundError(f"Missing replicate consensus summary: {consensus}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ALDH docking outputs.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    raise SystemExit(validate(args.root.resolve()))
