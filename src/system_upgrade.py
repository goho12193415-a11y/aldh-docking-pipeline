from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from Bio.PDB import PDBParser


def audit_structure_repair(pdb_path: Path, config: dict, output_json: Path) -> dict:
    """Audit repair needs; never silently invent missing coordinates."""
    structure = PDBParser(QUIET=True).get_structure("repair_audit", str(pdb_path))
    chain_id = config.get("receptor", {}).get("chain_id", "A")
    chain = structure[0][chain_id]
    residues = [residue for residue in chain if residue.id[0] == " "]
    missing_backbone = []
    gaps = []
    previous = None
    for residue in residues:
        number = int(residue.id[1])
        if previous is not None and number > previous + 1:
            gaps.append({"from": previous, "to": number, "missing_count": number - previous - 1})
        previous = number
        missing = [atom for atom in ("N", "CA", "C", "O") if atom not in residue]
        if missing:
            missing_backbone.append({"residue": number, "name": residue.get_resname().strip(), "atoms": missing})
    altloc = [
        {"residue": residue.id[1], "atom": atom.get_name(), "altloc": atom.get_altloc()}
        for residue in residues for atom in residue if atom.get_altloc() not in {" ", ""}
    ]
    flags = []
    if missing_backbone:
        flags.append("missing_backbone_atoms_require_repair_tool")
    if gaps:
        flags.append("sequence_gaps_require_template_or_modeling")
    if altloc:
        flags.append("alternate_locations_require_deterministic_selection")
    status = "repair_required" if flags else "audit_passed"
    result = {
        "status": status,
        "pdb": str(pdb_path),
        "chain": chain_id,
        "missing_backbone_atoms": missing_backbone,
        "sequence_gaps": gaps,
        "alternate_locations": altloc[:100],
        "flags": flags,
        "repair_policy": "No coordinates are synthesized automatically; use a validated repair/modeling tool and rerun the audit.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_multi_box_plan(config: dict, output_json: Path) -> dict:
    box = config.get("docking_box", {})
    candidates = box.get("candidates") or [{
        "name": "primary",
        "center_mode": box.get("center_mode", "midpoint_if_cofactor"),
        "size_x": box.get("size_x", 18.0),
        "size_y": box.get("size_y", 18.0),
        "size_z": box.get("size_z", 18.0),
    }]
    flags = []
    if len(candidates) > 1:
        flags.append("multiple_boxes_require_per_box_comparison")
    result = {
        "status": "plan_ready",
        "box_count": len(candidates),
        "boxes": candidates,
        "flags": flags,
        "interpretation": "Box plan only; scores from different boxes are not merged without pose/site normalization.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_flexible_receptor_plan(config: dict, output_json: Path) -> dict:
    residues = config.get("enzyme", {}).get("flexible_residues", []) or []
    waters = config.get("enzyme", {}).get("conserved_waters", []) or []
    flags = []
    if residues:
        flags.append("flexible_residues_require_rotamer_or_flexible_receptor_engine")
    if waters:
        flags.append("conserved_waters_require_explicit_water_validation")
    result = {
        "status": "configured_review_required" if flags else "rigid_receptor_default",
        "flexible_residues": residues,
        "conserved_waters": waters,
        "flags": flags,
        "interpretation": "Preparation plan only; no induced-fit or water thermodynamics are inferred.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_engine_consensus_plan(config: dict, output_json: Path) -> dict:
    engines = config.get("advanced", {}).get("engines", ["vina"])
    result = {
        "status": "single_engine" if len(engines) == 1 else "multi_engine_plan",
        "engines": engines,
        "normalization": "Per-engine ranks only; raw scores must not be averaged across engines.",
        "flags": [] if len(engines) == 1 else ["external_engines_require_tool_detection_and_pose_mapping"],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_workflow_readiness(config: dict, output_json: Path) -> dict:
    advanced = config.get("advanced", {})
    workflows = {
        "polysaccharide": bool(advanced.get("polysaccharide", {}).get("enabled")),
        "covalent": bool(advanced.get("covalent", {}).get("enabled")),
        "md": bool(advanced.get("md", {}).get("enabled")),
        "qmmm": bool(advanced.get("qmmm", {}).get("enabled")),
        "plip": bool(advanced.get("plip", {}).get("enabled")),
        "experimental_evaluation": bool(advanced.get("evaluation", {}).get("enabled")),
    }
    statuses = {}
    for name, enabled in workflows.items():
        statuses[name] = "configured_external_workflow" if enabled else "disabled"
    result = {
        "statuses": statuses,
        "flags": [
            f"{name}_requires_external_execution_and_result_ingestion"
            for name, status in statuses.items()
            if status == "configured_external_workflow" and name in {"md", "qmmm", "covalent"}
        ],
        "interpretation": "A manifest or input draft is not evidence that the external workflow completed successfully.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def append_run_history(project_root: Path, manifest: dict) -> None:
    """Persist lightweight run history without requiring a new dependency."""
    db_path = project_root / "reports" / "run_history.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, created_at TEXT, config_sha256 TEXT, mode TEXT, gate_status TEXT, planned_runs INTEGER)"
        )
        connection.execute(
            "INSERT INTO runs (created_at, config_sha256, mode, gate_status, planned_runs) VALUES (?, ?, ?, ?, ?)",
            (
                manifest.get("generated_at_utc", datetime.now(timezone.utc).isoformat()),
                manifest.get("config_sha256", ""),
                manifest.get("mode", ""),
                manifest.get("capability_gate", {}).get("interpretation_status", ""),
                manifest.get("planned_run_count", 0),
            ),
        )
        connection.commit()


def write_system_coverage_report(project_root: Path, config: dict, output_json: Path) -> dict:
    """Make the 14-item upgrade boundary explicit and machine-readable."""
    advanced = config.get("advanced", {})
    items = [
        ("chemical_state", "implemented_audit", "chemical_state.json"),
        ("receptor_repair", "implemented_audit", "structure_repair_audit.json"),
        ("mechanism_templates", "implemented_aldh_plus_audit_templates", "capability_report.json"),
        ("receptor_ensemble", "execution_plan_only", "execution_plan.json"),
        ("multi_box", "execution_plan_only", "multi_box_plan.json"),
        ("flexible_receptor", "configuration_audit_only", "flexibility_plan.json"),
        ("metal_cofactor_water", "implemented_detection_audit", "receptor_state.json"),
        ("multi_engine_consensus", "rank_normalization_plan_only", "engine_consensus_plan.json"),
        ("polysaccharide", "conformer_and_gate_layer", "ligand_compatibility.json"),
        ("covalent", "warhead_detection_and_external_manifest", "workflow_readiness.json"),
        ("md_plip_qmmm", "preparation_and_readiness_layer", "workflow_readiness.json"),
        ("experimental_calibration", "metrics_module", "evaluation_results.csv"),
        ("gui_analysis", "implemented_summary_cards", "run_history.sqlite3"),
        ("project_history_recovery", "implemented_local_history", "run_history.sqlite3"),
    ]
    coverage = []
    for name, status, artifact in items:
        coverage.append({
            "item": name,
            "status": status,
            "artifact": str(project_root / "reports" / artifact),
            "external_validation_required": status not in {"implemented_audit", "implemented_detection_audit", "implemented_summary_cards", "implemented_local_history"},
        })
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "count": len(coverage),
        "items": coverage,
        "interpretation": "Coverage matrix distinguishes implemented audits from external-tool execution and does not claim unsupported scientific validation.",
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
