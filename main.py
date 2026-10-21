from __future__ import annotations

import argparse
import json
import logging
import importlib.util
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.common import configure_logging, require_executable, resolve_path
from src.step1_ligand_prep import prepare_ligand, prepare_ligand_variants, prepare_polysaccharide_conformers
from src.step2_receptor_prep import prepare_receptor
from src.step3_box_locator import locate_box
from src.step4_vina_runner import run_vina
from src.step5_mechanism_scorer import score_docking
from src.step6_interaction_viz import analyze_interactions, generate_pymol_script
from src.step7_advanced import (
    aggregate_replicates,
    assess_structure_quality,
    assess_ligand_compatibility,
    assess_ligand_chemical_state,
    assess_enzyme_readiness,
    audit_receptor_state,
    audit_docking_box,
    build_execution_plan,
    build_capability_report,
    enforce_capability_gate,
    write_conformer_summary,
    detect_catalytic_cys,
    run_optional_plip,
    write_ligand_compatibility_report,
    write_chemical_state_report,
)
from src.mechanism_templates import load_mechanism_template, mechanism_capability
from src.step8_dynamics import write_md_setup
from src.step9_qmmm import write_qmmm_input
from src.system_upgrade import (
    append_run_history,
    audit_structure_repair,
    build_engine_consensus_plan,
    build_flexible_receptor_plan,
    build_multi_box_plan,
    build_workflow_readiness,
    write_system_coverage_report,
)

LOGGER = logging.getLogger("main")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def update_interaction_summary(summary_path: Path, ligand_name: str, interactions: pd.DataFrame) -> None:
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    if "Ligand" not in summary.columns:
        return
    for column in ("HBond_Residues", "Hydrophobic_Residues"):
        if column not in summary.columns:
            summary[column] = ""
        summary[column] = summary[column].astype("object")
    mask = summary["Ligand"].astype(str) == ligand_name
    hbond = sorted(interactions.loc[interactions["Type"] == "Hydrogen_Bond", "Receptor_Res"].unique())
    hydro = sorted(interactions.loc[interactions["Type"] == "Hydrophobic_Contact", "Receptor_Res"].unique())
    summary.loc[mask, "HBond_Residues"] = ", ".join(hbond)
    summary.loc[mask, "Hydrophobic_Residues"] = ", ".join(hydro)
    summary.to_csv(summary_path, index=False)


def preflight(config_path: Path) -> int:
    root = config_path.parent.parent
    config = load_config(config_path)
    checks: list[tuple[str, bool, str]] = []
    receptor = resolve_path(root, config["receptor"]["input_pdb"])
    checks.append(("receptor PDB", receptor.is_file(), str(receptor)))
    for ensemble_input in config.get("advanced", {}).get("receptor_ensemble", {}).get("inputs", []):
        ensemble_path = resolve_path(root, ensemble_input)
        checks.append(("ensemble receptor", ensemble_path.is_file(), str(ensemble_path)))
    try:
        resolved_vina = require_executable(config["vina"]["executable"])
        checks.append(("Vina executable", True, resolved_vina))
    except FileNotFoundError:
        checks.append(("Vina executable", False, config["vina"]["executable"]))
    for module in ("rdkit", "meeko", "Bio", "pandas", "yaml"):
        checks.append((f"Python package {module}", importlib.util.find_spec(module) is not None, module))
    for executable in ("obabel",):
        try:
            resolved = require_executable(executable)
            checks.append((f"{executable} executable", True, resolved))
        except FileNotFoundError:
            checks.append((f"{executable} executable", False, executable))
    try:
        resolved_pymol = require_executable(config["visualization"]["pymol_executable"])
        checks.append(("PyMOL executable", True, resolved_pymol))
    except FileNotFoundError:
        checks.append(("PyMOL executable", False, config["visualization"]["pymol_executable"]))
    failed = 0
    for label, ok, detail in checks:
        status = "OK" if ok else "MISSING"
        print(f"[{status:7}] {label}: {detail}")
        failed += int(not ok)
    if failed:
        print(f"\nPreflight found {failed} missing requirement(s).")
        return 1
    print("\nPreflight passed.")
    return 0


def run(config_path: Path, skip_vina: bool = False, allow_specialized: bool = False) -> None:
    root = config_path.parent.parent
    config = load_config(config_path)
    configure_logging(config["project"].get("log_level", "INFO"))
    receptor_input = resolve_path(root, config["receptor"]["input_pdb"])
    receptor_out = root / "data" / "03_prepped_receptors"
    advanced = config.get("advanced", {})
    mechanism = load_mechanism_template(
        root,
        config.get("enzyme", {}).get("mechanism_family", "aldh"),
    )
    mechanism_info = mechanism_capability(mechanism)
    if advanced.get("auto_detect_catalytic_cys"):
        detected = detect_catalytic_cys(
            receptor_input,
            config["receptor"]["chain_id"],
            advanced.get("motif_patterns"),
        )
        config["receptor"]["catalytic_residue"] = detected
        LOGGER.info("Auto-detected catalytic Cys: %s", detected)
    initial_box = locate_box(
        receptor_input,
        config["receptor"]["chain_id"],
        config["receptor"]["catalytic_residue"]["number"],
        (
            config["docking_box"]["size_x"],
            config["docking_box"]["size_y"],
            config["docking_box"]["size_z"],
        ),
        config["docking_box"]["offset_outward"],
        config["docking_box"].get("center_mode", "midpoint_if_cofactor"),
    )
    box_audit = audit_docking_box(
        receptor_input,
        initial_box,
        config,
        root / "reports" / "docking_box_audit.json",
    )
    repair_audit = audit_structure_repair(
        receptor_input,
        config,
        root / "reports" / "structure_repair_audit.json",
    )
    multi_box_plan = build_multi_box_plan(config, root / "reports" / "multi_box_plan.json")
    flexible_plan = build_flexible_receptor_plan(config, root / "reports" / "flexibility_plan.json")
    engine_plan = build_engine_consensus_plan(config, root / "reports" / "engine_consensus_plan.json")
    workflow_readiness = build_workflow_readiness(config, root / "reports" / "workflow_readiness.json")
    coverage = write_system_coverage_report(root, config, root / "reports" / "system_coverage.json")
    execution_plan = build_execution_plan(
        root,
        config,
        root / "reports" / "execution_plan.json",
    )
    assess_structure_quality(receptor_input, root / "reports" / "receptor_quality.json", config["receptor"]["chain_id"])
    compatibility = []
    chemical_states = []
    for ligand in config["ligands"]:
        if ligand.get("smiles"):
            assessment = assess_ligand_compatibility(ligand["name"], ligand["smiles"])
            chemical_state = assess_ligand_chemical_state(
                ligand["name"],
                ligand["smiles"],
                float(config.get("safety", {}).get("experimental_ph", 7.4)),
            )
            if chemical_state["flags"]:
                assessment["flags"].extend(
                    f"chemical_state:{flag}" for flag in chemical_state["flags"]
                )
                if assessment["status"] == "standard_small_molecule":
                    assessment["status"] = "manual_review_required"
                assessment["standard_vina_interpretation"] = False
            compatibility.append(assessment)
            chemical_states.append(chemical_state)
            if assessment["flags"]:
                LOGGER.warning("%s requires manual review before biological interpretation: %s", ligand["name"], "; ".join(assessment["flags"]))
    write_ligand_compatibility_report(compatibility, root / "reports" / "ligand_compatibility.json")
    write_chemical_state_report(
        chemical_states,
        root / "reports" / "chemical_state.json",
    )
    enzyme_readiness = assess_enzyme_readiness(
        receptor_input,
        config,
        root / "reports" / "enzyme_readiness.json",
    )
    receptor_state = audit_receptor_state(
        receptor_input,
        config,
        root / "reports" / "receptor_state.json",
    )
    capability = build_capability_report(config, compatibility, enzyme_readiness)
    capability["mechanism"] = mechanism_info
    capability["receptor_state"] = {
        "status": receptor_state["status"],
        "flags": receptor_state["flags"],
    }
    capability["docking_box"] = {
        "status": box_audit["status"],
        "flags": box_audit["flags"],
    }
    capability["execution_plan"] = {
        "status": execution_plan["status"],
        "planned_run_count": execution_plan["planned_run_count"],
        "flags": execution_plan["flags"],
    }
    capability["system_upgrade"] = {
        "structure_repair": {"status": repair_audit["status"], "flags": repair_audit["flags"]},
        "multi_box": {"status": multi_box_plan["status"], "flags": multi_box_plan["flags"]},
        "flexibility": {"status": flexible_plan["status"], "flags": flexible_plan["flags"]},
        "engine_consensus": {"status": engine_plan["status"], "flags": engine_plan["flags"]},
        "workflow_readiness": workflow_readiness,
    }
    if box_audit["flags"]:
        capability["ready_for_standard_screening"] = False
        capability["strict_gate_recommendation"] = "block_interpretation"
        capability["non_supported_claims"].append(
            "docking_box_coverage_requires_manual_review"
        )
    if not mechanism_info["implemented_scoring"]:
        capability["ready_for_standard_screening"] = False
        capability["strict_gate_recommendation"] = "block_interpretation"
        capability["non_supported_claims"].append(
            f"mechanism_family_{mechanism_info['family']}_requires_template_specific_scoring"
        )
    (root / "reports" / "capability_report.json").write_text(
        json.dumps(capability, indent=2),
        encoding="utf-8",
    )
    try:
        gate = enforce_capability_gate(
            config,
            compatibility,
            enzyme_readiness,
            allow_specialized,
            mechanism_info,
            box_audit,
            receptor_state,
        )
    except RuntimeError:
        gate = enforce_capability_gate(
            config,
            compatibility,
            enzyme_readiness,
            True,
            mechanism_info,
            box_audit,
            receptor_state,
        )
        gate["blocked"] = True
        gate["interpretation_status"] = "blocked_before_docking"
        (root / "reports" / "capability_gate.json").write_text(
            json.dumps(gate, indent=2),
            encoding="utf-8",
        )
        raise
    gate["blocked"] = False
    (root / "reports" / "capability_gate.json").write_text(
        json.dumps(gate, indent=2),
        encoding="utf-8",
    )
    config_bytes = config_path.read_bytes()
    (root / "reports" / "run_manifest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_path": str(config_path),
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "mechanism": mechanism_info,
                "capability_gate": gate,
                "receptor_state_status": receptor_state["status"],
                "receptor_state_flags": receptor_state["flags"],
                "docking_box_status": box_audit["status"],
                "docking_box_flags": box_audit["flags"],
                "execution_plan_status": execution_plan["status"],
                "planned_run_count": execution_plan["planned_run_count"],
                "execution_plan": str(root / "reports" / "execution_plan.json"),
                "system_upgrade_reports": {
                    "structure_repair": str(root / "reports" / "structure_repair_audit.json"),
                    "multi_box": str(root / "reports" / "multi_box_plan.json"),
                    "flexibility": str(root / "reports" / "flexibility_plan.json"),
                    "engine_consensus": str(root / "reports" / "engine_consensus_plan.json"),
                    "workflow_readiness": str(root / "reports" / "workflow_readiness.json"),
                    "system_coverage": str(root / "reports" / "system_coverage.json"),
                },
                "system_coverage_count": coverage["count"],
                "experimental_ph": config.get("safety", {}).get("experimental_ph", 7.4),
                "chemical_state_report": str(root / "reports" / "chemical_state.json"),
                "mode": "preparation_only" if skip_vina else "docking",
                "allow_specialized": allow_specialized,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    append_run_history(root, json.loads((root / "reports" / "run_manifest.json").read_text(encoding="utf-8")))
    if gate["manual_review_ligands"] or gate["enzyme_review_flags"]:
        LOGGER.warning("Capability review required before biological interpretation.")
    if advanced.get("replicates", {}).get("count", 1) > 1:
        LOGGER.info("Replicate stability uses conservative heuristic thresholds: affinity SD <= 1.0 kcal/mol, catalytic distance SD <= 0.75 A, competence fraction >= 0.5.")
    template = config["receptor"].get("template_pdb")
    template_path = resolve_path(root, template) if template else None
    cleaned, receptor_pdbqt = prepare_receptor(
        receptor_input,
        receptor_out,
        config["receptor"]["chain_id"],
        set(config["receptor"]["keep_cofactors"]),
        template_path,
        config["receptor"].get("cofactor_chain"),
    )
    box = locate_box(
        cleaned,
        config["receptor"]["chain_id"],
        config["receptor"]["catalytic_residue"]["number"],
        (
            config["docking_box"]["size_x"],
            config["docking_box"]["size_y"],
            config["docking_box"]["size_z"],
        ),
        config["docking_box"]["offset_outward"],
        config["docking_box"].get("center_mode", "midpoint_if_cofactor"),
    )
    for ligand in config["ligands"]:
        ligand_assessment = next((row for row in compatibility if row["name"] == ligand["name"]), {})
        pdbqt = prepare_ligand(ligand["name"], ligand.get("smiles"), root / "data" / "04_prepped_ligands")
        sugar_conformers = []
        if ligand_assessment.get("sugar_like") and ligand.get("smiles"):
            sugar_cfg = advanced.get("polysaccharide", {})
            sugar_conformers, manifest_path = prepare_polysaccharide_conformers(
                ligand["name"],
                ligand["smiles"],
                root / "data" / "04_prepped_ligands" / "polysaccharides",
                int(sugar_cfg.get("conformer_count", 24)),
                float(sugar_cfg.get("rmsd_threshold", 0.5)),
            )
            LOGGER.warning("Generated %d sugar conformers for %s; see %s", len(sugar_conformers), ligand["name"], manifest_path)
        if advanced.get("ligand_variants", {}).get("enabled") and ligand.get("smiles"):
            prepare_ligand_variants(
                ligand["name"],
                ligand["smiles"],
                root / "data" / "04_prepped_ligands" / "variants",
                advanced["ligand_variants"].get("max_variants", 8),
            )
        if skip_vina:
            LOGGER.info("Prepared %s; skipping Vina.", ligand["name"])
            continue
        if sugar_conformers and advanced.get("polysaccharide", {}).get("dock_conformer_ensemble", True):
            conformer_rows = []
            for conformer_path in sugar_conformers:
                conformer_name = conformer_path.stem
                sugar_output, _ = run_vina(
                    config["vina"]["executable"], receptor_pdbqt, conformer_path, box,
                    root / "data" / "05_docking_outputs" / "polysaccharides",
                    conformer_name, config["vina"]["exhaustiveness"], config["vina"]["num_modes"],
                    config["vina"]["energy_range"], None,
                )
                conformer_summary = score_docking(
                    sugar_output, cleaned, ligand["name"], config,
                    root / "reports" / "polysaccharides" / conformer_name,
                )
                conformer_summary["Conformer"] = conformer_name
                conformer_rows.append(conformer_summary)
            write_conformer_summary(conformer_rows, root / "reports" / "polysaccharides" / f"{ligand['name']}_conformer_summary.csv")
            LOGGER.warning("Polysaccharide results remain per-conformer; do not interpret the best Vina score as a binding free energy.")
        replicate_count = int(advanced.get("replicates", {}).get("count", 1))
        seeds = advanced.get("replicates", {}).get("seeds", [])
        replicate_rows = []
        output = None
        for replicate in range(max(1, replicate_count)):
            seed = seeds[replicate % len(seeds)] if seeds else None
            run_name = ligand["name"] if replicate_count == 1 else f'{ligand["name"]}__rep{replicate + 1}'
            output, _ = run_vina(config["vina"]["executable"], receptor_pdbqt, pdbqt, box, root / "data" / "05_docking_outputs", run_name, config["vina"]["exhaustiveness"], config["vina"]["num_modes"], config["vina"]["energy_range"], seed)
            if replicate_count > 1:
                replicate_summary = score_docking(output, cleaned, ligand["name"], config, root / "reports" / "replicates" / run_name)
                replicate_rows.append(replicate_summary)
        assert output is not None
        if replicate_count > 1:
            aggregate_replicates(replicate_rows, root / "reports" / "consensus_summary.csv")
            output, _ = run_vina(config["vina"]["executable"], receptor_pdbqt, pdbqt, box, root / "data" / "05_docking_outputs", ligand["name"], config["vina"]["exhaustiveness"], config["vina"]["num_modes"], config["vina"]["energy_range"], seeds[0] if seeds else None)
        summary = score_docking(output, cleaned, ligand["name"], config, root / "reports")
        interactions = analyze_interactions(Path(summary["active_pose"]), root / "reports" / "interactions" / f'{ligand["name"]}_interactions.csv', config["receptor"]["catalytic_residue"]["number"])
        update_interaction_summary(root / "reports" / "docking_summary.csv", ligand["name"], interactions)
        try:
            generate_pymol_script(Path(summary["active_pose"]), interactions, root / "reports" / "figures", ligand["name"], config["visualization"]["pymol_executable"])
        except FileNotFoundError as exc:
            LOGGER.warning("PyMOL not available; script was not rendered: %s", exc)
        if advanced.get("plip", {}).get("enabled"):
            run_optional_plip(Path(summary["active_pose"]), root / "reports" / "plip", advanced["plip"].get("executable", "plip"))
        if advanced.get("md", {}).get("enabled"):
            write_md_setup(Path(summary["active_pose"]), root / "reports" / "md", advanced["md"])
        if advanced.get("qmmm", {}).get("enabled"):
            write_qmmm_input(Path(summary["active_pose"]), root / "reports" / "qmmm", advanced["qmmm"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mechanism-guided ALDH AutoDock Vina pipeline")
    parser.add_argument("--config", type=Path, default=Path("config/pipeline_config.yaml"))
    parser.add_argument("--skip-vina", action="store_true", help="Prepare ligands/receptor only")
    parser.add_argument("--preflight", action="store_true", help="Check inputs and external dependencies without running")
    parser.add_argument("--allow-specialized", action="store_true", help="Explicitly allow exploratory runs for specialized systems")
    args = parser.parse_args()
    if args.preflight:
        raise SystemExit(preflight(args.config))
    run(args.config, args.skip_vina, args.allow_specialized)
