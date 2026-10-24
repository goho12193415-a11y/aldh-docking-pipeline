from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import logging
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.common import compute_admet_druglikeness, configure_logging, estimate_receptor_microenvironment_pka, require_executable, resolve_path
from src.mechanism_templates import load_mechanism_template, mechanism_capability
from src.step10_evaluation import evaluate_experimental_data
from src.step1_ligand_prep import (
    discover_raw_ligands,
    prepare_ligand,
    prepare_ligand_variants,
    prepare_polysaccharide_conformers,
    prepare_ring_conformer_ensemble,
)
from src.step2_receptor_prep import prepare_flexible_receptor_pdbqt, prepare_receptor, superimpose_receptor_kabsch
from src.step3_box_locator import discover_binding_pockets, locate_mechanism_box
from src.step4_vina_runner import run_vina, run_vina_ensemble
from src.step5_mechanism_scorer import score_docking
from src.step6_interaction_viz import analyze_interactions, generate_pymol_script
from src.step7_advanced import (
    aggregate_replicates,
    assess_enzyme_readiness,
    assess_ligand_chemical_state,
    assess_ligand_compatibility,
    assess_mol2_ligand,
    assess_structure_quality,
    audit_docking_box,
    audit_receptor_state,
    auto_configure_receptor,
    build_capability_report,
    build_execution_plan,
    detect_catalytic_cys,
    enforce_capability_gate,
    evaluate_redocking_control,
    run_optional_plip,
    write_admet_report,
    write_chemical_state_report,
    write_conformer_summary,
    write_ligand_compatibility_report,
)
from src.step8_dynamics import refine_active_pose_in_pocket, write_md_setup
from src.step9_qmmm import write_qmmm_input
from src.system_upgrade import (
    aggregate_ensemble_results,
    aggregate_multi_box_results,
    append_run_history,
    audit_structure_repair,
    build_covalent_complex_and_manifest,
    build_engine_consensus_plan,
    build_flexible_receptor_plan,
    build_multi_box_plan,
    build_workflow_readiness,
    generate_visual_html_report,
    write_system_coverage_report,
)

LOGGER = logging.getLogger("main")
_SUMMARY_LOCK = threading.Lock()


def discover_raw_receptors(project_root: Path, config: dict, cli_receptors: list[str] | None = None) -> list[Path]:
    """Resolve one or multiple receptor PDB files from CLI, config (input_pdb / input_pdbs), data/01_raw_receptors/*.pdb, and receptor_ensemble."""
    rec_cfg = config.get("receptor", {})
    candidates: list[str] = []
    if cli_receptors:
        for item in cli_receptors:
            for part in str(item).split(";"):
                if part.strip():
                    candidates.append(part.strip())
    if isinstance(rec_cfg.get("input_pdbs"), list):
        for item in rec_cfg["input_pdbs"]:
            if str(item).strip():
                candidates.append(str(item).strip())
    raw_in = rec_cfg.get("input_pdb")
    if isinstance(raw_in, list):
        for item in raw_in:
            if str(item).strip():
                candidates.append(str(item).strip())
    elif raw_in:
        for part in str(raw_in).split(";"):
            if part.strip():
                candidates.append(part.strip())

    resolved_list: list[Path] = []
    for c_str in candidates:
        p = resolve_path(project_root, c_str)
        if p not in resolved_list:
            resolved_list.append(p)

    raw_dir = project_root / "data" / "01_raw_receptors"
    if rec_cfg.get("auto_discover_receptors", True) and raw_dir.is_dir():
        for pdb_file in sorted(list(raw_dir.glob("*.pdb")) + list(raw_dir.glob("*.cif")) + list(raw_dir.glob("*.mmcif"))):
            if pdb_file not in resolved_list:
                resolved_list.append(pdb_file)

    ens_inputs = config.get("advanced", {}).get("receptor_ensemble", {}).get("inputs", []) or []
    for ens_in in ens_inputs:
        p = resolve_path(project_root, ens_in)
        if p not in resolved_list:
            resolved_list.append(p)

    return resolved_list


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def update_interaction_summary(summary_path: Path, ligand_name: str, interactions: pd.DataFrame) -> None:
    with _SUMMARY_LOCK:
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


def _pose_rank_key(summary_row: dict) -> tuple[int, float, float]:
    competent = 0 if str(summary_row.get("Catalytically_Competent")) == "True" else 1
    aff_raw = summary_row.get("Binding_Affinity_kcal_mol", "")
    aff = float(aff_raw) if aff_raw not in ("", None) else float("inf")
    nuc_raw = summary_row.get("d_nuc_A", "")
    nuc = float(nuc_raw) if nuc_raw not in ("", None) else float("inf")
    return (competent, aff, nuc)


def run(
    config_path: Path,
    skip_vina: bool = False,
    allow_specialized: bool = False,
    use_cache: bool = True,
    max_workers: int | None = None,
    cli_receptors: list[str] | None = None,
    cli_ligands: list[str] | None = None,
) -> None:
    root = config_path.parent.parent
    config = load_config(config_path)
    configure_logging(config["project"].get("log_level", "INFO"))
    all_receptors = discover_raw_receptors(root, config, cli_receptors=cli_receptors)
    if not all_receptors:
        raise FileNotFoundError("No receptor PDB found in config or data/01_raw_receptors/.")
    receptor_input = all_receptors[0]
    config["receptor"]["input_pdb"] = str(receptor_input)
    if len(all_receptors) > 1:
        ens_cfg = config.setdefault("advanced", {}).setdefault("receptor_ensemble", {})
        ens_cfg["enabled"] = True
        ens_cfg["inputs"] = [str(p) for p in all_receptors[1:]]
    receptor_out = root / "data" / "03_prepped_receptors"
    advanced = config.get("advanced", {})
    safety_cfg = config.get("safety", {})
    enzyme_cfg = config.get("enzyme", {})
    auto_remediate_chem = bool(safety_cfg.get("auto_remediate_chemical_state", True))
    experimental_ph = float(safety_cfg.get("experimental_ph", 7.4))
    workers = max(1, int(max_workers or config["project"].get("max_workers", 1)))

    configured_ligs = list(config.get("ligands", []) or [])
    for cli_lig in cli_ligands or []:
        for part in str(cli_lig).split(";"):
            tok = part.strip()
            if not tok:
                continue
            if "=" in tok:
                k, v = [x.strip() for x in tok.split("=", 1)]
                if v.lower().endswith((".mol2", ".sdf", ".mol", ".pdb", ".pdbqt")):
                    configured_ligs.append({"name": k, "file": v})
                else:
                    configured_ligs.append({"name": k, "smiles": v})
            elif tok.lower().endswith((".mol2", ".sdf", ".mol", ".pdb", ".pdbqt")):
                configured_ligs.append({"name": Path(tok).stem, "file": tok})
            else:
                configured_ligs.append({"name": tok, "smiles": tok})
    config["ligands"] = discover_raw_ligands(root / "data" / "02_raw_ligands", configured_ligs)

    mechanism = load_mechanism_template(
        root,
        enzyme_cfg.get("mechanism_family", "aldh"),
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
    auto_configure_receptor(receptor_input, config)

    cat_num = int(config["receptor"].get("catalytic_residue", {}).get("number", 302))
    template = config["receptor"].get("template_pdb")
    template_path = resolve_path(root, template) if template else None
    cleaned, receptor_pdbqt = prepare_receptor(
        receptor_input,
        receptor_out,
        config["receptor"]["chain_id"],
        set(config["receptor"]["keep_cofactors"]),
        template_path,
        config["receptor"].get("cofactor_chain"),
        enzyme_cfg.get("conserved_waters", []),
        cat_num,
        auto_retain_bridging_waters=bool(enzyme_cfg.get("auto_retain_bridging_waters", False)),
        optimize_cofactor_in_state=bool(config["receptor"].get("optimize_cofactor_in_state", False)),
    )

    pka_audit = estimate_receptor_microenvironment_pka(
        cleaned,
        chain_id=config["receptor"]["chain_id"],
        ph=experimental_ph,
        catalytic_resseq=cat_num,
    )
    (root / "reports" / "receptor_pka_audit.json").write_text(json.dumps(pka_audit, indent=2), encoding="utf-8")

    flex_residues = enzyme_cfg.get("flexible_residues", []) or []
    rigid_receptor_pdbqt, flex_receptor_pdbqt = prepare_flexible_receptor_pdbqt(
        cleaned,
        receptor_pdbqt,
        receptor_out,
        config["receptor"]["chain_id"],
        flex_residues,
    )

    if config.get("docking_box", {}).get("auto_discover_pockets"):
        discovered = discover_binding_pockets(
            cleaned,
            chain_id=config["receptor"]["chain_id"],
            top_k=int(config["docking_box"].get("max_discovered_pockets", 3)),
        )
        config.setdefault("docking_box", {})["candidates"] = discovered

    initial_box = locate_mechanism_box(cleaned, config)
    box_audit = audit_docking_box(
        cleaned,
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
    assess_structure_quality(cleaned, root / "reports" / "receptor_quality.json", config["receptor"]["chain_id"])

    compatibility = []
    chemical_states = []
    admet_rows = []
    for ligand in config["ligands"]:
        lig_file = ligand.get("file") or ligand.get("mol2")
        mol2_block = ligand.get("mol2_block")
        sdf_block = ligand.get("sdf_block")
        if mol2_block or sdf_block or (lig_file and str(lig_file).lower().endswith((".mol2", ".sdf", ".mol", ".pdb"))):
            resolved_f = resolve_path(root, str(lig_file)) if lig_file else None
            if resolved_f is not None:
                ligand["file"] = str(resolved_f)
            assessment, chemical_state = assess_mol2_ligand(
                ligand["name"],
                file_path=resolved_f,
                mol2_block=mol2_block,
                sdf_block=sdf_block,
                ph=experimental_ph,
            )
            compatibility.append(assessment)
            chemical_states.append(chemical_state)
            if assessment.get("admet"):
                admet_rows.append(assessment["admet"])
        elif ligand.get("smiles"):
            chemical_state = assess_ligand_chemical_state(
                ligand["name"],
                ligand["smiles"],
                experimental_ph,
            )
            effective_smiles = (
                chemical_state["desalted_smiles"]
                if auto_remediate_chem and chemical_state.get("was_desalted")
                else ligand["smiles"]
            )
            if auto_remediate_chem and chemical_state.get("was_desalted"):
                ligand["smiles"] = effective_smiles
            assessment = assess_ligand_compatibility(ligand["name"], effective_smiles)
            if chemical_state["flags"] and not auto_remediate_chem:
                assessment["flags"].extend(
                    f"chemical_state:{flag}" for flag in chemical_state["flags"]
                )
                if assessment["status"] == "standard_small_molecule":
                    assessment["status"] = "manual_review_required"
                assessment["standard_vina_interpretation"] = False
            compatibility.append(assessment)
            chemical_states.append(chemical_state)
            try:
                admet_rows.append(compute_admet_druglikeness(ligand["name"], smiles=effective_smiles))
            except Exception as exc:
                LOGGER.debug("Skipped ADMET for %s: %s", ligand["name"], exc)
            if assessment["flags"]:
                LOGGER.warning(
                    "%s requires review before biological interpretation: %s",
                    ligand["name"],
                    "; ".join(assessment["flags"]),
                )
    write_ligand_compatibility_report(compatibility, root / "reports" / "ligand_compatibility.json")
    write_admet_report(admet_rows, root / "reports" / "admet_druglikeness.csv")
    write_chemical_state_report(
        chemical_states,
        root / "reports" / "chemical_state.json",
    )

    enzyme_readiness = assess_enzyme_readiness(
        cleaned,
        config,
        root / "reports" / "enzyme_readiness.json",
    )
    receptor_state = audit_receptor_state(
        cleaned,
        config,
        root / "reports" / "receptor_state.json",
    )
    capability = build_capability_report(config, compatibility, enzyme_readiness)
    capability["mechanism"] = mechanism_info
    capability["receptor_state"] = {
        "status": receptor_state["status"],
        "flags": receptor_state["flags"],
    }
    capability["receptor_pka_summary"] = {
        "active_site_residue_count": pka_audit["active_site_residue_count"],
        "report": str(root / "reports" / "receptor_pka_audit.json"),
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
        "structure_repair": {
            "status": repair_audit["status"],
            "post_prep_status": repair_audit.get("post_prep_status", repair_audit["status"]),
            "flags": repair_audit["flags"],
        },
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
                "receptor_pka_audit": str(root / "reports" / "receptor_pka_audit.json"),
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
                "experimental_ph": experimental_ph,
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
        LOGGER.info(
            "Replicate stability uses conservative heuristic thresholds: affinity SD <= 1.0 kcal/mol, "
            "catalytic distance SD <= 0.75 A, competence fraction >= 0.5, symmetry-corrected 3D pose RMSD <= 2.0 A."
        )

    box = initial_box
    candidate_boxes = config.get("docking_box", {}).get("candidates", []) or []
    multi_box_rows: list[dict] = []
    ensemble_rows: list[dict] = []

    def _process_single_ligand(ligand: dict) -> None:
        ligand_assessment = next((row for row in compatibility if row["name"] == ligand["name"]), {})
        lig_file_path = resolve_path(root, str(ligand.get("file") or ligand.get("mol2"))) if (ligand.get("file") or ligand.get("mol2")) else None
        pdbqt = prepare_ligand(
            ligand["name"],
            ligand.get("smiles"),
            root / "data" / "04_prepped_ligands",
            file_path=lig_file_path,
            mol2_block=ligand.get("mol2_block"),
            sdf_block=ligand.get("sdf_block"),
        )
        sugar_conformers = []
        if (
            ligand_assessment.get("sugar_like")
            and ligand.get("smiles")
            and advanced.get("polysaccharide", {}).get("enabled", False)
        ):
            sugar_cfg = advanced.get("polysaccharide", {})
            sugar_conformers, manifest_path = prepare_polysaccharide_conformers(
                ligand["name"],
                ligand["smiles"],
                root / "data" / "04_prepped_ligands" / "polysaccharides",
                int(sugar_cfg.get("conformer_count", 24)),
                float(sugar_cfg.get("rmsd_threshold", 0.5)),
            )
            LOGGER.warning("Generated %d sugar conformers for %s; see %s", len(sugar_conformers), ligand["name"], manifest_path)
        variant_paths = []
        if advanced.get("ligand_variants", {}).get("enabled") and ligand.get("smiles"):
            variant_paths = prepare_ligand_variants(
                ligand["name"],
                ligand["smiles"],
                root / "data" / "04_prepped_ligands" / "variants",
                int(advanced["ligand_variants"].get("max_variants", 8)),
                ph=experimental_ph,
            )
        if skip_vina:
            LOGGER.info("Prepared %s; skipping Vina.", ligand["name"])
            return

        if sugar_conformers and advanced.get("polysaccharide", {}).get("dock_conformer_ensemble", True):
            conformer_rows = []
            for conformer_path in sugar_conformers:
                conformer_name = conformer_path.stem
                sugar_output, _ = run_vina(
                    config["vina"]["executable"], rigid_receptor_pdbqt, conformer_path, box,
                    root / "data" / "05_docking_outputs" / "polysaccharides",
                    conformer_name, config["vina"]["exhaustiveness"], config["vina"]["num_modes"],
                    config["vina"]["energy_range"], None, use_cache=use_cache, flex_pdbqt=flex_receptor_pdbqt,
                )
                conformer_summary = score_docking(
                    sugar_output, cleaned, ligand["name"], config,
                    root / "reports" / "polysaccharides" / conformer_name,
                )
                conformer_summary["Conformer"] = conformer_name
                conformer_rows.append(conformer_summary)
            write_conformer_summary(
                conformer_rows,
                root / "reports" / "polysaccharides" / f"{ligand['name']}_conformer_summary.csv",
            )
            LOGGER.warning("Polysaccharide results remain per-conformer; do not interpret the best Vina score as a binding free energy.")

        active_ligand_pdbqt = pdbqt
        ring_conformers: list[Path] = []
        if advanced.get("ring_ensemble", {}).get("enabled") and int(ligand_assessment.get("ring_count", 0)) >= 1 and ligand.get("smiles"):
            ring_conformers, _ = prepare_ring_conformer_ensemble(
                ligand["name"],
                ligand["smiles"],
                root / "data" / "04_prepped_ligands" / "ring_ensembles",
                num_conformers=int(advanced.get("ring_ensemble", {}).get("conformer_count", 4)),
            )
        if variant_paths and advanced.get("ligand_variants", {}).get("dock_variants", False) and len(variant_paths) > 1:
            best_var_summary = None
            best_var_pdbqt = pdbqt
            for var_pdbqt in variant_paths:
                var_name = var_pdbqt.stem
                var_out, _ = run_vina(
                    config["vina"]["executable"], rigid_receptor_pdbqt, var_pdbqt, box,
                    root / "data" / "05_docking_outputs" / "variants",
                    var_name, config["vina"]["exhaustiveness"], config["vina"]["num_modes"],
                    config["vina"]["energy_range"], None, use_cache=use_cache, flex_pdbqt=flex_receptor_pdbqt,
                )
                var_sum = score_docking(
                    var_out, cleaned, ligand["name"], config,
                    root / "reports" / "variants" / var_name,
                )
                if best_var_summary is None or _pose_rank_key(var_sum) < _pose_rank_key(best_var_summary):
                    best_var_summary = var_sum
                    best_var_pdbqt = var_pdbqt
            active_ligand_pdbqt = best_var_pdbqt

        selected_box = box
        if len(candidate_boxes) > 1:
            best_box_summary = None
            for box_idx, cand_box_cfg in enumerate(candidate_boxes, 1):
                box_name = str(cand_box_cfg.get("name", f"box{box_idx}"))
                cand_box = locate_mechanism_box(cleaned, config, cand_box_cfg)
                box_run_name = f"{ligand['name']}__{box_name}"
                box_out, _ = run_vina(
                    config["vina"]["executable"], rigid_receptor_pdbqt, active_ligand_pdbqt, cand_box,
                    root / "data" / "05_docking_outputs" / "multi_box",
                    box_run_name, config["vina"]["exhaustiveness"], config["vina"]["num_modes"],
                    config["vina"]["energy_range"], None, use_cache=use_cache, flex_pdbqt=flex_receptor_pdbqt,
                )
                box_sum = score_docking(
                    box_out, cleaned, ligand["name"], config,
                    root / "reports" / "multi_box" / box_run_name,
                )
                box_sum["Box_Name"] = box_name
                with _SUMMARY_LOCK:
                    multi_box_rows.append(box_sum)
                if best_box_summary is None or _pose_rank_key(box_sum) < _pose_rank_key(best_box_summary):
                    best_box_summary = box_sum
                    selected_box = cand_box

        replicate_count = int(advanced.get("replicates", {}).get("count", 1))
        seeds = advanced.get("replicates", {}).get("seeds", [])
        replicate_rows = []
        best_rep_output: Path | None = None
        best_rep_summary: dict | None = None

        for replicate in range(max(1, replicate_count)):
            seed = seeds[replicate % len(seeds)] if seeds else None
            run_name = ligand["name"] if replicate_count == 1 else f'{ligand["name"]}__rep{replicate + 1}'
            rep_output, _ = (run_vina_ensemble(config["vina"]["executable"], rigid_receptor_pdbqt, ring_conformers, selected_box, root / "data" / "05_docking_outputs", run_name, config["vina"]["exhaustiveness"], config["vina"]["num_modes"], config["vina"]["energy_range"], seed, use_cache=use_cache, flex_pdbqt=flex_receptor_pdbqt) if len(ring_conformers) > 1 else run_vina(
                config["vina"]["executable"],
                rigid_receptor_pdbqt,
                active_ligand_pdbqt,
                selected_box,
                root / "data" / "05_docking_outputs",
                run_name,
                config["vina"]["exhaustiveness"],
                config["vina"]["num_modes"],
                config["vina"]["energy_range"],
                seed,
                use_cache=use_cache,
                flex_pdbqt=flex_receptor_pdbqt,
            ))
            if replicate_count > 1:
                replicate_summary = score_docking(
                    rep_output,
                    cleaned,
                    ligand["name"],
                    config,
                    root / "reports" / "replicates" / run_name,
                )
                replicate_rows.append(replicate_summary)
                if best_rep_summary is None or _pose_rank_key(replicate_summary) < _pose_rank_key(best_rep_summary):
                    best_rep_summary = replicate_summary
                    best_rep_output = rep_output
            else:
                best_rep_output = rep_output

        assert best_rep_output is not None
        canonical_output = root / "data" / "05_docking_outputs" / f"{ligand['name']}_out.pdbqt"
        if replicate_count > 1:
            with _SUMMARY_LOCK:
                aggregate_replicates(replicate_rows, root / "reports" / "consensus_summary.csv")
            if best_rep_output.resolve() != canonical_output.resolve():
                shutil.copyfile(best_rep_output, canonical_output)
            output = canonical_output
        else:
            output = best_rep_output

        with _SUMMARY_LOCK:
            summary = score_docking(output, cleaned, ligand["name"], config, root / "reports")
            summary["Receptor"] = receptor_input.stem
            ensemble_rows.append(dict(summary))

        refined_pose_path = root / "reports" / "active_poses" / f"{ligand['name']}_refined.pdb"
        refinement_info = refine_active_pose_in_pocket(
            Path(summary["active_pose"]),
            refined_pose_path,
            catalytic_resseq=cat_num,
        )
        # If in-pocket L-BFGS-B rigid+torsion refinement improved pocket energy, propagate refined coordinates into active_pose for all downstream analyses
        if refinement_info.get("refined") and float(refinement_info.get("energy_improvement_kcal", 0.0)) > 0.0 and refined_pose_path.is_file():
            shutil.copyfile(refined_pose_path, Path(summary["active_pose"]))

        with _SUMMARY_LOCK:
            summary_csv_path = root / "reports" / "docking_summary.csv"
            if summary_csv_path.is_file():
                df_sum = pd.read_csv(summary_csv_path)
                mask = df_sum["Ligand"].astype(str) == ligand["name"]
                if "Receptor" not in df_sum.columns:
                    df_sum.insert(0, "Receptor", receptor_input.stem)
                else:
                    df_sum.loc[mask, "Receptor"] = receptor_input.stem
                df_sum.loc[mask, "Refinement_Delta_E_kcal"] = round(float(refinement_info.get("energy_improvement_kcal", 0.0)), 3)
                df_sum.loc[mask, "Refinement_RMSD_A"] = round(float(refinement_info.get("refinement_rmsd_A", 0.0)), 4)
                df_sum.to_csv(summary_csv_path, index=False)

        interactions = analyze_interactions(
            Path(summary["active_pose"]),
            root / "reports" / "interactions" / f'{ligand["name"]}_interactions.csv',
            cat_num,
            target_chain=config["receptor"]["chain_id"],
        )
        update_interaction_summary(root / "reports" / "docking_summary.csv", ligand["name"], interactions)
        try:
            generate_pymol_script(
                Path(summary["active_pose"]),
                interactions,
                root / "reports" / "figures",
                ligand["name"],
                config["visualization"]["pymol_executable"],
                target_chain=config["receptor"]["chain_id"],
                png_width=int(config.get("visualization", {}).get("png_width", 2400)),
                png_height=int(config.get("visualization", {}).get("png_height", 1800)),
            )
        except FileNotFoundError as exc:
            LOGGER.warning("PyMOL not available; script was not rendered: %s", exc)
        if advanced.get("plip", {}).get("enabled"):
            run_optional_plip(Path(summary["active_pose"]), root / "reports" / "plip", advanced["plip"].get("executable", "plip"))
        if advanced.get("md", {}).get("enabled"):
            write_md_setup(Path(summary["active_pose"]), root / "reports" / "md", advanced["md"])
        if advanced.get("qmmm", {}).get("enabled"):
            write_qmmm_input(Path(summary["active_pose"]), root / "reports" / "qmmm", advanced["qmmm"])
        if advanced.get("covalent", {}).get("enabled") or any(
            flag.endswith("requires_covalent_docking_or_qm_mm") for flag in ligand_assessment.get("flags", [])
        ):
            build_covalent_complex_and_manifest(Path(summary["active_pose"]), config, root / "reports" / "covalent")

    if workers > 1 and len(config["ligands"]) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_single_ligand, lig) for lig in config["ligands"]]
            for fut in futures:
                fut.result()
    else:
        for ligand in config["ligands"]:
            _process_single_ligand(ligand)

    if multi_box_rows:
        aggregate_multi_box_results(multi_box_rows, root / "reports" / "multi_box_results.csv")

    if not skip_vina and advanced.get("receptor_ensemble", {}).get("enabled"):
        for ens_idx, ens_input in enumerate(advanced.get("receptor_ensemble", {}).get("inputs", []), start=2):
            ens_path = resolve_path(root, ens_input)
            if not ens_path.is_file():
                continue
            ens_stem = ens_path.stem
            ens_out_dir = root / "data" / "03_prepped_receptors" / f"ensemble_{ens_idx}_{ens_stem}"
            ens_cleaned, ens_pdbqt = prepare_receptor(
                ens_path,
                ens_out_dir,
                config["receptor"]["chain_id"],
                set(config["receptor"]["keep_cofactors"]),
                template_path,
                config["receptor"].get("cofactor_chain"),
                enzyme_cfg.get("conserved_waters", []),
                cat_num,
            )
            kabsch_info = superimpose_receptor_kabsch(
                ens_cleaned,
                cleaned,
                ens_cleaned,
                mobile_chain=config["receptor"]["chain_id"],
                reference_chain=config["receptor"]["chain_id"],
            )
            LOGGER.info("Superimposed ensemble receptor %s onto primary frame (CA RMSD=%.3f A)", ens_stem, kabsch_info["ca_rmsd_A"])
            ens_box = locate_mechanism_box(ens_cleaned, config)
            for ligand in config["ligands"]:
                lig_pdbqt = root / "data" / "04_prepped_ligands" / f"{ligand['name']}.pdbqt"
                ens_run_name = f"{ens_stem}__{ligand['name']}"
                ens_dock_out, _ = run_vina(
                    config["vina"]["executable"],
                    ens_pdbqt,
                    lig_pdbqt,
                    ens_box,
                    root / "data" / "05_docking_outputs" / "ensemble" / ens_stem,
                    ens_run_name,
                    config["vina"]["exhaustiveness"],
                    config["vina"]["num_modes"],
                    config["vina"]["energy_range"],
                    None,
                    use_cache=use_cache,
                )
                ens_rep_dir = root / "reports" / "ensemble" / ens_stem
                ens_sum = score_docking(
                    ens_dock_out,
                    ens_cleaned,
                    ligand["name"],
                    config,
                    ens_rep_dir,
                )
                ens_sum["Receptor"] = ens_stem
                ens_refined = ens_rep_dir / "active_poses" / f"{ligand['name']}_refined.pdb"
                ens_ref_info = refine_active_pose_in_pocket(Path(ens_sum["active_pose"]), ens_refined, catalytic_resseq=cat_num)
                if ens_ref_info.get("refined") and float(ens_ref_info.get("energy_improvement_kcal", 0.0)) > 0.0 and ens_refined.is_file():
                    shutil.copyfile(ens_refined, Path(ens_sum["active_pose"]))
                ens_inter = analyze_interactions(
                    Path(ens_sum["active_pose"]),
                    ens_rep_dir / "interactions" / f"{ligand['name']}_interactions.csv",
                    cat_num,
                    target_chain=config["receptor"]["chain_id"],
                )
                update_interaction_summary(ens_rep_dir / "docking_summary.csv", ligand["name"], ens_inter)
                ensemble_rows.append(dict(ens_sum))
        if len(ensemble_rows) > len(config["ligands"]):
            aggregate_ensemble_results(ensemble_rows, root / "reports" / "ensemble_consensus.csv")

    cocrystal_pdb = receptor_out / "cocrystal_ligand.pdb"
    if not skip_vina and cocrystal_pdb.is_file():
        for ligand in config["ligands"]:
            pose_pdb = root / "reports" / "active_poses" / f"{ligand['name']}_active.pdb"
            if pose_pdb.is_file():
                redock_res = evaluate_redocking_control(
                    cocrystal_pdb,
                    pose_pdb,
                    rmsd_threshold_A=float(advanced.get("redocking_rmsd_threshold_A", 2.0)),
                    output_json=root / "reports" / "redocking_validation.json",
                )
                if redock_res.get("evaluated"):
                    LOGGER.info("Co-crystal redocking control for %s: RMSD=%.3f A (%s)", ligand["name"], redock_res["symmetry_corrected_rmsd_A"], redock_res["status"])
                    break

    if not skip_vina:
        try:
            html_report = generate_visual_html_report(root, catalytic_num=cat_num)
            LOGGER.info("Generated standalone visual HTML report: %s", html_report)
        except Exception as exc:
            LOGGER.warning("Visual HTML report generation skipped: %s", exc)

    if not skip_vina and advanced.get("evaluation", {}).get("enabled"):
        exp_csv = resolve_path(root, advanced["evaluation"].get("experimental_csv", "data/experimental_activity.csv"))
        if exp_csv.is_file() and (root / "reports" / "docking_summary.csv").is_file():
            metrics = evaluate_experimental_data(
                root / "reports" / "docking_summary.csv",
                exp_csv,
                root / "reports" / "evaluation_results.csv",
            )
            (root / "reports" / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Mechanism-Guided AutoDock Vina Pipeline v4.0")
    parser.add_argument("--config", type=Path, default=Path("config/pipeline_config.yaml"))
    parser.add_argument("--skip-vina", action="store_true", help="Prepare ligands/receptor only")
    parser.add_argument("--preflight", action="store_true", help="Check inputs and external dependencies without running")
    parser.add_argument("--allow-specialized", action="store_true", help="Explicitly allow exploratory runs for specialized systems")
    parser.add_argument("--no-cache", action="store_true", help="Force re-execution of Vina even when checkpoint cache matches")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker threads for batch ligand screening")
    parser.add_argument("--receptor", action="append", default=None, help="Receptor PDB path(s); repeat or separate with ';' for multiple receptors")
    parser.add_argument("--ligand", action="append", default=None, help="Ligand entry (path/to/file.mol2, Name=path.mol2, or Name=SMILES); repeat for multiple")
    args = parser.parse_args()
    if args.preflight:
        raise SystemExit(preflight(args.config))
    run(
        args.config,
        args.skip_vina,
        args.allow_specialized,
        use_cache=not args.no_cache,
        max_workers=args.workers,
        cli_receptors=args.receptor,
        cli_ligands=args.ligand,
    )
