import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (
    PDBParser,
    atom_dict,
    compute_peoe_charges,
    enumerate_ph_protonation_states,
    enumerate_stereoisomers_and_tautomers,
    estimate_receptor_microenvironment_pka,
    parse_mol2_block,
    compute_admet_druglikeness,
    convert_cif_to_pdb,
    parse_mol2_file,
    parse_pdb_ligand_file,
    parse_sdf_block,
    parse_sdf_file,
    parse_smiles_tokens,
    generate_fallback_3d_coords,
    standardize_and_desalt_smiles,
    write_fallback_pdbqt,
)
from src.mechanism_templates import list_supported_mechanisms, load_mechanism_template, mechanism_capability
from src.step10_evaluation import _bedroc, _enrichment_factor, _pr_auc, _roc_auc, _spearman, evaluate_experimental_data
from src.step6_interaction_viz import analyze_interactions
from src.step1_ligand_prep import discover_raw_ligands, enumerate_ligand_variants, prepare_ligand, prepare_polysaccharide_conformers, prepare_ring_conformer_ensemble
from src.step2_receptor_prep import (
    detect_bridging_waters,
    prepare_flexible_receptor_pdbqt,
    prepare_receptor,
    reconstruct_missing_sidechains,
    extract_cocrystal_ligand,
    superimpose_receptor_kabsch,
    transplant_cofactor,
)
from src.step3_box_locator import discover_binding_pockets, locate_box, locate_mechanism_box
from src.step4_vina_runner import merge_conformer_pdbqt_outputs
from src.step5_mechanism_scorer import clean_ligand_pdbqt, compute_physicochemical_rescore, parse_models, parse_models_and_flex, score_docking
from src.step7_advanced import (
    aggregate_replicates,
    assess_enzyme_readiness,
    assess_ligand_chemical_state,
    assess_ligand_compatibility,
    assess_mol2_ligand,
    audit_docking_box,
    audit_receptor_state,
    build_capability_report,
    build_execution_plan,
    compute_pose_rmsd,
    detect_catalytic_cys,
    enforce_capability_gate,
    evaluate_redocking_control,
    resolve_replicate_seeds,
)
from src.step8_dynamics import refine_active_pose_in_pocket, write_md_setup
from src.step9_qmmm import write_qmmm_input
from src.system_upgrade import (
    aggregate_ensemble_results,
    audit_structure_repair,
    build_covalent_complex_and_manifest,
    build_engine_consensus_plan,
    build_flexible_receptor_plan,
    build_multi_box_plan,
    build_workflow_readiness,
    generate_visual_html_report,
)


def _pdb_atom(record, serial, name, resname, chain, resseq, x, y, z, element, altloc=" ", occ=1.00):
    return f"{record:<6}{serial:5d} {name:>4s}{altloc}{resname:>3s} {chain}{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f} 20.00          {element:>2s}"


class GeometryTests(unittest.TestCase):
    def test_ligand_cleanup_maps_autodock_elements(self):
        model = [{
            "name": "OA",
            "element": "O",
            "coord": [1.0, 2.0, 3.0],
        }]
        line = clean_ligand_pdbqt(model)[0]
        self.assertTrue(line.startswith("HETATM"))
        self.assertEqual(line[17:20], "LIG")
        self.assertEqual(line[76:78].strip(), "O")

    def test_scoring_prefers_lowest_energy_competent_pose(self):
        config = {
            "receptor": {"catalytic_residue": {"number": 302, "chain": "A"}},
            "thresholds": {
                "carbonyl_bond_min": 1.15,
                "carbonyl_bond_max": 1.35,
                "d_nuc_min": 3.0,
                "d_nuc_max": 4.5,
            },
        }
        receptor = "\n".join([
            _pdb_atom("ATOM", 1, "SG", "CYS", "A", 302, 7.0, 0, 0, "S"),
            "END",
        ]) + "\n"
        docking = "\n".join([
            "REMARK VINA RESULT: -8.0 0.000 0.000",
            "MODEL        1",
            _pdb_atom("ATOM", 1, "OA", "LIG", "L", 999, 0, 0, 0, "OA"),
            _pdb_atom("ATOM", 2, "C1", "LIG", "L", 999, 1.2, 0, 0, "C"),
            "ENDMDL",
            "REMARK VINA RESULT: -6.0 0.000 0.000",
            "MODEL        2",
            _pdb_atom("ATOM", 1, "OA", "LIG", "L", 999, 2.3, 0, 0, "OA"),
            _pdb_atom("ATOM", 2, "C1", "LIG", "L", 999, 3.5, 0, 0, "C"),
            "ENDMDL",
        ]) + "\n"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            receptor_path = root / "receptor.pdb"
            docking_path = root / "out.pdbqt"
            receptor_path.write_text(receptor, encoding="utf-8")
            docking_path.write_text(docking, encoding="utf-8")
            result = score_docking(docking_path, receptor_path, "Hexanal", config, root / "reports")
            self.assertEqual(result["Top_Affinity_Mode"], 1)
            self.assertEqual(result["Best_Catalytic_Mode"], 2)
            self.assertTrue(result["Catalytically_Competent"])

    def test_box_uses_cofactor_midpoint_when_configured(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "receptor.pdb"
            path.write_text("\n".join([
                _pdb_atom("ATOM", 1, "SG", "CYS", "A", 302, 0, 0, 0, "S"),
                _pdb_atom("HETATM", 2, "C4N", "NAD", "B", 5501, 10, 0, 0, "C"),
                "END",
            ]) + "\n", encoding="utf-8")
            box = locate_box(path, "A", 302, (18.0, 18.0, 18.0), 2.0, "midpoint_if_cofactor")
            self.assertEqual(box["center_x"], 5.0)
            self.assertEqual(box["center_y"], 0.0)
            self.assertEqual(box["center_z"], 0.0)

    def test_advanced_cys_detection_maps_sequence_position(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "receptor.pdb"
            lines = []
            sequence = [("PHE", 298), ("GLY", 299), ("ALA", 300), ("CYS", 301)]
            for serial, (resname, number) in enumerate(sequence, 1):
                lines.append(_pdb_atom("ATOM", serial, "CA", resname, "A", number, serial, 0, 0, "C"))
            path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
            result = detect_catalytic_cys(path, "A", [r"F.{0,3}G.{0,2}C"])
            self.assertEqual(result["number"], 301)

    def test_roc_auc_ranks_active_compounds(self):
        self.assertEqual(_roc_auc([1, 0, 1, 0], [4.0, 1.0, 3.0, 2.0]), 1.0)

    def test_ligand_compatibility_flags_metal_species(self):
        result = assess_ligand_compatibility("sodium_ion", "[Na+]")
        self.assertEqual(result["status"], "specialized_method_required")
        self.assertIn("metal_or_inorganic_species_requires_specialized_parameters", result["flags"])

    def test_thioether_is_not_marked_as_covalent_warhead(self):
        result = assess_ligand_compatibility("thioether", "CSC")
        self.assertNotIn("possible_covalent_reactivity_requires_covalent_docking_or_qm_mm", result["flags"])

    def test_chemical_state_audit_flags_ionizable_and_salt_inputs(self):
        result = assess_ligand_chemical_state("glycine_salt", "[NH3+]CC(=O)[O-].[Na+]", 7.4)
        self.assertEqual(result["status"], "manual_review_required")
        self.assertGreater(result["fragment_count"], 1)
        self.assertIn("ionizable_groups_detected_pka_and_protonation_state_not_predicted", result["flags"])

    def test_capability_report_blocks_specialized_ligands(self):
        result = build_capability_report({}, [{"name": "metal", "status": "specialized_method_required"}])
        self.assertEqual(result["strict_gate_recommendation"], "block_interpretation")

    def test_compatibility_flags_sugar_rich_oligosaccharide(self):
        glucose_dimer = "OC[C@H]1O[C@H](O)[C@@H](O)[C@H](O)[C@H]1O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O"
        result = assess_ligand_compatibility("sugar_dimer", glucose_dimer)
        self.assertTrue(result["sugar_like"])
        self.assertEqual(result["status"], "specialized_method_required")

    def test_polysaccharide_conformer_manifest_is_written(self):
        glucose_dimer = "OC[C@H]1O[C@H](O)[C@@H](O)[C@H](O)[C@H]1O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O"
        with TemporaryDirectory() as directory:
            paths, manifest = prepare_polysaccharide_conformers(
                "sugar_dimer", glucose_dimer, Path(directory), num_conformers=2
            )
            self.assertEqual(len(paths), 2)
            self.assertTrue(manifest.is_file())

    def test_enzyme_readiness_checks_catalytic_residue_and_cofactor(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdb = root / "enzyme.pdb"
            pdb.write_text("\n".join([
                _pdb_atom("ATOM", 1, "CA", "CYS", "A", 302, 0, 0, 0, "C"),
                _pdb_atom("ATOM", 2, "SG", "CYS", "A", 302, 1, 0, 0, "S"),
                _pdb_atom("HETATM", 3, "C4N", "NAD", "A", 5501, 3, 0, 0, "C"),
                "END",
            ]) + "\n", encoding="utf-8")
            config = {
                "receptor": {
                    "chain_id": "A",
                    "cofactor_chain": "A",
                    "keep_cofactors": ["NAD"],
                    "catalytic_residue": {"name": "CYS", "number": 302},
                },
                "enzyme": {"requires_cofactor": True},
            }
            report = assess_enzyme_readiness(pdb, config, root / "readiness.json")
            self.assertEqual(report["residue_checks"][0]["ok"], True)
            self.assertEqual(report["found_cofactors"], ["NAD"])

    def test_capability_gate_blocks_specialized_system_by_default(self):
        with self.assertRaises(RuntimeError):
            enforce_capability_gate(
                {"safety": {"strict_capability_gate": True}},
                [{"name": "sugar", "status": "specialized_method_required"}],
            )

    def test_capability_gate_records_explicit_specialized_override(self):
        result = enforce_capability_gate(
            {"safety": {"strict_capability_gate": True}},
            [{"name": "sugar", "status": "specialized_method_required"}],
            allow_specialized=True,
        )
        self.assertEqual(result["interpretation_status"], "exploratory_override")

    def test_audit_receptor_state_reports_altloc_and_cofactor(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdb = root / "state.pdb"
            pdb.write_text("\n".join([
                _pdb_atom("ATOM", 1, "CA", "CYS", "A", 302, 0, 0, 0, "C", "A"),
                _pdb_atom("HETATM", 2, "C4N", "NAD", "A", 5501, 3, 0, 0, "C"),
                "END",
            ]) + "\n", encoding="utf-8")
            report = audit_receptor_state(
                pdb,
                {
                    "receptor": {
                        "chain_id": "A",
                        "keep_cofactors": ["NAD"],
                    }
                },
                root / "state.json",
            )
            self.assertEqual(report["status"], "manual_review_required")
            self.assertEqual(report["present_cofactors"], ["NAD"])
            self.assertIn("alternate_locations_require_deterministic_selection", report["flags"])

    def test_audit_only_mechanism_is_blocked_without_override(self):
        with self.assertRaises(RuntimeError):
            enforce_capability_gate(
                {"safety": {"strict_capability_gate": True}},
                [],
                mechanism={"family": "serine_hydrolase", "implemented_scoring": False},
            )

    def test_docking_box_audit_detects_uncovered_catalytic_landmark(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdb = root / "box.pdb"
            pdb.write_text("\n".join([
                _pdb_atom("ATOM", 1, "SG", "CYS", "A", 302, 0, 0, 0, "S"),
                _pdb_atom("HETATM", 2, "C4N", "NAD", "A", 5501, 10, 0, 0, "C"),
                "END",
            ]) + "\n", encoding="utf-8")
            report = audit_docking_box(
                pdb,
                {"center_x": 0.0, "center_y": 0.0, "center_z": 0.0, "size_x": 8.0, "size_y": 8.0, "size_z": 8.0},
                {"receptor": {"chain_id": "A", "cofactor_chain": "A", "catalytic_residue": {"number": 302}, "keep_cofactors": ["NAD"]}},
                root / "box.json",
            )
            self.assertEqual(report["status"], "manual_review_required")
            self.assertIn("cofactor_key_outside_docking_box", report["flags"])

    def test_mechanism_template_declares_implementation_status(self):
        project_root = Path(__file__).resolve().parents[1]
        template = load_mechanism_template(project_root, "aldh")
        capability = mechanism_capability(template)
        self.assertTrue(capability["implemented_scoring"])
        self.assertEqual(capability["family"], "aldh")

    def test_capability_gate_preserves_receptor_state_review_flags(self):
        result = enforce_capability_gate(
            {"safety": {"strict_capability_gate": True}},
            [],
            receptor_state={"flags": ["alternate_locations_require_deterministic_selection"]},
        )
        self.assertEqual(
            result["receptor_state_review_flags"],
            ["alternate_locations_require_deterministic_selection"],
        )

    def test_system_upgrade_plans_cover_external_workflows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdb = root / "receptor.pdb"
            pdb.write_text(
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
                "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C\n"
                "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 20.00           C\n"
                "ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00 20.00           O\nEND\n",
                encoding="utf-8",
            )
            config = {
                "receptor": {"chain_id": "A"},
                "docking_box": {"candidates": [{"name": "active_site", "size_x": 18, "size_y": 18, "size_z": 18}]},
                "advanced": {
                    "engines": ["vina", "gnina"],
                    "covalent": {"enabled": True},
                    "md": {"enabled": True},
                },
            }
            repair = audit_structure_repair(pdb, config, root / "repair.json")
            boxes = build_multi_box_plan(config, root / "boxes.json")
            flexibility = build_flexible_receptor_plan(config, root / "flex.json")
            engines = build_engine_consensus_plan(config, root / "engines.json")
            workflows = build_workflow_readiness(config, root / "workflow.json")
            self.assertEqual(repair["status"], "audit_passed")
            self.assertEqual(boxes["box_count"], 1)
            self.assertEqual(flexibility["status"], "rigid_receptor_default")
            self.assertEqual(engines["status"], "multi_engine_plan")
            self.assertIn("covalent", workflows["statuses"])

    def test_evaluation_adds_pr_auc_and_enrichment(self):
        labels = [1, 0, 1, 0]
        scores = [4.0, 1.0, 3.0, 2.0]
        self.assertEqual(_pr_auc(labels, scores), 1.0)
        self.assertEqual(_enrichment_factor(labels, scores, 0.5), 2.0)

    def test_execution_plan_expands_receptors_ligands_and_seeds(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            receptor = root / "receptor.pdb"
            receptor.write_text("END\n", encoding="utf-8")
            output = root / "execution_plan.json"
            result = build_execution_plan(
                root,
                {
                    "receptor": {"input_pdb": str(receptor)},
                    "ligands": [{"name": "A", "smiles": "CC"}, {"name": "B", "smiles": "CCC"}],
                    "advanced": {"replicates": {"count": 2, "seeds": [11, 22]}},
                },
                output,
            )
            self.assertEqual(result["status"], "plan_ready")
            self.assertEqual(result["planned_run_count"], 4)
            self.assertEqual([row["seed"] for row in result["runs"]], [11, 22, 11, 22])
            self.assertTrue(output.is_file())

    def test_replicate_summary_reports_instability(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "consensus.csv"
            rows = [
                {
                    "Ligand": "A",
                    "Binding_Affinity_kcal_mol": "-8.0",
                    "Catalytically_Competent": "True",
                    "d_nuc_A": "3.2",
                    "d_hyd_A": "3.5",
                },
                {
                    "Ligand": "A",
                    "Binding_Affinity_kcal_mol": "-5.0",
                    "Catalytically_Competent": "False",
                    "d_nuc_A": "4.8",
                    "d_hyd_A": "5.2",
                },
            ]
            result = aggregate_replicates(rows, output)
            self.assertEqual(result[0]["Stability_Status"], "unstable_manual_review_required")
            self.assertGreater(result[0]["Affinity_SD_kcal_mol"], 1.0)
            self.assertEqual(result[0]["Competence_Fraction"], 0.5)

    def test_automated_desalting_and_ph_microstate_enumeration(self):
        desalt = standardize_and_desalt_smiles("[NH3+]CC(=O)[O-].[Na+]")
        self.assertTrue(desalt["was_desalted"])
        self.assertEqual(desalt["desalted_smiles"], "[NH3+]CC(=O)[O-]")
        self.assertEqual(desalt["stripped_counterions"], ["[Na+]"])
        states = enumerate_ph_protonation_states("CCCCC(=O)O", ph=7.4)
        labels = [item["label"] for item in states]
        self.assertIn("carboxylate_deprotonated", labels)
        self.assertIn("protonation_penalty_kcal_mol", states[0])

    def test_receptor_prep_resolves_altloc_and_reconstructs_missing_oxygen(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw_pdb = root / "raw.pdb"
            raw_pdb.write_text("\n".join([
                _pdb_atom("ATOM", 1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
                _pdb_atom("ATOM", 2, "CA", "ALA", "A", 1, 1.45, 0.0, 0.0, "C", "A", 0.30),
                _pdb_atom("ATOM", 3, "CA", "ALA", "A", 1, 1.48, 0.1, 0.0, "C", "B", 0.70),
                _pdb_atom("ATOM", 4, "C", "ALA", "A", 1, 2.50, 1.1, 0.0, "C"),
                "END",
            ]) + "\n", encoding="utf-8")
            cleaned, _ = prepare_receptor(raw_pdb, root / "out", "A", {"NAD"})
            lines = [ln for ln in cleaned.read_text(encoding="utf-8").splitlines() if ln.startswith("ATOM")]
            atom_names = [ln[12:16].strip() for ln in lines]
            altlocs = {ln[16:17] for ln in lines}
            self.assertEqual(atom_names, ["N", "CA", "C", "O"])
            self.assertEqual(altlocs, {" "})

    def test_sequence_aligned_cofactor_transplantation_handles_insertion_offset(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target_pdb = root / "target.pdb"
            template_pdb = root / "template.pdb"
            motif = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "ILE", "LEU", "LYS", "PHE"]
            target_lines = [
                _pdb_atom("ATOM", 1, "CA", "GLY", "A", 1, -5.0, 0.0, 0.0, "C"),
                _pdb_atom("ATOM", 2, "CA", "GLY", "A", 2, -2.5, 0.0, 0.0, "C"),
            ]
            template_lines = []
            for idx, rname in enumerate(motif, start=1):
                x = float(idx * 3.8)
                target_lines.append(_pdb_atom("ATOM", idx + 2, "CA", rname, "A", idx + 2, x, 0.0, 0.0, "C"))
                template_lines.append(_pdb_atom("ATOM", idx, "CA", rname, "A", idx, x, 0.0, 0.0, "C"))
            template_lines.append(_pdb_atom("HETATM", 99, "C4N", "NAD", "A", 501, 20.0, 8.0, 0.0, "C"))
            target_pdb.write_text("\n".join(target_lines + ["END"]) + "\n", encoding="utf-8")
            template_pdb.write_text("\n".join(template_lines + ["END"]) + "\n", encoding="utf-8")
            out_pdb = root / "transplanted.pdb"
            audit = transplant_cofactor(target_pdb, template_pdb, out_pdb, "A", {"NAD"})
            self.assertTrue(audit["transplanted"])
            self.assertLess(audit["superposition_rmsd_A"], 0.01)
            self.assertEqual(audit["aligned_ca_count"], 12)

    def test_all_nine_universal_mechanism_templates_are_implemented(self):
        project_root = Path(__file__).resolve().parents[1]
        families = list_supported_mechanisms()
        self.assertEqual(len(families), 9)
        for family in families:
            tmpl = load_mechanism_template(project_root, family)
            cap = mechanism_capability(tmpl)
            self.assertTrue(cap["implemented_scoring"], f"{family} should have implemented_scoring=True")

    def test_contracted_nad_pose_is_reported_but_not_competent(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            receptor_path = root / "receptor.pdb"
            docking_path = root / "out.pdbqt"
            receptor_path.write_text("\n".join([
                _pdb_atom("ATOM", 1, "N", "CYS", "A", 302, 69.0, 58.0, 65.0, "N"),
                _pdb_atom("ATOM", 2, "SG", "CYS", "A", 302, 70.244, 58.419, 65.954, "S"),
                _pdb_atom("ATOM", 3, "ND2", "ASN", "A", 169, 74.8, 56.5, 66.2, "N"),
                _pdb_atom("HETATM", 4, "C4N", "NAD", "A", 5501, 73.787, 63.159, 62.551, "C"),
                "END",
            ]) + "\n", encoding="utf-8")
            docking_path.write_text("\n".join([
                "REMARK VINA RESULT: -4.533 0.000 0.000",
                "MODEL        1",
                _pdb_atom("ATOM", 1, "OA", "LIG", "L", 999, 74.20, 57.60, 66.30, "OA"),
                _pdb_atom("ATOM", 2, "C1", "LIG", "L", 999, 73.58, 58.61, 66.68, "C"),
                "ENDMDL",
            ]) + "\n", encoding="utf-8")
            config = {
                "receptor": {
                    "chain_id": "A",
                    "cofactor_chain": "A",
                    "catalytic_residue": {"name": "CYS", "number": 302, "chain": "A"},
                    "oxyanion_residue": {"name": "ASN", "number": 169, "atom": "ND2"},
                },
                "enzyme": {"mechanism_family": "aldh"},
                "thresholds": {
                    "carbonyl_bond_min": 1.15,
                    "carbonyl_bond_max": 1.35,
                    "d_nuc_min": 3.0,
                    "d_nuc_max": 4.5,
                    "d_hyd_max": 4.5,
                },
            }
            res = score_docking(docking_path, receptor_path, "Octanal", config, root / "reports")
            self.assertFalse(res["Catalytically_Competent"])
            self.assertFalse(res["Hydride_Transfer_Competent"])
            self.assertTrue(res["Nucleophile_Competent"])
            self.assertEqual(res["Cofactor_Conformation"], "contracted_hydrolysis_out")
            self.assertEqual(res["Effective_d_hyd_Max_A"], 4.5)
            self.assertEqual(res["Reliability_Grade"], "C_Low_Confidence_Affinity_Fallback")
            self.assertIn("PhysChem_Score_kcal_mol", res)
            self.assertIn("Universal_Consensus_Rank", res)

    def test_pose_3d_rmsd_and_qmmm_md_extraction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pose1 = root / "pose1.pdb"
            pose2 = root / "pose2.pdb"
            pose1.write_text("\n".join([
                _pdb_atom("ATOM", 1, "SG", "CYS", "A", 302, 0.0, 0.0, 0.0, "S"),
                _pdb_atom("HETATM", 2, "C4N", "NAD", "A", 5501, 4.0, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 9000, "C1", "LIG", "L", 999, 3.2, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 9001, "O1", "LIG", "L", 999, 3.2, 1.22, 0.0, "O"),
                "END",
            ]) + "\n", encoding="utf-8")
            pose2.write_text("\n".join([
                _pdb_atom("ATOM", 1, "SG", "CYS", "A", 302, 0.0, 0.0, 0.0, "S"),
                _pdb_atom("HETATM", 2, "C4N", "NAD", "A", 5501, 4.0, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 9000, "C1", "LIG", "L", 999, 3.2, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 9001, "O1", "LIG", "L", 999, 3.2, 1.22, 0.0, "O"),
                "END",
            ]) + "\n", encoding="utf-8")
            self.assertEqual(compute_pose_rmsd(pose1, pose2), 0.0)
            qmmm_inp = write_qmmm_input(pose1, root / "qmmm", {"charge": 0, "multiplicity": 1, "qm_residues": ["CYS302", "NAD5501"]})
            self.assertIn("Extracted QM atom count: 4", qmmm_inp.read_text(encoding="utf-8"))
            md_json = write_md_setup(pose1, root / "md", {"engine": "gromacs", "steps": 1000, "temperature_K": 300.0})
            self.assertTrue((root / "md" / "pose1_ligand.pdb").is_file())
            self.assertTrue(md_json.is_file())

    def test_3d_microenvironment_pka_and_missing_sidechain_reconstruction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdb = root / "cys_incomplete.pdb"
            pdb.write_text("\n".join([
                _pdb_atom("ATOM", 1, "N", "CYS", "A", 302, 0.0, 0.0, 0.0, "N"),
                _pdb_atom("ATOM", 2, "CA", "CYS", "A", 302, 1.46, 0.0, 0.0, "C"),
                _pdb_atom("ATOM", 3, "C", "CYS", "A", 302, 2.10, 1.40, 0.0, "C"),
                _pdb_atom("ATOM", 4, "O", "CYS", "A", 302, 1.50, 2.45, 0.0, "O"),
                _pdb_atom("ATOM", 5, "ND2", "ASN", "A", 169, 2.0, -2.2, 1.2, "N"),
                _pdb_atom("HETATM", 6, "C4N", "NAD", "A", 501, 2.5, -1.8, -1.5, "C"),
                _pdb_atom("HETATM", 7, "O", "HOH", "A", 601, 0.0, -2.8, 0.0, "O"),
                "END",
            ]) + "\n", encoding="utf-8")
            structure = PDBParser(QUIET=True).get_structure("test", str(pdb))
            reconstructed = reconstruct_missing_sidechains(structure, "A")
            reconstructed_atoms = {item["reconstructed_atom"] for item in reconstructed}
            self.assertIn("CB", reconstructed_atoms)
            self.assertIn("SG", reconstructed_atoms)
            waters = detect_bridging_waters(structure, "A", {"NAD"})
            self.assertEqual(len(waters), 1)

            cleaned, _ = prepare_receptor(pdb, root / "prep", "A", {"NAD"}, catalytic_residue_number=302, auto_retain_bridging_waters=True)
            pka_report = estimate_receptor_microenvironment_pka(cleaned, "A", 7.4, catalytic_resseq=302)
            cys_entry = next(r for r in pka_report["residues"] if r["number"] == 302)
            self.assertLess(cys_entry["predicted_pka"], 7.4)
            self.assertEqual(cys_entry["assigned_state_at_ph"], "CYM_thiolate(-1)")

    def test_flexible_receptor_splitter_and_blind_cavity_discovery(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdb = root / "pocket.pdb"
            pdb.write_text("\n".join([
                _pdb_atom("ATOM", 1, "N", "PHE", "A", 170, 0.0, 0.0, 0.0, "N"),
                _pdb_atom("ATOM", 2, "CA", "PHE", "A", 170, 1.45, 0.0, 0.0, "C"),
                _pdb_atom("ATOM", 3, "C", "PHE", "A", 170, 2.10, 1.35, 0.0, "C"),
                _pdb_atom("ATOM", 4, "O", "PHE", "A", 170, 1.50, 2.40, 0.0, "O"),
                _pdb_atom("ATOM", 5, "CB", "PHE", "A", 170, 1.95, -0.80, 1.20, "C"),
                _pdb_atom("ATOM", 6, "CG", "PHE", "A", 170, 3.40, -0.90, 1.35, "C"),
                _pdb_atom("ATOM", 7, "CD1", "PHE", "A", 170, 4.10, -1.80, 0.60, "C"),
                "END",
            ]) + "\n", encoding="utf-8")
            cleaned, pdbqt = prepare_receptor(pdb, root / "prep", "A", set())
            rigid, flex = prepare_flexible_receptor_pdbqt(cleaned, pdbqt, root / "prep", "A", [170])
            self.assertIsNotNone(flex)
            assert flex is not None
            flex_txt = flex.read_text(encoding="utf-8")
            self.assertIn("BEGIN_RES PHE A  170", flex_txt)
            # Verify nested chi2 BRANCH (CB -> CG) is emitted
            self.assertEqual(flex_txt.count("BRANCH"), 4)
            pockets = discover_binding_pockets(cleaned, "A", top_k=1)
            self.assertEqual(len(pockets), 1)
            gen_box = locate_mechanism_box(cleaned, {"enzyme": {"mechanism_family": "general_binding"}, "receptor": {"chain_id": "A"}})
            self.assertIn("center_x", gen_box)

    def test_covalent_adduct_builder_and_in_pocket_lbfgsb_refinement(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pose = root / "complex.pdb"
            pose.write_text("\n".join([
                _pdb_atom("ATOM", 1, "SG", "CYS", "A", 302, 0.0, 0.0, 0.0, "S"),
                _pdb_atom("ATOM", 2, "ND2", "ASN", "A", 169, 3.2, 2.1, 0.0, "N"),
                _pdb_atom("HETATM", 9000, "C1", "LIG", "L", 999, 2.15, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 9001, "O1", "LIG", "L", 999, 2.85, 1.05, 0.0, "O"),
                "END",
            ]) + "\n", encoding="utf-8")
            cov_manifest = build_covalent_complex_and_manifest(
                pose,
                {"receptor": {"catalytic_residue": {"name": "CYS", "number": 302, "chain": "A"}}},
                root / "covalent",
            )
            self.assertTrue(cov_manifest.is_file())
            manifest_data = json.loads(cov_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest_data["warhead_mechanism_type"], "carbonyl_sp2_to_sp3_hemithioacetal")
            self.assertAlmostEqual(manifest_data["rehybridized_tetrahedral_angle_deg"], 109.47, delta=1.5)
            adduct_pdb = root / "covalent" / "complex_covalent_adduct.pdb"
            self.assertTrue(adduct_pdb.is_file())
            self.assertIn("CONECT    1 9000", adduct_pdb.read_text(encoding="utf-8"))

            refinement = refine_active_pose_in_pocket(pose, root / "complex_refined.pdb", catalytic_resseq=302)
            self.assertTrue(refinement["refined"])
            self.assertLessEqual(refinement["refined_pocket_score"], refinement["initial_pocket_score"])

    def test_flex_output_parsing_separates_flex_residues_from_ligand_and_merges(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            receptor_pdb = root / "receptor.pdb"
            docking_pdbqt = root / "flex_out.pdbqt"
            receptor_pdb.write_text("\n".join([
                _pdb_atom("ATOM", 1, "SG", "CYS", "A", 302, 7.0, 0.0, 0.0, "S"),
                _pdb_atom("ATOM", 2, "CA", "PHE", "A", 170, 5.0, 2.0, 0.0, "C"),
                _pdb_atom("ATOM", 3, "CB", "PHE", "A", 170, 5.0, 3.5, 0.0, "C"),
                _pdb_atom("ATOM", 4, "CG", "PHE", "A", 170, 5.0, 5.0, 0.0, "C"),
                "END",
            ]) + "\n", encoding="utf-8")
            docking_pdbqt.write_text("\n".join([
                "REMARK VINA RESULT: -6.5 0.000 0.000",
                "MODEL        1",
                "ROOT",
                _pdb_atom("ATOM", 1, "OA", "UNL", " ", 1, 2.3, 0.0, 0.0, "OA"),
                _pdb_atom("ATOM", 2, "C1", "UNL", " ", 1, 3.5, 0.0, 0.0, "C"),
                "ENDROOT",
                "TORSDOF 1",
                "BEGIN_RES PHE A  170",
                "ROOT",
                _pdb_atom("ATOM", 3, "CA", "PHE", "A", 170, 5.0, 2.0, 0.0, "C"),
                "ENDROOT",
                "BRANCH   3   4",
                _pdb_atom("ATOM", 4, "CB", "PHE", "A", 170, 5.0, 3.5, 0.0, "C"),
                _pdb_atom("ATOM", 5, "CG", "PHE", "A", 170, 6.2, 4.2, 1.1, "C"),
                "ENDBRANCH   3   4",
                "END_RES PHE A  170",
                "ENDMDL",
            ]) + "\n", encoding="utf-8")
            models = parse_models(docking_pdbqt)
            self.assertEqual(len(models[0]), 2)
            pairs = parse_models_and_flex(docking_pdbqt)
            self.assertEqual(len(pairs[0][0]), 2)
            self.assertEqual(len(pairs[0][1]), 3)
            config = {
                "receptor": {"chain_id": "A", "catalytic_residue": {"number": 302, "chain": "A"}},
                "thresholds": {"carbonyl_bond_min": 1.15, "carbonyl_bond_max": 1.35, "d_nuc_min": 3.0, "d_nuc_max": 4.5},
            }
            summary = score_docking(docking_pdbqt, receptor_pdb, "FlexLig", config, root / "reports")
            active_lines = Path(summary["active_pose"]).read_text(encoding="utf-8").splitlines()
            lig_lines = [ln for ln in active_lines if ln.startswith("HETATM") and ln[17:20].strip() == "LIG"]
            phe_cg_lines = [ln for ln in active_lines if ln.startswith("ATOM") and ln[17:20].strip() == "PHE" and ln[12:16].strip() == "CG"]
            self.assertEqual(len(lig_lines), 2)
            self.assertIn("6.200   4.200   1.100", phe_cg_lines[0])

    def test_vectorized_cavity_scanner_covers_spans_beyond_100_angstroms(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdb = root / "large_dimer.pdb"
            lines = [_pdb_atom("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C")]
            # Build a spherical shell of 24 protein atoms centered at (125.0, 0.0, 0.0) with radius 4.0 A
            serial = 2
            for dx in (-4.0, 0.0, 4.0):
                for dy in (-4.0, 0.0, 4.0):
                    for dz in (-4.0, 0.0, 4.0):
                        if dx == 0.0 and dy == 0.0 and dz == 0.0:
                            continue
                        lines.append(_pdb_atom("ATOM", serial, "CA", "LEU", "A", serial, 125.0 + dx, dy, dz, "C"))
                        serial += 1
            lines.append(_pdb_atom("HETATM", 99, "C4N", "NAD", "A", 501, 125.5, 0.5, 0.0, "C"))
            pdb.write_text("\n".join(lines + ["END"]) + "\n", encoding="utf-8")
            pockets = discover_binding_pockets(pdb, "A", top_k=1)
            self.assertGreater(pockets[0]["center_x"], 115.0)

    def test_lbfgsb_refinement_optimizes_internal_rotatable_torsions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pose = root / "torsion_complex.pdb"
            pose.write_text("\n".join([
                _pdb_atom("ATOM", 1, "SG", "CYS", "A", 302, 0.0, 0.0, 0.0, "S"),
                _pdb_atom("ATOM", 2, "CB", "PHE", "A", 170, 5.2, 1.5, 0.0, "C"),
                _pdb_atom("HETATM", 9000, "O1", "LIG", "L", 999, 2.8, 1.2, 0.0, "O"),
                _pdb_atom("HETATM", 9001, "C1", "LIG", "L", 999, 3.2, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 9002, "C2", "LIG", "L", 999, 4.7, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 9003, "C3", "LIG", "L", 999, 5.2, 1.4, 0.0, "C"),
                _pdb_atom("HETATM", 9004, "C4", "LIG", "L", 999, 6.7, 1.4, 0.0, "C"),
                "END",
            ]) + "\n", encoding="utf-8")
            res = refine_active_pose_in_pocket(pose, root / "refined.pdb", catalytic_resseq=302)
            self.assertTrue(res["refined"])
            self.assertGreaterEqual(res["optimized_torsion_count"], 1)
            self.assertLess(res["refined_pocket_score"], res["initial_pocket_score"])

    def test_peoe_topological_charges_and_coupled_zwitterion_microstates(self):
        parsed = parse_smiles_tokens("CCC=O")
        charges = compute_peoe_charges(parsed["atoms"], parsed["adj"])
        # Carbonyl oxygen (index 3) is negative and carbonyl carbon (index 2) is positive
        self.assertLess(charges[3], -0.15)
        self.assertGreater(charges[2], 0.05)

        zwitter = enumerate_ph_protonation_states("NCC(=O)O", ph=7.4)
        smiles_set = {item["smiles"] for item in zwitter}
        self.assertIn("[NH3+]CC(=O)[O-]", smiles_set)

    def test_symmetry_corrected_hungarian_rmsd_handles_flipped_carboxylate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pose1 = root / "p1.pdb"
            pose2 = root / "p2.pdb"
            pose1.write_text("\n".join([
                _pdb_atom("HETATM", 9000, "C1", "LIG", "L", 999, 0.0, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 9001, "C2", "LIG", "L", 999, 1.5, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 9002, "O1", "LIG", "L", 999, 2.1, 1.1, 0.0, "O"),
                _pdb_atom("HETATM", 9003, "O2", "LIG", "L", 999, 2.1, -1.1, 0.0, "O"),
                "END",
            ]) + "\n", encoding="utf-8")
            # Swap O1 and O2 coordinates (180-degree carboxylate flip)
            pose2.write_text("\n".join([
                _pdb_atom("HETATM", 9000, "C1", "LIG", "L", 999, 0.0, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 9001, "C2", "LIG", "L", 999, 1.5, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 9002, "O1", "LIG", "L", 999, 2.1, -1.1, 0.0, "O"),
                _pdb_atom("HETATM", 9003, "O2", "LIG", "L", 999, 2.1, 1.1, 0.0, "O"),
                "END",
            ]) + "\n", encoding="utf-8")
            naive_rmsd = compute_pose_rmsd(pose1, pose2, symmetry_corrected=False)
            sym_rmsd = compute_pose_rmsd(pose1, pose2, symmetry_corrected=True)
            assert naive_rmsd is not None and sym_rmsd is not None
            self.assertGreater(naive_rmsd, 1.5)
            self.assertEqual(sym_rmsd, 0.0)


    def test_stereoisomer_and_tautomer_enumeration_expands_chiral_and_keto_enol(self):
        # Unassigned chiral center in lactic acid CC(O)C(=O)O should expand into [C@H] and [C@@H] stereoisomers
        iso_list = enumerate_stereoisomers_and_tautomers("CC(O)C(=O)O", max_isomers=8)
        smiles_set = {item["smiles"] for item in iso_list}
        self.assertTrue(any("@H" in s or "@" in s for s in smiles_set))
        # Imidazole ring tautomerism and keto-enol tautomerism
        taut_list = enumerate_stereoisomers_and_tautomers("CCC=O", max_isomers=8)
        self.assertTrue(any(item["category"] == "tautomer" for item in taut_list))
        variants = enumerate_ligand_variants("CC(O)C(=O)O", max_variants=8, ph=7.4)
        self.assertGreaterEqual(len(variants), 3)

    def test_ring_conformer_ensemble_generation_and_multi_model_pdbqt_merge(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdbqts, manifest = prepare_ring_conformer_ensemble(
                "cyclohexane_carbaldehyde", "O=CC1CCCCC1", root / "confs",
                num_conformers=3, rmsd_threshold=0.0,
            )
            self.assertEqual(len(pdbqts), 3)
            self.assertTrue(manifest.is_file())
            out1 = root / "conf1_out.pdbqt"
            out2 = root / "conf2_out.pdbqt"
            out1.write_text("\n".join([
                "MODEL 1",
                "REMARK VINA RESULT:   -6.10      0.000      0.000",
                _pdb_atom("HETATM", 1, "C1", "UNL", "L", 1, 1.0, 1.0, 1.0, "C"),
                "ENDMDL",
            ]) + "\n", encoding="utf-8")
            out2.write_text("\n".join([
                "MODEL 1",
                "REMARK VINA RESULT:   -7.40      0.000      0.000",
                _pdb_atom("HETATM", 1, "C1", "UNL", "L", 1, 2.0, 2.0, 2.0, "C"),
                "ENDMDL",
            ]) + "\n", encoding="utf-8")
            merged = merge_conformer_pdbqt_outputs([("conf1", out1), ("conf2", out2)], root / "merged_out.pdbqt")
            models = parse_models(merged)
            self.assertEqual(len(models), 2)
            # Lowest energy (-7.40 from conf2) must be sorted first (MODEL 1)
            self.assertAlmostEqual(float(models[0][0]["coord"][0]), 2.0, places=2)

    def test_kabsch_receptor_superposition_and_boltzmann_ensemble_scoring(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ref_pdb = root / "ref.pdb"
            mob_pdb = root / "mob.pdb"
            out_pdb = root / "aligned.pdb"
            ref_lines = [
                _pdb_atom("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
                _pdb_atom("ATOM", 2, "CA", "GLY", "A", 2, 3.8, 0.0, 0.0, "C"),
                _pdb_atom("ATOM", 3, "CA", "SER", "A", 3, 3.8, 3.8, 0.0, "C"),
                _pdb_atom("ATOM", 4, "CA", "CYS", "A", 4, 0.0, 3.8, 3.8, "C"),
                "END",
            ]
            # Mobile receptor shifted by (+25.0, -15.0, +10.0)
            mob_lines = [
                _pdb_atom("ATOM", 1, "CA", "ALA", "A", 1, 25.0, -15.0, 10.0, "C"),
                _pdb_atom("ATOM", 2, "CA", "GLY", "A", 2, 28.8, -15.0, 10.0, "C"),
                _pdb_atom("ATOM", 3, "CA", "SER", "A", 3, 28.8, -11.2, 10.0, "C"),
                _pdb_atom("ATOM", 4, "CA", "CYS", "A", 4, 25.0, -11.2, 13.8, "C"),
                "END",
            ]
            ref_pdb.write_text("\n".join(ref_lines) + "\n", encoding="utf-8")
            mob_pdb.write_text("\n".join(mob_lines) + "\n", encoding="utf-8")
            info = superimpose_receptor_kabsch(mob_pdb, ref_pdb, out_pdb, "A", "A")
            self.assertTrue(info["aligned"])
            self.assertLess(info["ca_rmsd_A"], 0.01)

            ens_rows = [
                {"Ligand": "Hexanal", "Receptor": "conf_open", "Binding_Affinity_kcal_mol": -6.0, "Catalytically_Competent": True},
                {"Ligand": "Hexanal", "Receptor": "conf_closed", "Binding_Affinity_kcal_mol": -8.0, "Catalytically_Competent": True},
            ]
            summary = aggregate_ensemble_results(ens_rows, root / "ensemble.csv")
            self.assertEqual(summary[0]["Boltzmann_Dominant_Receptor"], "conf_closed")
            self.assertLess(float(summary[0]["Boltzmann_Weighted_Affinity_kcal_mol"]), -7.5)
            self.assertGreater(float(summary[0]["Boltzmann_Dominant_Weight"]), 0.90)

    def test_cocrystal_ligand_extraction_and_redocking_rmsd_gate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            holo_pdb = root / "holo.pdb"
            holo_pdb.write_text("\n".join([
                _pdb_atom("ATOM", 1, "SG", "CYS", "A", 302, 0.0, 0.0, 0.0, "S"),
                _pdb_atom("HETATM", 2, "C4N", "NAD", "A", 501, 4.5, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 3, "S", "SO4", "A", 601, 15.0, 0.0, 0.0, "S"),
                _pdb_atom("HETATM", 4, "C1", "SUB", "A", 701, 2.5, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 5, "O1", "SUB", "A", 701, 2.5, 1.23, 0.0, "O"),
                _pdb_atom("HETATM", 6, "C2", "SUB", "A", 701, 3.8, -0.5, 0.0, "C"),
                _pdb_atom("HETATM", 7, "C3", "SUB", "A", 701, 5.0, 0.2, 0.0, "C"),
                "END",
            ]) + "\n", encoding="utf-8")
            cocrystal_out = root / "cocrystal.pdb"
            extracted = extract_cocrystal_ligand(holo_pdb, cocrystal_out, chain_id="A", keep_cofactors={"NAD"})
            self.assertTrue(extracted["found"])
            self.assertEqual(extracted["resname"], "SUB")
            self.assertEqual(extracted["heavy_atom_count"], 4)

            docked_good = root / "docked_good.pdb"
            docked_good.write_text("\n".join([
                _pdb_atom("HETATM", 9000, "C1", "LIG", "L", 999, 2.6, 0.1, 0.0, "C"),
                _pdb_atom("HETATM", 9001, "O1", "LIG", "L", 999, 2.5, 1.30, 0.0, "O"),
                _pdb_atom("HETATM", 9002, "C2", "LIG", "L", 999, 3.9, -0.4, 0.0, "C"),
                _pdb_atom("HETATM", 9003, "C3", "LIG", "L", 999, 5.1, 0.2, 0.0, "C"),
                "END",
            ]) + "\n", encoding="utf-8")
            gate_res = evaluate_redocking_control(cocrystal_out, docked_good, rmsd_threshold_A=2.0)
            self.assertTrue(gate_res["evaluated"])
            self.assertTrue(gate_res["passed_redocking_gate"])
            self.assertLess(gate_res["symmetry_corrected_rmsd_A"], 0.25)

    def test_dual_state_water_bridge_vs_displaced_water_scoring(self):
        from src.common import atom_dict
        receptor_atoms = [
            atom_dict(_pdb_atom("ATOM", 1, "ND2", "ASN", "A", 169, 0.0, 2.85, 0.0, "N")),
            # Water 1 at (0, 0, 0): forms tetrahedral bridge between ligand O at (2.8, 0, 0) and ASN ND2 at (0, 2.85, 0) (angle = 90 deg)
            atom_dict(_pdb_atom("HETATM", 2, "O", "HOH", "A", 801, 0.0, 0.0, 0.0, "O")),
            # Water 2 at (6.0, 0.0, 0.0): overlapped/displaced by ligand C2 at (5.9, 0.1, 0.0) (d = 0.14 A < 2.2 A)
            atom_dict(_pdb_atom("HETATM", 3, "O", "HOH", "A", 802, 6.0, 0.0, 0.0, "O")),
        ]
        lig_model = [
            atom_dict(_pdb_atom("HETATM", 10, "OA", "LIG", "L", 999, 2.80, 0.0, 0.0, "O")),
            atom_dict(_pdb_atom("HETATM", 11, "C1", "LIG", "L", 999, 4.20, 0.0, 0.0, "C")),
            atom_dict(_pdb_atom("HETATM", 12, "C2", "LIG", "L", 999, 5.90, 0.1, 0.0, "C")),
        ]
        rescore = compute_physicochemical_rescore(lig_model, receptor_atoms)
        self.assertEqual(rescore["Bridged_Waters_Count"], 1)
        self.assertEqual(rescore["Displaced_Waters_Count"], 1)
        self.assertLess(rescore["Water_Bridge_kcal"], -0.4)
        self.assertAlmostEqual(rescore["Water_Displacement_kcal"], -0.45, places=2)
        # Ensure displaced water did NOT trigger a positive steric clash penalty
        self.assertLess(rescore["PhysChem_Score_kcal_mol"], -0.5)


    def test_pdbqt_spanning_tree_compiler_emits_nested_branch_and_torsdof(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parsed = parse_smiles_tokens("CCCCCC=O")
            coords = generate_fallback_3d_coords(parsed, seed=42)
            out_pdbqt = write_fallback_pdbqt("Hexanal", coords, root / "hexanal.pdbqt", bonds=parsed["bonds"])
            content = out_pdbqt.read_text(encoding="utf-8")
            self.assertIn("BRANCH", content)
            self.assertIn("ENDBRANCH", content)
            self.assertIn("TORSDOF 4", content)
            self.assertEqual(content.count("\nBRANCH "), 4)
            self.assertEqual(content.count("\nENDBRANCH "), 4)

    def test_interaction_analyzer_detects_all_polar_hbonds_salt_bridges_pi_halogen_and_nonpolar_safe(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            complex_pdb = root / "complex.pdb"
            complex_pdb.write_text("\n".join([
                _pdb_atom("ATOM", 1, "SG", "CYS", "A", 302, 0.0, 0.0, 0.0, "S"),
                _pdb_atom("ATOM", 2, "ND2", "ASN", "A", 169, 2.8, 0.0, 0.0, "N"),
                _pdb_atom("ATOM", 3, "NZ", "LYS", "A", 192, 0.0, 5.8, 0.0, "N"),
                _pdb_atom("ATOM", 4, "CG", "PHE", "A", 401, 0.0, 0.0, 3.7, "C"),
                # Full aromatic ring for pi-stacking (centroid 10,10,13.7)
                _pdb_atom("ATOM", 20, "CG", "PHE", "A", 402, 11.4, 10.0, 13.7, "C"),
                _pdb_atom("ATOM", 21, "CD1", "PHE", "A", 402, 10.7, 11.212, 13.7, "C"),
                _pdb_atom("ATOM", 22, "CD2", "PHE", "A", 402, 9.3, 11.212, 13.7, "C"),
                _pdb_atom("ATOM", 23, "CE1", "PHE", "A", 402, 8.6, 10.0, 13.7, "C"),
                _pdb_atom("ATOM", 24, "CE2", "PHE", "A", 402, 9.3, 8.788, 13.7, "C"),
                _pdb_atom("ATOM", 25, "CZ", "PHE", "A", 402, 10.7, 8.788, 13.7, "C"),
                _pdb_atom("HETATM", 5, "ZN", "ZN", "A", 601, -2.1, 0.0, 0.0, "ZN"),
                # Ligand with TWO polar oxygens O1 and O2, aromatic/aliphatic C1, and Cl1 halogen
                _pdb_atom("HETATM", 10, "O1", "LIG", "L", 999, 0.0, 0.0, 0.0, "O"),
                _pdb_atom("HETATM", 11, "C1", "LIG", "L", 999, 1.2, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 12, "O2", "LIG", "L", 999, 0.0, 3.0, 0.0, "O"),
                _pdb_atom("HETATM", 13, "CL1", "LIG", "L", 999, 2.8, 3.1, 0.0, "CL"),
                # Planar six-membered ligand ring (centroid 10,10,10), parallel to PHE402
                _pdb_atom("HETATM", 30, "C10", "LIG", "L", 999, 11.4, 10.0, 10.0, "C"),
                _pdb_atom("HETATM", 31, "C11", "LIG", "L", 999, 10.7, 11.212, 10.0, "C"),
                _pdb_atom("HETATM", 32, "C12", "LIG", "L", 999, 9.3, 11.212, 10.0, "C"),
                _pdb_atom("HETATM", 33, "C13", "LIG", "L", 999, 8.6, 10.0, 10.0, "C"),
                _pdb_atom("HETATM", 34, "C14", "LIG", "L", 999, 9.3, 8.788, 10.0, "C"),
                _pdb_atom("HETATM", 35, "C15", "LIG", "L", 999, 10.7, 8.788, 10.0, "C"),
                "END",
            ]) + "\n", encoding="utf-8")
            df = analyze_interactions(complex_pdb, root / "inter.csv", 302, target_chain="A")
            types = set(df["Type"].tolist())
            self.assertIn("Hydrogen_Bond", types)
            self.assertIn("Metal_Coordination", types)
            self.assertIn("Pi_Stacking", types)
            self.assertIn("Halogen_Bond", types)
            # Both O1 (near ASN169) and O2 (near LYS192) must be detected as H-bond participants
            hb_atoms = set(df.loc[df["Type"] == "Hydrogen_Bond", "Ligand_Atom"].tolist())
            self.assertIn("O1", hb_atoms)
            self.assertIn("O2", hb_atoms)

            # Purely nonpolar hydrocarbon ligand must not crash with ValueError
            nonpolar_pdb = root / "nonpolar.pdb"
            nonpolar_pdb.write_text("\n".join([
                _pdb_atom("ATOM", 1, "CB", "LEU", "A", 120, 0.0, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 10, "C1", "LIG", "L", 999, 3.6, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 11, "C2", "LIG", "L", 999, 4.9, 0.0, 0.0, "C"),
                "END",
            ]) + "\n", encoding="utf-8")
            df_np = analyze_interactions(nonpolar_pdb, root / "np.csv", 302, target_chain="A")
            self.assertIn("Hydrophobic_Contact", set(df_np["Type"].tolist()))

    def test_qmmm_input_adds_hydrogen_link_atom_caps_and_validates_spin_parity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            comp = root / "comp.pdb"
            comp.write_text("\n".join([
                _pdb_atom("ATOM", 1, "N", "CYS", "A", 302, -1.2, 1.0, 0.0, "N"),
                _pdb_atom("ATOM", 2, "CA", "CYS", "A", 302, 0.0, 0.0, 0.0, "C"),
                _pdb_atom("ATOM", 3, "CB", "CYS", "A", 302, 1.53, 0.0, 0.0, "C"),
                _pdb_atom("ATOM", 4, "SG", "CYS", "A", 302, 2.50, 1.50, 0.0, "S"),
                _pdb_atom("HETATM", 10, "C1", "LIG", "L", 999, 4.5, 1.5, 0.0, "C"),
                _pdb_atom("HETATM", 11, "O1", "LIG", "L", 999, 5.7, 1.5, 0.0, "O"),
                "END",
            ]) + "\n", encoding="utf-8")
            inp = write_qmmm_input(comp, root / "qmmm", {"charge": 0, "multiplicity": 1, "qm_residues": ["CYS302"]})
            manifest = json.loads((root / "qmmm" / "comp_qmmm_manifest.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(manifest["h_link_cap_count"], 1)
            self.assertTrue(manifest["spin_parity_validated"])
            self.assertIn("H   ", inp.read_text(encoding="utf-8"))

    def test_evaluation_computes_bedroc_deltag_rmse_and_multi_scorer_benchmark(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sum_csv = root / "summary.csv"
            exp_csv = root / "exp.csv"
            sum_csv.write_text(
                "Ligand,Binding_Affinity_kcal_mol,PhysChem_Score_kcal_mol,Catalytic_Geometry_Score,Universal_Consensus_Rank,Catalytically_Competent\n"
                "L1,-8.5,-4.2,9.1,1.0,True\n"
                "L2,-7.8,-3.5,8.0,2.0,True\n"
                "L3,-5.2,-1.0,2.1,3.0,False\n"
                "L4,-4.8,-0.5,1.2,4.0,False\n",
                encoding="utf-8",
            )
            exp_csv.write_text(
                "Ligand,Experimental_Value\nL1,1\nL2,1\nL3,0\nL4,0\n",
                encoding="utf-8",
            )
            metrics = evaluate_experimental_data(sum_csv, exp_csv, root / "eval_out.csv")
            self.assertEqual(metrics["bedroc_alpha20"], 1.0)
            self.assertIn("catalytic_geometry", metrics["by_scorer"])
            self.assertEqual(metrics["by_scorer"]["catalytic_geometry"]["roc_auc"], 1.0)

            # Test continuous Ki/IC50 (in uM) -> DeltaG RMSE/MAE calculation
            exp_ki = root / "exp_ki.csv"
            exp_ki.write_text(
                "Ligand,Experimental_Value,Unit\nL1,0.8,uM\nL2,2.5,uM\nL3,150.0,uM\nL4,300.0,uM\n",
                encoding="utf-8",
            )
            m_ki = evaluate_experimental_data(sum_csv, exp_ki, root / "eval_ki.csv")
            self.assertIn("dg_rmse_kcal_mol", m_ki)
            self.assertIn("dg_mae_kcal_mol", m_ki)

    def test_refined_pose_propagation_records_refinement_delta_e_and_rmsd(self):
        import pandas as pd
        df = pd.read_csv(Path(__file__).resolve().parents[1] / "reports" / "docking_summary.csv")
        self.assertIn("Refinement_Delta_Score", df.columns)
        self.assertIn("Refinement_RMSD_A", df.columns)
        self.assertIn("Run_ID", df.columns)
        self.assertIn("Seed", df.columns)


    def test_mol2_single_and_multimol_parsing_preserves_3d_coords_and_charges(self):
        sample_mol2 = """@<TRIPOS>MOLECULE
Hexanal_Mol2
 7 6 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 C.3     1  LIG1       -0.0600
      2 C2          1.5200    0.0000    0.0000 C.3     1  LIG1       -0.0500
      3 C3          2.2800    1.3100    0.0000 C.3     1  LIG1       -0.0400
      4 C4          3.8000    1.3100    0.0000 C.3     1  LIG1       -0.0300
      5 C5          4.5600    2.6200    0.0000 C.3     1  LIG1        0.0400
      6 C6          6.0600    2.6200    0.0000 C.2     1  LIG1        0.2800
      7 O1          6.6800    3.6800    0.0000 O.2     1  LIG1       -0.3800
@<TRIPOS>BOND
     1     1     2    1
     2     2     3    1
     3     3     4    1
     4     4     5    1
     5     5     6    1
     6     6     7    2
"""
        parsed = parse_mol2_block(sample_mol2)
        self.assertEqual(parsed["name"], "Hexanal_Mol2")
        self.assertEqual(parsed["heavy_atom_count"], 7)
        self.assertEqual(parsed["rotatable_bond_count"], 4)
        self.assertTrue(parsed["has_mol2_charges"])
        self.assertAlmostEqual(float(parsed["atoms"][6]["charge"]), -0.38, places=2)

    def test_prepare_ligand_from_mol2_compiles_flexible_branch_pdbqt(self):
        sample_mol2 = """@<TRIPOS>MOLECULE
Hexanal_Mol2
 7 6 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 C.3     1  LIG1        0.0000
      2 C2          1.5200    0.0000    0.0000 C.3     1  LIG1        0.0000
      3 C3          2.2800    1.3100    0.0000 C.3     1  LIG1        0.0000
      4 C4          3.8000    1.3100    0.0000 C.3     1  LIG1        0.0000
      5 C5          4.5600    2.6200    0.0000 C.3     1  LIG1        0.0000
      6 C6          6.0600    2.6200    0.0000 C.2     1  LIG1        0.0000
      7 O1          6.6800    3.6800    0.0000 O.2     1  LIG1        0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     2     3    1
     3     3     4    1
     4     4     5    1
     5     5     6    1
     6     6     7    2
"""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            mol2_file = root / "Hexanal_Mol2.mol2"
            mol2_file.write_text(sample_mol2, encoding="utf-8")
            pdbqt = prepare_ligand("Hexanal_Mol2", None, root / "out", file_path=mol2_file)
            txt = pdbqt.read_text(encoding="utf-8")
            self.assertIn("TORSDOF 4", txt)
            self.assertEqual(txt.count("\nBRANCH "), 4)
            self.assertIn("6.680   3.680   0.000", txt)

    def test_discover_raw_ligands_auto_ingests_mol2_files_and_csv_mol2_paths(self):
        multimol_text = """@<TRIPOS>MOLECULE
MolA
 2 1 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 C.2     1  LIG         0.0000
      2 O1          1.2200    0.0000    0.0000 O.2     1  LIG         0.0000
@<TRIPOS>BOND
     1     1     2    2
@<TRIPOS>MOLECULE
MolB
 2 1 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 C.3     1  LIG         0.0000
      2 O1          1.4200    0.0000    0.0000 O.3     1  LIG         0.0000
@<TRIPOS>BOND
     1     1     2    1
"""
        with TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            (raw_dir / "library.mol2").write_text(multimol_text, encoding="utf-8")
            found = discover_raw_ligands(raw_dir, [])
            names = [item["name"] for item in found]
            self.assertIn("MolA", names)
            self.assertIn("MolB", names)

    def test_assess_mol2_ligand_audits_compatibility_and_chemical_state(self):
        mol2_with_salt = """@<TRIPOS>MOLECULE
Acetate_Salt
 4 2 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 C.2     1  LIG         0.4000
      2 O1          1.2500    0.0000    0.0000 O.co2   1  LIG        -0.7000
      3 O2         -0.6500    1.1000    0.0000 O.co2   1  LIG        -0.7000
      4 NA1         5.0000    5.0000    5.0000 Na      2  NA          1.0000
@<TRIPOS>BOND
     1     1     2   ar
     2     1     3   ar
"""
        compat, chem = assess_mol2_ligand("Acetate_Salt", mol2_block=mol2_with_salt, ph=7.4)
        self.assertEqual(compat["input_format"], "mol2")
        self.assertEqual(compat["heavy_atom_count"], 3)
        self.assertTrue(chem["was_desalted"])
        self.assertIn("NA", chem["stripped_counterions"])
        self.assertIn("carboxylic_acid_or_carboxylate", chem["acidic_site_types"])

    def test_multi_receptor_discovery_and_quick_start_mol2_support(self):
        from main import discover_raw_receptors
        from scripts.quick_start import _parse_ligands
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rec_dir = root / "data" / "01_raw_receptors"
            rec_dir.mkdir(parents=True)
            r1 = rec_dir / "rec1.pdb"
            r2 = rec_dir / "rec2.pdb"
            r1.write_text(_pdb_atom("ATOM", 1, "CA", "CYS", "A", 302, 0.0, 0.0, 0.0, "C") + "\nEND\n", encoding="utf-8")
            r2.write_text(_pdb_atom("ATOM", 1, "CA", "CYS", "A", 302, 1.0, 0.0, 0.0, "C") + "\nEND\n", encoding="utf-8")
            cfg = {"receptor": {"input_pdb": str(r1), "auto_discover_receptors": True}}
            receptors = discover_raw_receptors(root, cfg)
            self.assertEqual(len(receptors), 2)
            ligs = _parse_ligands(["MyMol=test_ligand.mol2", "other.mol2"])
            self.assertEqual(ligs[0]["file"], "test_ligand.mol2")
            self.assertEqual(ligs[1]["name"], "other")


    def test_sdf_and_pdb_ligand_3d_parsing_and_branch_compilation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parsed_smi = parse_smiles_tokens("CCCCCC=O")
            coords = generate_fallback_3d_coords(parsed_smi, seed=11)
            from src.common import write_fallback_sdf
            sdf_file = write_fallback_sdf("Hexanal_SDF", coords, parsed_smi["bonds"], root / "Hexanal_SDF.sdf")
            sdf_entries = parse_sdf_file(sdf_file)
            self.assertEqual(len(sdf_entries), 1)
            self.assertEqual(sdf_entries[0]["heavy_atom_count"], 7)
            pdbqt_from_sdf = prepare_ligand("Hexanal_SDF", None, root / "out_sdf", file_path=sdf_file)
            self.assertIn("TORSDOF 4", pdbqt_from_sdf.read_text(encoding="utf-8"))

            lig_pdb = root / "lig.pdb"
            lig_pdb.write_text("\n".join([
                _pdb_atom("HETATM", 1, "C1", "LIG", "L", 1, 0.0, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 2, "C2", "LIG", "L", 1, 1.52, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 3, "C3", "LIG", "L", 1, 2.28, 1.31, 0.0, "C"),
                _pdb_atom("HETATM", 4, "O1", "LIG", "L", 1, 3.50, 1.31, 0.0, "O"),
                "END",
            ]) + "\n", encoding="utf-8")
            pdbqt_from_pdb = prepare_ligand("Propanal_PDB", None, root / "out_pdb", file_path=lig_pdb)
            self.assertIn("BRANCH", pdbqt_from_pdb.read_text(encoding="utf-8"))

    def test_cif_mmcif_receptor_auto_conversion_and_parsing(self):
        cif_text = """data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . CYS A 302 1.000 2.000 3.000 1.00 15.00 1
ATOM 2 C CA . CYS A 302 2.450 2.000 3.000 1.00 15.00 1
ATOM 3 C C . CYS A 302 3.100 3.300 3.000 1.00 15.00 1
ATOM 4 O O . CYS A 302 2.500 4.350 3.000 1.00 15.00 1
ATOM 5 C CB . CYS A 302 3.000 1.100 1.900 1.00 15.00 1
ATOM 6 S SG . CYS A 302 4.750 1.100 1.900 1.00 15.00 1
"""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cif_path = root / "receptor.cif"
            cif_path.write_text(cif_text, encoding="utf-8")
            pdb_out = convert_cif_to_pdb(cif_path, root / "converted.pdb")
            self.assertTrue(pdb_out.is_file())
            struct = PDBParser(QUIET=True).get_structure("cif_test", str(pdb_out))
            self.assertIn("A", struct[0])
            self.assertIn((" ", 302, " "), struct[0]["A"])

    def test_admet_lipinski_veber_and_qed_druglikeness_calculator(self):
        admet = compute_admet_druglikeness("Hexanal", smiles="CCCCCC=O")
        self.assertTrue(admet["Lipinski_Rule_of_5_Pass"])
        self.assertTrue(admet["Veber_Oral_Pass"])
        self.assertEqual(admet["HBA_Count"], 1)
        self.assertEqual(admet["HBD_Count"], 0)
        self.assertGreater(admet["TPSA_A2"], 15.0)
        self.assertGreater(admet["QED_DrugLikeness_Score"], 0.4)

    def test_visual_html_report_generates_chinese_scorecard_and_3d_viewer_json(self):
        project_root = Path(__file__).resolve().parents[1]
        html_path = generate_visual_html_report(project_root)
        self.assertTrue(html_path.is_file())
        html_text = html_path.read_text(encoding="utf-8")
        self.assertIn("const POSES =", html_text)
        self.assertIn("Lipinski", html_text)
        self.assertIn("Hexanal", html_text)

    def test_pipeline_generates_admet_csv_and_receptor_column_in_docking_summary(self):
        import pandas as pd
        project_root = Path(__file__).resolve().parents[1]
        admet_csv = project_root / "reports" / "admet_druglikeness.csv"
        summary_csv = project_root / "reports" / "docking_summary.csv"
        html_report = project_root / "reports" / "对接结果可视化总览报告.html"
        self.assertTrue(admet_csv.is_file())
        self.assertTrue(html_report.is_file())
        df_sum = pd.read_csv(summary_csv)
        self.assertIn("Receptor", df_sum.columns)


    def test_pdbqt_atom_type_column_maps_aromatic_carbon_and_halogens(self):
        pdbqt_line = "ATOM      1  C   UNL     1       1.000   2.000   3.000  1.00  0.00     0.115 A "
        atom = atom_dict(pdbqt_line)
        self.assertEqual(atom["element"], "C")
        self.assertTrue(atom["aromatic"])
        self.assertEqual(atom["type"], "A")

        pdb_line = _pdb_atom("ATOM", 1, "CL1", "LIG", "L", 999, 0.0, 0.0, 0.0, "CL")
        self.assertEqual(atom_dict(pdb_line.rstrip("\n"))["element"], "CL")

    def test_clean_ligand_pdbqt_writes_aromatic_carbon_as_element_c(self):
        model = [{
            "name": "C1",
            "element": "C",
            "type": "A",
            "aromatic": True,
            "serial": 1,
            "coord": [0.0, 0.0, 0.0],
            "resname": "LIG",
            "chain": "L",
            "resseq": 999,
        }]
        line = clean_ligand_pdbqt(model)[0]
        self.assertEqual(line[76:78].strip(), "C")

    def test_replicate_seed_resolution_never_recycles_seeds(self):
        seeds = resolve_replicate_seeds(5, [101, 202])
        self.assertEqual(len(seeds), 5)
        self.assertEqual(len(set(seeds)), 5)
        self.assertEqual(seeds[:2], [101, 202])
        self.assertEqual(resolve_replicate_seeds(1, []), [None])

    def test_spearman_uses_average_ranks_for_ties(self):
        self.assertAlmostEqual(_spearman([1, 1, 2, 2], [1, 2, 3, 4]), 0.894427, places=6)

    def test_box_audit_does_not_require_catalytic_cys_for_general_binding(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdb = root / "receptor.pdb"
            pdb.write_text("\n".join([
                _pdb_atom("ATOM", 1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0, "C"),
                _pdb_atom("ATOM", 2, "CA", "ALA", "A", 2, 5.0, 0.0, 0.0, "C"),
                "END",
            ]) + "\n", encoding="utf-8")
            config = {
                "receptor": {"chain_id": "A", "catalytic_residue": {}, "keep_cofactors": []},
                "enzyme": {"mechanism_family": "general_binding"},
            }
            box = {"center_x": 0.0, "center_y": 0.0, "center_z": 0.0, "size_x": 20.0, "size_y": 20.0, "size_z": 20.0}
            result = audit_docking_box(pdb, box, config, root / "audit.json")
            self.assertNotIn("catalytic_cys_sg_not_found_for_box_audit", result["flags"])

    def test_interaction_cutoffs_and_aromaticity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            complex_pdb = root / "complex.pdb"
            complex_pdb.write_text("\n".join([
                _pdb_atom("ATOM", 1, "SG", "CYS", "A", 302, 20.0, 0.0, 0.0, "S"),
                _pdb_atom("ATOM", 2, "C4N", "NAD", "A", 5501, 25.0, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 3, "O1", "LIG", "L", 999, 0.0, 0.0, 0.0, "O"),
                _pdb_atom("HETATM", 4, "C1", "LIG", "L", 999, 1.2, 0.0, 0.0, "C"),
                _pdb_atom("HETATM", 5, "C2", "LIG", "L", 999, 2.7, 0.0, 0.0, "C"),
                "END",
            ]) + "\n", encoding="utf-8")
            frame = analyze_interactions(complex_pdb, root / "inter.csv", 302, target_chain="A")
            types = set(frame["Type"].tolist())
            self.assertNotIn("Catalytic_Attack", types)
            self.assertNotIn("Hydride_Transfer", types)
            self.assertNotIn("Pi_Stacking", types)


if __name__ == "__main__":
    unittest.main()
