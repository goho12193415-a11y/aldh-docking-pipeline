import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.step5_mechanism_scorer import clean_ligand_pdbqt, score_docking
from src.step3_box_locator import locate_box
from src.step1_ligand_prep import prepare_polysaccharide_conformers
from src.step7_advanced import (
    assess_enzyme_readiness,
    assess_ligand_compatibility,
    assess_ligand_chemical_state,
    audit_receptor_state,
    audit_docking_box,
    build_execution_plan,
    build_capability_report,
    aggregate_replicates,
    detect_catalytic_cys,
    enforce_capability_gate,
)
from src.mechanism_templates import load_mechanism_template, mechanism_capability
from src.system_upgrade import (
    audit_structure_repair,
    build_engine_consensus_plan,
    build_flexible_receptor_plan,
    build_multi_box_plan,
    build_workflow_readiness,
)
from src.step10_evaluation import _roc_auc
from src.step10_evaluation import _enrichment_factor, _pr_auc


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
        def pdb_atom(record, serial, name, resname, chain, resseq, x, y, z, element):
            return f"{record:<6}{serial:5d} {name:>4s} {resname:>3s} {chain}{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"

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
            pdb_atom("ATOM", 1, "SG", "CYS", "A", 302, 7.0, 0, 0, "S"),
            "END",
        ]) + "\n"
        docking = "\n".join([
            "REMARK VINA RESULT: -8.0 0.000 0.000",
            "MODEL        1",
            pdb_atom("ATOM", 1, "OA", "LIG", "L", 999, 0, 0, 0, "OA"),
            pdb_atom("ATOM", 2, "C1", "LIG", "L", 999, 1.2, 0, 0, "C"),
            "ENDMDL",
            "REMARK VINA RESULT: -6.0 0.000 0.000",
            "MODEL        2",
            pdb_atom("ATOM", 1, "OA", "LIG", "L", 999, 2.3, 0, 0, "OA"),
            pdb_atom("ATOM", 2, "C1", "LIG", "L", 999, 3.5, 0, 0, "C"),
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
        def pdb_atom(record, serial, name, resname, chain, resseq, x, y, z, element):
            return f"{record:<6}{serial:5d} {name:>4s} {resname:>3s} {chain}{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"

        with TemporaryDirectory() as directory:
            path = Path(directory) / "receptor.pdb"
            path.write_text("\n".join([
                pdb_atom("ATOM", 1, "SG", "CYS", "A", 302, 0, 0, 0, "S"),
                pdb_atom("HETATM", 2, "C4N", "NAD", "B", 5501, 10, 0, 0, "C"),
                "END",
            ]) + "\n", encoding="utf-8")
            box = locate_box(path, "A", 302, (18.0, 18.0, 18.0), 2.0, "midpoint_if_cofactor")
            self.assertEqual(box["center_x"], 5.0)
            self.assertEqual(box["center_y"], 0.0)
            self.assertEqual(box["center_z"], 0.0)

    def test_advanced_cys_detection_maps_sequence_position(self):
        def pdb_atom(record, serial, name, resname, chain, resseq, x, y, z, element):
            return f"{record:<6}{serial:5d} {name:>4s} {resname:>3s} {chain}{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"

        with TemporaryDirectory() as directory:
            path = Path(directory) / "receptor.pdb"
            lines = []
            sequence = [("PHE", 298), ("GLY", 299), ("ALA", 300), ("CYS", 301)]
            for serial, (resname, number) in enumerate(sequence, 1):
                lines.append(pdb_atom("ATOM", serial, "CA", resname, "A", number, serial, 0, 0, "C"))
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
        def pdb_atom(record, serial, name, resname, chain, resseq, x, y, z, element):
            return f"{record:<6}{serial:5d} {name:>4s} {resname:>3s} {chain}{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"

        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdb = root / "enzyme.pdb"
            pdb.write_text("\n".join([
                pdb_atom("ATOM", 1, "CA", "CYS", "A", 302, 0, 0, 0, "C"),
                pdb_atom("ATOM", 2, "SG", "CYS", "A", 302, 1, 0, 0, "S"),
                pdb_atom("HETATM", 3, "C4N", "NAD", "A", 5501, 3, 0, 0, "C"),
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
        def pdb_atom(record, serial, name, resname, chain, resseq, x, y, z, element, altloc=" "):
            return f"{record:<6}{serial:5d} {name:>4s}{altloc}{resname:>3s} {chain}{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"

        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdb = root / "state.pdb"
            pdb.write_text("\n".join([
                pdb_atom("ATOM", 1, "CA", "CYS", "A", 302, 0, 0, 0, "C", "A"),
                pdb_atom("HETATM", 2, "C4N", "NAD", "A", 5501, 3, 0, 0, "C"),
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
        def pdb_atom(record, serial, name, resname, chain, resseq, x, y, z, element):
            return f"{record:<6}{serial:5d} {name:>4s} {resname:>3s} {chain}{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"

        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdb = root / "box.pdb"
            pdb.write_text("\n".join([
                pdb_atom("ATOM", 1, "SG", "CYS", "A", 302, 0, 0, 0, "S"),
                pdb_atom("HETATM", 2, "C4N", "NAD", "A", 5501, 10, 0, 0, "C"),
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
