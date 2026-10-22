# ALDH_Docking_Pipeline Universal Technical Dossier (`v4.0`)

## 1. Executive Summary (`v4.0` Deep Algorithmic & Engineering Completion)
`ALDH_Docking_Pipeline` (`v4.0`) is a self-contained, zero-human-intervention, universal molecular docking and mechanism-aware rescoring pipeline. In `v4.0`, all 6 deep algorithmic and engineering bottlenecks identified during the `v3.0` audit have been resolved and verified by **37 automated unit tests**:

1. **Flexible Sidechain (`--flex`) Output Separation & Nested $\chi_1 + \chi_2$ Torsion Trees (`src/step5_mechanism_scorer.py`, `src/step2_receptor_prep.py`)**:
   - `parse_models_and_flex()` and `merge_receptor_and_flex()` properly separate Vina's `BEGIN_RES ... END_RES` flexible sidechain output blocks from ligand `LIG` atoms, preventing sidechain coordinates from polluting ligand `HETATM` records and merging adapted sidechain conformations back into the receptor coordinates of `active_pose.pdb`.
   - `prepare_flexible_receptor_pdbqt()` builds nested $\chi_1$ (`CA-CB`) + $\chi_2$ (`CB-CG`) `BRANCH` torsion trees so bulky pocket sidechains (e.g., Tyr, Phe, Trp, Glu, Gln, Arg, Lys, Met, His) can undergo genuine two-dihedral induced-fit rotation.

2. **Full-Protein Vectorized 3D Grid Cavity Discovery (`src/step3_box_locator.py`)**:
   - Removed the `[:20]` grid truncation bug in `discover_binding_pockets()`.
   - Replaced the triple nested Python loop with a vectorized NumPy 3D distance-field grid scanner across the entire protein bounding box ($>100\text{ \AA}$ span supported) with an **8-octant burial enclosure check** ($\ge 5/8$ octants enclosed by protein heavy atoms within $14\text{ \AA}$) so surface grooves are distinguished from true internal binding pockets.

3. **Coupled Rigid-Body + Internal Rotatable Torsion L-BFGS-B Pocket Refinement (`src/step8_dynamics.py`)**:
   - `refine_active_pose_in_pocket()` now detects up to 4 internal non-ring rotatable single bonds (`_detect_ligand_rotatable_bonds()`) and simultaneously optimizes **6 rigid-body degrees of freedom + internal torsion angles** via Rodrigues' rotation formula (`_rodrigues_rotate()`), regularized by an intramolecular 1,4+ non-bonded steric clash penalty.

4. **Topological Warhead Identification & Tetrahedral $sp^2 \to sp^3$ Covalent Rehybridization (`src/system_upgrade.py`)**:
   - `_identify_electrophilic_warhead_carbon()` topologically locates the actual electrophilic carbon ($C=O$ aldehyde/ketone/Michael acceptor, $C\equiv N$ nitrile, strained epoxide/aziridine, or C–halogen/leaving group) rather than picking the nearest geometric atom.
   - `build_covalent_complex_and_manifest()` performs stereochemical $sp^2 \to sp^3$ rehybridization on carbonyl adducts, lengthening $\text{C=O} \to \text{C-O}$ ($1.41\text{ \AA}$) and deflecting the oxyanion to a tetrahedral angle ($\approx 109.5^\circ$) toward the oxyanion hole.

5. **Iterative Gasteiger-Marsili PEOE Partial Charges, PDBQT Charge Extraction & Coupled Multi-Site Microstates (`src/common.py`, `src/step5_mechanism_scorer.py`)**:
   - `atom_dict()` preserves real partial charges parsed from PDBQT columns 67–76 (`pdbqt_charge`).
   - `compute_peoe_charges()` implements 6-iteration Gasteiger-Marsili Partial Equalization of Orbital Electronegativity (PEOE) over the molecular bond graph when PDBQT charges are absent or uniform, feeding directly into `compute_physicochemical_rescore()`.
   - `enumerate_ph_protonation_states()` evaluates coupled multi-site Henderson-Hasselbalch microstates (up to 4 ionizable sites, $2^k$ states), accurately generating zwitterionic species (e.g., amino acids, $\beta$-lactams, peptidomimetics) at physiological pH.

6. **Hungarian Symmetry-Corrected 3D Heavy-Atom RMSD & Multi-Worker Parallel Screening (`src/step7_advanced.py`, `main.py`)**:
   - `compute_pose_rmsd()` builds an element + topological degree signature for each heavy atom and solves the minimum-distance assignment via `scipy.optimize.linear_sum_assignment` (Hungarian algorithm), eliminating false RMSD inflation from $180^\circ$ flips of symmetric carboxylate, phenyl, sulfonyl, or isopropyl groups.
   - `main.py` supports `--workers N` via `concurrent.futures.ThreadPoolExecutor` for thread-safe parallel ligand/replicate docking and refinement.

## 2. Verification & Validation Summary
- **Unit Test Suite**: `python3 -m unittest discover -s tests -v` — **37 / 37 tests pass**.
- **Pipeline End-to-End Execution**: `python3 main.py --config config/pipeline_config.yaml --workers 2` — completes all 8 stages and writes SQLite run history (`reports/run_history.sqlite3`), CSV summaries, interactive HTML report, and PyMOL visualization scripts.
- **Artifact Integrity Check**: `python3 scripts/validate_outputs.py` — passes with zero missing outputs.
