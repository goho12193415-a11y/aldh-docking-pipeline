# ALDH_Docking_Pipeline Universal Technical Dossier (`v9.0`)

## 1. Executive Summary (`v9.0` Live3D Release)
`ALDH_Docking_Pipeline` (`v9.0`) is a self-contained, zero-human-intervention, universal molecular docking, mechanism-aware rescoring, and post-docking refinement platform. Across `v4.0`–`v9.0`, the algorithmic, physical and workflow bottlenecks are verified by **66 automated unit tests**:

### `v9.0` New Capabilities (Live3D Release)
1. **Multi-mode trajectory export**: `score_docking()` writes `reports/active_poses/<ligand>_trajectory.json` with up to 12 Vina search modes (`vina_mode_search`) plus the L-BFGS-B refined final frame (`lbfgsb_refined_final`).
2. **Live pose stream**: `main.py` updates `reports/live_pose_stream.json` after each ligand; the GUI's right-hand 3D canvas follows the current receptor/ligand/pose/trajectory while the batch runs.
3. **HTML conformation player**: the standalone report embeds a frame-interpolating 3D viewer (`getInterpolatedLigand`) that can replay mode search and the refinement endpoint directly in the browser.

> The "trajectory" is a visual replay of docking modes and the refinement endpoint, **not** an MD time series.

### `v8.5` Capabilities Retained (Sketchbook Release)
1. **3D ligand formats**: native MOL2/SDF/MOL/PDB/PDBQT input with automatic desalting and fragment extraction (`src/step1_ligand_prep.py`).
2. **Multi-receptor discovery**: `discover_raw_receptors()` expands CLI/config/`data/01_raw_receptors/*.pdb`, runs an ensemble, and records each receptor in the execution plan (`main.py`).
3. **ADMET profiling**: Lipinski/Veber/QED descriptors written to `reports/admet_druglikeness.csv` (`src/common.py`).
4. **HTML visual folio**: a standalone sketchbook-style report with an embedded 3D pose canvas (`src/system_upgrade.py`).
5. **Scientific audit hardening**: fixed hydride-transfer cutoff (no cofactor-out auto-promotion), PDBQT atom-type parsing (`A`→aromatic C), distance-gated interaction rows, centroid/planarity π-stacking, per-run `run_id` archival, distinct replicate seeds, and explicit heuristic-refinement semantics.

### `v6.0` Capabilities Retained (Closing All Post-Processing, QM/MM, PDBQT Tree & Benchmarking Disconnects)
1. **Spanning-Tree Torsional `ROOT` + Nested `BRANCH/ENDBRANCH` Ligand PDBQT Compiler (`src/common.py`, `src/step1_ligand_prep.py`)**:
   - Replaced the rigid `TORSDOF 0` fallback in `write_fallback_pdbqt()` with a graph cycle/bridge analyzer that identifies all acyclic rotatable single bonds, partitions the molecule into rigid fragments, selects the optimal `ROOT`, and recursively emits nested `BRANCH u v ... ENDBRANCH u v` blocks with exact `TORSDOF N` counts.

2. **Full-Ligand 3D Non-Covalent Interaction Profiler (`src/step6_interaction_viz.py`)**:
   - Eliminated the `next()` single-polar-atom bug and the nonpolar ligand `ValueError` crash in `analyze_interactions()`.
   - Evaluates **all** ligand polar, charged, aromatic, halogen, and aliphatic atoms to detect `Catalytic_Attack` (across Cys/Ser/Thr/Lys/Asp/Glu/His), `Hydride_Transfer`, `Metal_Coordination`, `Hydrogen_Bond`, `Salt_Bridge`, `Pi_Stacking`, `Halogen_Bond`, and `Hydrophobic_Contact`.

3. **QM/MM Covalent Boundary Cleavage & Automated Hydrogen Link-Atom (`H_link`) Capping (`src/step9_qmmm.py`)**:
   - `write_qmmm_input()` cleanly severs protein QM residues at the `CA-CB` bond (leaving the peptide backbone `N, CA, C, O` in MM) and cofactors at the diphosphate bridge, automatically placing **Hydrogen Link Atoms (`H`)** at $1.090\text{ \AA}$ along each severed `QM -> MM` covalent bond vector to eliminate open-shell dangling radicals.
   - Automatically validates total electron count vs. spin multiplicity parity (`spin_parity_validated: True`) for ORCA B3LYP-D3BJ/def2-SVP calculations.

4. **End-to-End Downstream Propagation of L-BFGS-B Refined Poses & Automatic Ring Ensemble Routing (`main.py`)**:
   - In-pocket L-BFGS-B rigid+torsion refinement (`refine_active_pose_in_pocket`) produces a diagnostic `*_refined.pdb` and records `Refinement_Delta_Score` (arbitrary units, **not** kcal/mol) and `Refinement_RMSD_A` in `docking_summary.csv`. The refined coordinates replace the docked pose for downstream interaction profiling, figures, and MD/QM preparation only when `advanced.pose_refinement.propagate_to_downstream: true`.
   - Automatically routes flexible-ring ligands through `prepare_ring_conformer_ensemble()` and `run_vina_ensemble()` when `ring_ensemble` is enabled.

5. **Multi-Scorer Experimental Benchmarking, $\Delta G_{\text{exp}}$ RMSE/MAE & Early Enrichment `BEDROC` ($\alpha=20$) (`src/step10_evaluation.py`)**:
   - Computes Truchon & Bayly (2007) **BEDROC ($\alpha=20.0$)** early-recognition enrichment alongside ROC-AUC, PR-AUC, and EF25%.
   - Converts experimental $K_i / K_d / \text{IC}_{50}$ ($\text{nM}, \mu\text{M}, \text{mM}, \text{pIC}_{50}$) into experimental binding free energy $\Delta G_{\text{exp}} = RT \ln K_d$ ($298.15\text{ K}$) to report `dg_rmse_kcal_mol` and `dg_mae_kcal_mol`.
   - Benchmarks all 4 scoring functions (`vina_affinity`, `physchem_rescore`, `catalytic_geometry`, `universal_consensus_rank`) side by side in `by_scorer`.

### `v4.0` & `v5.0` Core Foundations Retained
- `--flex` output separation (`parse_models_and_flex` / `merge_receptor_and_flex`) and nested $\chi_1 + \chi_2$ `BRANCH` torsion trees.
- Full-protein vectorized NumPy 3D cavity scanner with 8-octant burial enclosure check ($>100\text{ \AA}$ spans).
- Coupled 6-DOF rigid-body + internal rotatable bond torsion L-BFGS-B refinement with 1,4+ intramolecular clash regularization.
- Topological electrophilic warhead identification and $sp^2 \to sp^3$ tetrahedral rehybridization ($\text{C-O}=1.41\text{ \AA}$, $\angle\text{Nu-C-O}\approx 109.5^\circ$).
- Iterative Gasteiger-Marsili PEOE partial charges, PDBQT charge extraction, coupled multi-site zwitterionic microstates, and R/S & E/Z stereoisomer + tautomer enumeration.
- Multi-receptor Kabsch SVD 3D superposition, Boltzmann-weighted ensemble free energy, co-crystal ligand extraction & redocking RMSD control, and dual-state water bridge vs. displacement scoring.
- Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) symmetry-corrected 3D RMSD and `--workers` multi-threaded execution.

## 2. Verification & Validation Summary
- **Unit Test Suite**: `python3 -m unittest discover -s tests -v` — **66 / 66 tests pass**.
- **Pipeline End-to-End Execution**: `python3 main.py --config config/pipeline_config.yaml --workers 2` — completes all stages and writes SQLite run history, CSV summaries, and PyMOL visualization scripts.
- **Artifact Integrity Check**: `python3 scripts/validate_outputs.py` — passes with zero missing outputs.
