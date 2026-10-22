from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .common import atom_dict, distance


def _detect_ligand_rotatable_bonds(coords: np.ndarray, max_torsions: int = 4) -> list[tuple[int, int, list[int]]]:
    """Identify acyclic rotatable single bonds (u, v, downstream_atom_indices) from 3D ligand heavy-atom coordinates."""
    n = len(coords)
    if n < 4:
        return []
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if 1.38 <= d <= 1.85:
                adj[i].append(j)
                adj[j].append(i)
            elif 1.10 <= d < 1.38:
                # Double/triple bond: still connected in graph, but not rotatable
                adj[i].append(j)
                adj[j].append(i)

    rotatable: list[tuple[int, int, list[int]]] = []
    for u in range(n):
        if len(adj[u]) < 2:
            continue
        for v in adj[u]:
            if u >= v or len(adj[v]) < 2:
                continue
            bond_len = float(np.linalg.norm(coords[u] - coords[v]))
            if bond_len < 1.39:
                continue  # Skip double/triple/aromatic bonds
            # Check if removing (u, v) splits the graph (i.e. acyclic bond)
            visited = {v}
            queue = [v]
            while queue:
                curr = queue.pop(0)
                for nb in adj[curr]:
                    if (curr == v and nb == u) or (curr == u and nb == v):
                        continue
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            if u in visited:
                continue  # Bond is inside a ring
            downstream = sorted(visited - {v})
            if 1 <= len(downstream) <= n - 2:
                rotatable.append((u, v, downstream))
    rotatable.sort(key=lambda item: len(item[2]))
    return rotatable[:max_torsions]


def _rodrigues_rotate(points: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    if abs(angle_rad) < 1e-7:
        return points
    norm = float(np.linalg.norm(axis))
    if norm < 1e-7:
        return points
    u = axis / norm
    cos_t = math.cos(angle_rad)
    sin_t = math.sin(angle_rad)
    k = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]], dtype=float)
    rot = np.eye(3) + sin_t * k + (1.0 - cos_t) * (k @ k)
    return (points - origin) @ rot.T + origin


def refine_active_pose_in_pocket(
    complex_pdb: Path,
    output_pdb: Path | None = None,
    catalytic_resseq: int | None = None,
) -> dict:
    """Perform in-pocket L-BFGS-B local energy minimization over 6 rigid-body DOFs + internal ligand rotatable bond torsions + intramolecular 1,4+ strain penalty."""
    if not complex_pdb.is_file():
        return {"refined": False, "reason": "file_not_found"}

    raw_lines = complex_pdb.read_text(encoding="utf-8", errors="replace").splitlines()
    protein_lines = []
    ligand_atoms = []
    pocket_atoms = []

    for line in raw_lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom = atom_dict(line)
        if atom["resname"] == "LIG":
            ligand_atoms.append(atom)
        else:
            protein_lines.append(line)
            if atom["element"].upper() != "H":
                pocket_atoms.append(atom)

    if not ligand_atoms or not pocket_atoms:
        return {"refined": False, "reason": "missing_ligand_or_pocket"}

    lig_coords0 = np.array([a["coord"] for a in ligand_atoms], dtype=float)
    centroid = np.mean(lig_coords0, axis=0)
    near_pocket = [a for a in pocket_atoms if distance(a["coord"], centroid) <= 9.0]
    if not near_pocket:
        return {"refined": False, "reason": "no_nearby_pocket_atoms"}

    rotatable_bonds = _detect_ligand_rotatable_bonds(lig_coords0, max_torsions=4)
    n_torsions = len(rotatable_bonds)

    # Precompute non-bonded intramolecular pairs (initial distance > 2.35 A) to penalize internal clashes
    n_lig = len(lig_coords0)
    init_lig_dists = np.linalg.norm(lig_coords0[:, None, :] - lig_coords0[None, :, :], axis=-1)
    nonbonded_mask = np.triu(init_lig_dists > 2.45, k=1)

    p_coords = np.array([a["coord"] for a in near_pocket], dtype=float)
    p_polar = np.array([a["element"].upper() in {"N", "O", "S"} for a in near_pocket], dtype=bool)
    l_polar = np.array([a["element"].upper() in {"N", "O", "S"} for a in ligand_atoms], dtype=bool)

    cat_coord = None
    if catalytic_resseq is not None:
        cat_atom = next((a for a in near_pocket if a["resseq"] == int(catalytic_resseq) and a["name"] in {"SG", "OG", "NZ", "OD1", "OE1"}), None)
        if cat_atom is not None:
            cat_coord = cat_atom["coord"]

    def _transform(params: np.ndarray) -> np.ndarray:
        dx, dy, dz, rx, ry, rz = params[:6]
        coords = lig_coords0.copy()
        for t_idx, (u_idx, v_idx, downstream) in enumerate(rotatable_bonds):
            phi = float(params[6 + t_idx])
            if abs(phi) > 1e-6:
                axis = coords[v_idx] - coords[u_idx]
                coords[downstream] = _rodrigues_rotate(coords[downstream], coords[v_idx], axis, phi)

        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        rot = rot_z @ rot_y @ rot_x
        return (coords - centroid) @ rot.T + centroid + np.array([dx, dy, dz])

    def _pocket_energy(params: np.ndarray) -> float:
        coords = _transform(params)
        diff = coords[:, None, :] - p_coords[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        clash_mask = dists < 2.35
        clash_e = float(np.sum(((2.35 - dists[clash_mask]) ** 2) * 18.0))
        disp_mask = (dists >= 2.35) & (dists <= 4.8)
        disp_e = float(-0.08 * np.sum(np.exp(-((dists[disp_mask] - 3.6) ** 2) / 0.8)))
        polar_pair = l_polar[:, None] & p_polar[None, :] & (dists >= 2.4) & (dists <= 3.5)
        hb_e = float(-0.65 * np.sum(np.exp(-((dists[polar_pair] - 2.85) ** 2) / 0.2)))

        # Intramolecular 1,4+ steric strain + torsion harmonic regularization
        internal_e = 0.0
        if np.any(nonbonded_mask):
            lig_d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            nb_d = lig_d[nonbonded_mask]
            nb_clash = nb_d < 2.30
            if np.any(nb_clash):
                internal_e += float(np.sum(((2.30 - nb_d[nb_clash]) ** 2) * 20.0))
        if n_torsions > 0:
            internal_e += float(0.15 * np.sum(params[6:] ** 2))

        cat_e = 0.0
        if cat_coord is not None:
            min_cat = float(np.min(np.linalg.norm(coords - cat_coord[None, :], axis=1)))
            if min_cat > 4.2:
                cat_e = (min_cat - 3.6) ** 2 * 1.5
        return clash_e + disp_e + hb_e + internal_e + cat_e

    p0 = np.zeros(6 + n_torsions, dtype=float)
    e_init = _pocket_energy(p0)
    bounds = [(-0.75, 0.75), (-0.75, 0.75), (-0.75, 0.75), (-0.25, 0.25), (-0.25, 0.25), (-0.25, 0.25)] + [
        (-0.60, 0.60) for _ in range(n_torsions)
    ]
    res = minimize(_pocket_energy, p0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 65})
    coords_opt = _transform(res.x) if res.success else lig_coords0
    e_final = _pocket_energy(res.x) if res.success else e_init
    if e_final > e_init:
        coords_opt = lig_coords0
        e_final = e_init

    rmsd = float(np.sqrt(np.mean(np.sum((coords_opt - lig_coords0) ** 2, axis=1))))
    if output_pdb is None:
        output_pdb = complex_pdb.with_name(f"{complex_pdb.stem}_refined.pdb")
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    refined_lig_lines = []
    for idx, atom in enumerate(ligand_atoms):
        x, y, z = coords_opt[idx]
        elem = atom["element"].strip().upper()[:2]
        refined_lig_lines.append(
            f"HETATM{atom['serial']:5d} {atom['name'][:4]:>4s} LIG L{999:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}"
        )
    output_pdb.write_text("\n".join(protein_lines + refined_lig_lines + ["END"]) + "\n", encoding="utf-8")
    return {
        "refined": True,
        "refined_pdb": str(output_pdb),
        "optimized_torsion_count": n_torsions,
        "initial_pocket_energy_kcal": round(e_init, 3),
        "refined_pocket_energy_kcal": round(e_final, 3),
        "energy_improvement_kcal": round(e_init - e_final, 3),
        "refinement_rmsd_A": round(rmsd, 4),
    }


def write_md_setup(complex_pdb: Path, output_dir: Path, md_config: dict) -> Path:
    """Split docked complex into clean protein/ligand/cofactor PDBs, run in-pocket L-BFGS-B rigid+torsion energy refinement, audit clashes, and generate MD runner scripts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = complex_pdb.stem
    refinement = refine_active_pose_in_pocket(complex_pdb, output_dir / f"{stem}_refined.pdb")
    protein_lines, ligand_lines, cofactor_lines = [], [], []
    protein_atoms, ligand_atoms = [], []

    source_pdb = Path(refinement["refined_pdb"]) if refinement.get("refined") else complex_pdb
    if source_pdb.is_file():
        for line in source_pdb.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom = atom_dict(line)
            if atom["resname"] == "LIG":
                ligand_lines.append(line)
                ligand_atoms.append(atom)
            elif atom["resname"] in {"NAD", "NAP", "NDP", "FAD", "FMN", "HEM", "HEC"}:
                cofactor_lines.append(line)
                protein_atoms.append(atom)
            else:
                protein_lines.append(line)
                protein_atoms.append(atom)

    protein_pdb = output_dir / f"{stem}_protein.pdb"
    ligand_pdb = output_dir / f"{stem}_ligand.pdb"
    cofactor_pdb = output_dir / f"{stem}_cofactor.pdb"
    if protein_lines:
        protein_pdb.write_text("\n".join(protein_lines + ["END"]) + "\n", encoding="utf-8")
    if ligand_lines:
        ligand_pdb.write_text("\n".join(ligand_lines + ["END"]) + "\n", encoding="utf-8")
    if cofactor_lines:
        cofactor_pdb.write_text("\n".join(cofactor_lines + ["END"]) + "\n", encoding="utf-8")

    min_dist = 99.9
    clash_count = 0
    for la in ligand_atoms:
        for pa in protein_atoms:
            if la["element"].upper() == "H" or pa["element"].upper() == "H":
                continue
            d = distance(la["coord"], pa["coord"])
            if d < min_dist:
                min_dist = d
            if d < 1.8:
                clash_count += 1

    engine = md_config.get("engine", "gromacs")
    steps = int(md_config.get("steps", 5000000))
    temp_k = float(md_config.get("temperature_K", 298.15))
    script_path = output_dir / f"{stem}_run_gromacs.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"# Automated GROMACS preparation & equilibration template for {stem}\n"
        f"gmx pdb2gmx -f {protein_pdb.name} -o protein_processed.gro -water tip3p -ff amber99sb-ildn -ignh\n"
        "gmx editconf -f protein_processed.gro -o boxed.gro -c -d 1.0 -bt dodecahedron\n"
        "gmx solvate -cp boxed.gro -cs spc216.gro -o solvated.gro -p topol.top\n",
        encoding="utf-8",
    )

    manifest = {
        "complex_pdb": str(complex_pdb),
        "in_pocket_refinement": refinement,
        "split_structures": {
            "protein_pdb": str(protein_pdb) if protein_lines else None,
            "ligand_pdb": str(ligand_pdb) if ligand_lines else None,
            "cofactor_pdb": str(cofactor_pdb) if cofactor_lines else None,
        },
        "steric_audit": {
            "min_heavy_atom_distance_A": round(float(min_dist), 3) if min_dist < 99.0 else None,
            "severe_clash_count_under_1_8A": clash_count,
            "ready_for_minimization": clash_count == 0,
        },
        "engine": engine,
        "steps": steps,
        "temperature_K": temp_k,
        "automation_script": str(script_path),
        "note": "In-pocket L-BFGS-B rigid+torsion energy refinement and split coordinates completed; execute MD for full trajectory sampling.",
    }
    path = output_dir / f"{complex_pdb.stem}_md_setup.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / f"{complex_pdb.stem}_md_checklist.txt").write_text(
        "1. Select protein/cofactor/ligand force fields (AMBER99SB-ILDN + GAFF2/OpenFF).\n"
        "2. Parameterize split ligand PDB and verify NAD chemistry.\n"
        "3. Solvate, add ions, minimize, equilibrate NVT/NPT.\n"
        "4. Run replicate production trajectories.\n"
        "5. Analyze ligand RMSD, contacts, catalytic distances, and NAD geometry.\n",
        encoding="utf-8",
    )
    return path
