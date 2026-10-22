from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import PDBParser, distance

METAL_RESNAMES = {"ZN", "MG", "FE", "MN", "CA", "CU", "NI", "CO"}
COFACTOR_RESNAMES = {"NAD", "NAP", "NDP", "FAD", "FMN", "HEM", "HEC", "HEA"}
HYDROPHOBIC_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR"}


def locate_atom(pdb_path: Path, resname: str, atom_name: str, resseq: int | None, chain_id: str) -> np.ndarray:
    structure = PDBParser(QUIET=True).get_structure("structure", str(pdb_path))
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.get_resname().strip() != resname:
                    continue
                if resseq is not None and residue.id[1] != resseq:
                    continue
                if atom_name in residue:
                    return np.array(residue[atom_name].coord, dtype=float)
    raise ValueError(f"Could not locate {resname} {atom_name} in chain {chain_id}.")


def discover_binding_pockets(
    pdb_path: Path,
    chain_id: str = "A",
    top_k: int = 3,
    default_box_size: tuple[float, float, float] = (20.0, 20.0, 20.0),
) -> list[dict]:
    """Full-protein vectorized 3D grid cavity & octant-enclosure scanner without grid truncation."""
    structure = PDBParser(QUIET=True).get_structure("cavity_scan", str(pdb_path))
    model = structure[0]
    if chain_id not in model:
        chain_id = next(iter(model)).id
    chain = model[chain_id]

    protein_coords = []
    protein_res = []
    cofactor_centroids = []
    for r in chain:
        rname = r.get_resname().strip()
        if rname in {"HOH", "WAT"}:
            continue
        if r.id[0] != " ":
            pts = [np.array(a.get_coord(), dtype=float) for a in r]
            if pts:
                cofactor_centroids.append((rname, int(r.id[1]), np.mean(pts, axis=0)))
            continue
        for a in r:
            elem = (getattr(a, "element", "") or a.get_name()[:1]).strip().upper()
            if elem != "H":
                protein_coords.append(np.array(a.get_coord(), dtype=float))
                protein_res.append((rname, int(r.id[1])))

    if not protein_coords:
        return [{
            "name": "pocket_1",
            "center_x": 0.0, "center_y": 0.0, "center_z": 0.0,
            "size_x": default_box_size[0], "size_y": default_box_size[1], "size_z": default_box_size[2],
            "pocket_score": 0.0, "lining_residues": [],
        }]

    coords = np.array(protein_coords, dtype=float)
    min_xyz = np.min(coords, axis=0) + 1.5
    max_xyz = np.max(coords, axis=0) - 1.5
    span = max_xyz - min_xyz
    if np.any(span <= 0.0):
        c = np.mean(coords, axis=0)
        return [{
            "name": "pocket_1",
            "center_x": round(float(c[0]), 3), "center_y": round(float(c[1]), 3), "center_z": round(float(c[2]), 3),
            "size_x": default_box_size[0], "size_y": default_box_size[1], "size_z": default_box_size[2],
            "pocket_score": 1.0, "lining_residues": [],
        }]

    # Adaptive grid spacing covering 100% of the protein volume (never truncated at [:20])
    max_span = float(np.max(span))
    spacing = max(3.2, max_span / 28.0)
    xs = np.arange(min_xyz[0], max_xyz[0] + 0.1, spacing)
    ys = np.arange(min_xyz[1], max_xyz[1] + 0.1, spacing)
    zs = np.arange(min_xyz[2], max_xyz[2] + 0.1, spacing)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

    candidate_points: list[tuple[float, np.ndarray]] = []
    chunk_size = 512
    for start in range(0, len(grid_pts), chunk_size):
        batch = grid_pts[start:start + chunk_size]
        diff = batch[:, None, :] - coords[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        d_min = np.min(dists, axis=1)
        valid_mask = (d_min >= 2.6) & (d_min <= 5.2)
        if not np.any(valid_mask):
            continue
        for idx in np.where(valid_mask)[0]:
            pt = batch[idx]
            pt_dists = dists[idx]
            shell_mask = pt_dists <= 8.5
            shell_count = int(np.sum(shell_mask))
            if shell_count < 14:
                continue
            # Check 3D octant enclosure so flat exterior surfaces are rejected
            rel_vecs = coords[shell_mask] - pt
            octants = (rel_vecs[:, 0] > 0).astype(int) * 4 + (rel_vecs[:, 1] > 0).astype(int) * 2 + (rel_vecs[:, 2] > 0).astype(int)
            occupied_octants = len(np.unique(octants))
            if occupied_octants < 5 and not cofactor_centroids:
                continue
            cof_bonus = 30.0 if any(distance(pt, cc) <= 6.5 for _, _, cc in cofactor_centroids) else 0.0
            score = shell_count + occupied_octants * 3.5 + cof_bonus
            candidate_points.append((score, pt))

    candidate_points.sort(key=lambda item: -item[0])
    pockets = []
    for score, pt in candidate_points:
        if any(distance(pt, np.array([p["center_x"], p["center_y"], p["center_z"]])) < 8.5 for p in pockets):
            continue
        dists = np.linalg.norm(coords - pt, axis=1)
        lining = sorted({f"{protein_res[idx][0]}{protein_res[idx][1]}" for idx in np.where(dists <= 6.0)[0]})
        pockets.append({
            "name": f"pocket_{len(pockets) + 1}",
            "center_x": round(float(pt[0]), 3),
            "center_y": round(float(pt[1]), 3),
            "center_z": round(float(pt[2]), 3),
            "size_x": float(default_box_size[0]),
            "size_y": float(default_box_size[1]),
            "size_z": float(default_box_size[2]),
            "pocket_score": round(float(score), 2),
            "lining_residues": lining[:15],
        })
        if len(pockets) >= top_k:
            break

    if not pockets:
        c = cofactor_centroids[0][2] if cofactor_centroids else np.mean(coords, axis=0)
        pockets.append({
            "name": "pocket_1",
            "center_x": round(float(c[0]), 3),
            "center_y": round(float(c[1]), 3),
            "center_z": round(float(c[2]), 3),
            "size_x": float(default_box_size[0]),
            "size_y": float(default_box_size[1]),
            "size_z": float(default_box_size[2]),
            "pocket_score": 10.0,
            "lining_residues": [],
        })
    return pockets


def locate_box(
    pdb_path: Path,
    chain_id: str,
    cys_number: int,
    box_size: tuple[float, float, float],
    offset_outward: float = 2.0,
    center_mode: str = "midpoint_if_cofactor",
    resname: str = "CYS",
    atom_name: str = "SG",
) -> dict[str, float]:
    try:
        sg = locate_atom(pdb_path, resname, atom_name, cys_number, chain_id)
    except ValueError:
        structure = PDBParser(QUIET=True).get_structure("structure", str(pdb_path))
        sg = None
        for model in structure:
            for chain in model:
                if chain.id != chain_id:
                    continue
                for residue in chain:
                    if residue.id[1] == cys_number:
                        for candidate_atom in (atom_name, "SG", "OG", "NZ", "OD1", "OE1", "CA"):
                            if candidate_atom in residue:
                                sg = np.array(residue[candidate_atom].coord, dtype=float)
                                break
                    if sg is not None:
                        break
        if sg is None:
            raise ValueError(f"Could not locate {resname} {atom_name} at {cys_number} in chain {chain_id}.")

    cofactor = None
    fallback_cofactor = None
    structure = PDBParser(QUIET=True).get_structure("structure", str(pdb_path))
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip() in {"NAD", "NAP", "NDP", "FAD", "FMN", "HEM", "HEC"}:
                    for cof_atom in ("C4N", "C4", "N5", "C4X", "FE"):
                        if cof_atom in residue:
                            candidate = np.array(residue[cof_atom].coord, dtype=float)
                            if chain.id == chain_id:
                                cofactor = candidate
                            elif fallback_cofactor is None or distance(candidate, sg) < distance(fallback_cofactor, sg):
                                fallback_cofactor = candidate
                            break
                    if cofactor is not None:
                        break
            if cofactor is not None:
                break
    cofactor = cofactor if cofactor is not None else fallback_cofactor
    if cofactor is not None and center_mode == "midpoint_if_cofactor":
        center = (sg + cofactor) / 2.0
    elif cofactor is not None and center_mode == "catalytic_focused":
        direction = cofactor - sg
        norm = np.linalg.norm(direction)
        center = sg if norm == 0 else sg + direction / norm * min(offset_outward, norm * 0.35)
    elif cofactor is not None:
        direction = cofactor - sg
        norm = np.linalg.norm(direction)
        center = sg if norm == 0 else sg + direction / norm * offset_outward
    else:
        center = sg
    return {
        "center_x": round(float(center[0]), 3),
        "center_y": round(float(center[1]), 3),
        "center_z": round(float(center[2]), 3),
        "size_x": float(box_size[0]),
        "size_y": float(box_size[1]),
        "size_z": float(box_size[2]),
    }


def locate_mechanism_box(pdb_path: Path, config: dict, candidate_override: dict | None = None) -> dict[str, float]:
    """Locate docking box across all 9 enzyme/binding families, including blind cavity discovery."""
    box_cfg = dict(config.get("docking_box", {}))
    if candidate_override:
        box_cfg.update(candidate_override)
    size = (
        float(box_cfg.get("size_x", 18.0)),
        float(box_cfg.get("size_y", 18.0)),
        float(box_cfg.get("size_z", 18.0)),
    )
    if all(k in box_cfg for k in ("center_x", "center_y", "center_z")):
        return {
            "center_x": round(float(box_cfg["center_x"]), 3),
            "center_y": round(float(box_cfg["center_y"]), 3),
            "center_z": round(float(box_cfg["center_z"]), 3),
            "size_x": size[0],
            "size_y": size[1],
            "size_z": size[2],
        }
    family = str(config.get("enzyme", {}).get("mechanism_family", "aldh")).lower()
    receptor_cfg = config.get("receptor", {})
    chain_id = receptor_cfg.get("chain_id", "A")
    catalytic = receptor_cfg.get("catalytic_residue")

    structure = PDBParser(QUIET=True).get_structure("mech_box", str(pdb_path))
    if family == "metalloenzyme":
        for model in structure:
            for chain in model:
                if chain.id != chain_id:
                    continue
                for residue in chain:
                    if residue.get_resname().strip() in METAL_RESNAMES:
                        atom = next(iter(residue))
                        coord = np.array(atom.coord, dtype=float)
                        return {
                            "center_x": round(float(coord[0]), 3),
                            "center_y": round(float(coord[1]), 3),
                            "center_z": round(float(coord[2]), 3),
                            "size_x": size[0], "size_y": size[1], "size_z": size[2],
                        }
    elif family == "cytochrome_p450":
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname().strip() in {"HEM", "HEC", "HEA"} and "FE" in residue:
                        coord = np.array(residue["FE"].coord, dtype=float)
                        return {
                            "center_x": round(float(coord[0]), 3),
                            "center_y": round(float(coord[1]), 3),
                            "center_z": round(float(coord[2]), 3),
                            "size_x": size[0], "size_y": size[1], "size_z": size[2],
                        }
    elif family == "flavoenzyme":
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname().strip() in {"FAD", "FMN"}:
                        for aname in ("N5", "C4X", "C4A"):
                            if aname in residue:
                                coord = np.array(residue[aname].coord, dtype=float)
                                return {
                                    "center_x": round(float(coord[0]), 3),
                                    "center_y": round(float(coord[1]), 3),
                                    "center_z": round(float(coord[2]), 3),
                                    "size_x": size[0], "size_y": size[1], "size_z": size[2],
                                }
    elif family == "general_binding" or not catalytic:
        pockets = discover_binding_pockets(pdb_path, chain_id=chain_id, top_k=1, default_box_size=size)
        top_p = pockets[0]
        return {
            "center_x": top_p["center_x"],
            "center_y": top_p["center_y"],
            "center_z": top_p["center_z"],
            "size_x": size[0], "size_y": size[1], "size_z": size[2],
        }

    default_resname = {
        "serine_hydrolase": "SER",
        "kinase": "LYS",
        "glycosidase": "ASP",
    }.get(family, "CYS")
    resname = str(catalytic.get("name", default_resname)).upper()
    atom_name = {
        "SER": "OG", "THR": "OG1", "LYS": "NZ", "ASP": "OD1", "GLU": "OE1",
    }.get(resname, "SG")
    try:
        return locate_box(
            pdb_path,
            chain_id,
            int(catalytic.get("number", 302)),
            size,
            float(box_cfg.get("offset_outward", 2.0)),
            str(box_cfg.get("center_mode", "midpoint_if_cofactor")),
            resname=resname,
            atom_name=atom_name,
        )
    except ValueError:
        pockets = discover_binding_pockets(pdb_path, chain_id=chain_id, top_k=1, default_box_size=size)
        top_p = pockets[0]
        return {
            "center_x": top_p["center_x"],
            "center_y": top_p["center_y"],
            "center_z": top_p["center_z"],
            "size_x": size[0], "size_y": size[1], "size_z": size[2],
        }
