from __future__ import annotations

import json
import logging
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .common import atom_dict, distance, require_executable

LOGGER = logging.getLogger(__name__)
HYDROPHOBIC_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR"}
INTERACTION_COLUMNS = [
    "Type", "Ligand_Atom", "Receptor_Res", "Receptor_Atom", "Distance_A"
]


def parse_complex(path: Path) -> tuple[list[dict], list[dict]]:
    receptor, ligand = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom = atom_dict(line)
        (ligand if atom["resname"] == "LIG" else receptor).append(atom)
    return receptor, ligand


AROMATIC_RES = {"PHE", "TYR", "TRP", "HIS"}
CHARGED_REC_ATOMS = {
    ("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"),
    ("LYS", "NZ"), ("ARG", "NH1"), ("ARG", "NH2"), ("ARG", "NE"),
    ("HIS", "ND1"), ("HIS", "NE2"),
}
METAL_RES = {"ZN", "MG", "FE", "MN", "CA", "CU", "NI", "CO"}
COFACTOR_RES = {"NAD", "NAP", "NDP", "FAD", "FMN", "HEM", "HEC", "HEA"}

# Atom names that define the aromatic ring of each residue (PDB naming).
RECEPTOR_RING_ATOMS = {
    "PHE": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
    "TYR": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
    "TRP": [
        ["CG", "CD1", "NE1", "CE2", "CD2"],
        ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    ],
    "HIS": [["CG", "ND1", "CD2", "CE1", "NE2"]],
}


def _plane_geometry(points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ring centroid, unit normal and maximum out-of-plane deviation."""
    coords = np.asarray(points, dtype=float)
    center = coords.mean(axis=0)
    centered = coords - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    deviation = float(np.max(np.abs(centered @ normal)))
    return center, normal, deviation


def _ligand_aromatic_rings(ligand: list[dict]) -> list[dict]:
    """Infer planar 5/6-membered ligand rings from a distance-based bond graph."""
    heavy = [atom for atom in ligand if atom["element"].upper() != "H"]
    adjacency = {index: set() for index in range(len(heavy))}
    for i in range(len(heavy)):
        for j in range(i + 1, len(heavy)):
            bond = distance(heavy[i]["coord"], heavy[j]["coord"])
            if 1.10 <= bond <= 1.90:
                adjacency[i].add(j)
                adjacency[j].add(i)

    cycles: set[tuple[int, ...]] = set()

    def _walk(start: int, current: int, path: list[int], visited: set[int]) -> None:
        for neighbour in adjacency[current]:
            if neighbour == start and len(path) >= 5:
                cycles.add(tuple(sorted(path)))
            elif neighbour > start and neighbour not in visited and len(path) < 6:
                _walk(start, neighbour, path + [neighbour], visited | {neighbour})

    for start in range(len(heavy)):
        _walk(start, start, [start], {start})

    rings: list[dict] = []
    for cycle in sorted(cycles, key=len):
        if not 5 <= len(cycle) <= 6:
            continue
        center, normal, deviation = _plane_geometry([heavy[index]["coord"] for index in cycle])
        if deviation <= 0.20:  # planar rings only; chair cyclohexane is excluded
            rings.append({
                "atoms": [heavy[index] for index in cycle],
                "center": center,
                "normal": normal,
            })
    return rings


def _receptor_aromatic_rings(receptor: list[dict], target_chain: str | None) -> list[dict]:
    by_residue: dict[tuple[str, str, int], dict[str, dict]] = {}
    for atom in receptor:
        if target_chain and atom["chain"] != target_chain:
            continue
        by_residue.setdefault((atom["chain"], atom["resname"], atom["resseq"]), {})[atom["name"]] = atom

    rings: list[dict] = []
    for (_, resname, resseq), atoms in by_residue.items():
        for names in RECEPTOR_RING_ATOMS.get(resname, []):
            if not all(name in atoms for name in names):
                continue
            center, normal, deviation = _plane_geometry([atoms[name]["coord"] for name in names])
            if deviation <= 0.25:
                rings.append({
                    "label": f"{resname}{resseq}",
                    "atoms": [atoms[name] for name in names],
                    "center": center,
                    "normal": normal,
                })
    return rings


def _pi_stack_geometry(ligand_ring: dict, receptor_ring: dict) -> tuple[float, str] | None:
    """Apply centroid distance, interplanar angle and lateral-offset criteria."""
    vector = ligand_ring["center"] - receptor_ring["center"]
    centroid_distance = float(np.linalg.norm(vector))
    if centroid_distance > 5.5:
        return None
    cosine = abs(float(np.dot(ligand_ring["normal"], receptor_ring["normal"])))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    if angle <= 30.0:
        lateral = float(np.linalg.norm(vector - np.dot(vector, receptor_ring["normal"]) * receptor_ring["normal"]))
        if 3.3 <= centroid_distance <= 4.5 and lateral <= 2.2:
            return centroid_distance, "parallel"
    elif angle >= 60.0 and 3.5 <= centroid_distance <= 5.5:
        return centroid_distance, "t_shaped"
    return None


def analyze_interactions(
    complex_pdb: Path,
    output_csv: Path,
    cys_res_num: int,
    target_chain: str | None = None,
    max_catalytic_attack_A: float = 4.5,
    max_hydride_transfer_A: float = 5.0,
) -> pd.DataFrame:
    """Profile protein-ligand interactions with explicit geometric cutoffs.

    Only contacts inside the configured distance windows are reported; catalytic
    and hydride-transfer rows are not emitted for poses that are merely in the
    same protein-ligand complex.
    """
    receptor, ligand = parse_complex(complex_pdb)
    if not ligand:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame([], columns=INTERACTION_COLUMNS)
        empty.to_csv(output_csv, index=False)
        return empty

    lig_heavy = [a for a in ligand if a["element"].upper() != "H" and a["name"] != "HD"] or ligand
    lig_polar = [a for a in lig_heavy if a["element"].upper() in {"O", "N", "S"}]
    lig_c = [a for a in lig_heavy if a["element"].upper() == "C"]
    lig_halogens = [a for a in lig_heavy if a["element"].upper() in {"F", "CL", "BR", "I"}]

    # Prefer a genuine C=O / C=N / C=S bond over an arbitrary C-O single bond.
    reactive_pairs = [
        (carbon, polar, distance(carbon["coord"], polar["coord"]))
        for carbon in lig_c
        for polar in lig_polar
        if 1.15 <= distance(carbon["coord"], polar["coord"]) <= 1.34
    ]
    if reactive_pairs:
        carbonyl_c = min(reactive_pairs, key=lambda item: item[2])[0]
    elif lig_polar and lig_c:
        pairs = [(c, p, distance(c["coord"], p["coord"])) for c in lig_c for p in lig_polar]
        carbonyl_c = min(pairs, key=lambda item: item[2])[0]
    elif lig_c:
        carbonyl_c = lig_c[0]
    else:
        carbonyl_c = lig_heavy[0]

    aliphatic = (
        [a for a in lig_c if a["serial"] != carbonyl_c["serial"]]
        if lig_polar
        else lig_c
    ) or lig_c or lig_heavy
    ligand_rings = _ligand_aromatic_rings(ligand)
    receptor_rings = _receptor_aromatic_rings(receptor, target_chain)
    rows: list[dict] = []
    seen_contacts: set[tuple[str, str, str, str]] = set()

    def _add_row(itype: str, lig_name: str, rec_res: str, rec_atom: str, dist_val: float) -> None:
        key = (itype, lig_name, rec_res, rec_atom)
        if key not in seen_contacts:
            seen_contacts.add(key)
            rows.append({
                "Type": itype,
                "Ligand_Atom": lig_name,
                "Receptor_Res": rec_res,
                "Receptor_Atom": rec_atom,
                "Distance_A": round(float(dist_val), 3),
            })

    for atom in receptor:
        if target_chain and atom["chain"] != target_chain and atom["resname"] not in (COFACTOR_RES | METAL_RES | {"HOH", "WAT"}):
            continue
        r_res_label = f"{atom['resname']}{atom['resseq']}"
        r_el = atom["element"].upper()

        # 1. Catalytic attack: only within the configured nucleophile window.
        if atom["resseq"] == cys_res_num and atom["name"] in {"SG", "OG", "OG1", "NZ", "OD1", "OE1", "NE2"}:
            attack_distance = distance(carbonyl_c["coord"], atom["coord"])
            if attack_distance <= float(max_catalytic_attack_A):
                _add_row("Catalytic_Attack", carbonyl_c["name"], r_res_label, atom["name"], attack_distance)

        # 2. Hydride transfer / cofactor reactive contact: only within the window.
        if atom["resname"] in COFACTOR_RES and atom["name"] in {"C4N", "C4", "N5", "C4X", "FE"}:
            hydride_distance = distance(carbonyl_c["coord"], atom["coord"])
            if hydride_distance <= float(max_hydride_transfer_A):
                _add_row("Hydride_Transfer", carbonyl_c["name"], r_res_label, atom["name"], hydride_distance)

        # 3. Metal coordination across all ligand polar atoms
        if atom["resname"] in METAL_RES or (atom["resname"] in {"HEM", "HEC"} and atom["name"] == "FE"):
            for lp in lig_polar:
                d_m = distance(lp["coord"], atom["coord"])
                if 1.65 <= d_m <= 3.20:
                    _add_row("Metal_Coordination", lp["name"], r_res_label, atom["name"], d_m)

        # 4. Hydrogen bonds across all ligand polar atoms (distance-based screen)
        if r_el in {"N", "O", "S"}:
            for lp in lig_polar:
                d_hb = distance(lp["coord"], atom["coord"])
                if 2.20 <= d_hb <= 3.50:
                    _add_row("Hydrogen_Bond", lp["name"], r_res_label, atom["name"], d_hb)

        # 5. Salt bridges between ligand polar atoms and charged receptor sidechains
        if (atom["resname"], atom["name"]) in CHARGED_REC_ATOMS:
            for lp in lig_polar:
                d_sb = distance(lp["coord"], atom["coord"])
                if 2.40 <= d_sb <= 4.00:
                    _add_row("Salt_Bridge", lp["name"], r_res_label, atom["name"], d_sb)

        # 6. Halogen bonds (C-X ... O/N/S)
        if r_el in {"O", "N", "S"} and lig_halogens:
            for lx in lig_halogens:
                d_xb = distance(lx["coord"], atom["coord"])
                if 2.70 <= d_xb <= 3.80:
                    _add_row("Halogen_Bond", lx["name"], r_res_label, atom["name"], d_xb)

    # 7. Pi-stacking between ligand rings and receptor aromatic rings (centroid based)
    pi_hits: dict[str, dict] = {}
    for receptor_ring in receptor_rings:
        for ligand_ring in ligand_rings:
            geometry = _pi_stack_geometry(ligand_ring, receptor_ring)
            if geometry is None:
                continue
            centroid_distance, _ = geometry
            label = receptor_ring["label"]
            if label not in pi_hits or centroid_distance < pi_hits[label]["Distance_A"]:
                pi_hits[label] = {
                    "Type": "Pi_Stacking",
                    "Ligand_Atom": ligand_ring["atoms"][0]["name"],
                    "Receptor_Res": label,
                    "Receptor_Atom": receptor_ring["atoms"][0]["name"],
                    "Distance_A": round(centroid_distance, 3),
                }

    # 8. Hydrophobic contacts: shortest carbon-carbon contact per residue
    hydro_hits: dict[tuple[str, str, int], dict] = {}
    for atom in receptor:
        if target_chain and atom["chain"] != target_chain:
            continue
        rname = atom["resname"]
        key = (atom["chain"], rname, atom["resseq"])
        if rname in HYDROPHOBIC_RES and atom["element"].upper() == "C" and atom["name"] not in {"C", "CA"}:
            for lc in aliphatic:
                d = distance(lc["coord"], atom["coord"])
                if 3.3 <= d <= 4.0 and (key not in hydro_hits or d < hydro_hits[key]["Distance_A"]):
                    hydro_hits[key] = {
                        "Type": "Hydrophobic_Contact",
                        "Ligand_Atom": lc["name"],
                        "Receptor_Res": f"{rname}{atom['resseq']}",
                        "Receptor_Atom": atom["name"],
                        "Distance_A": round(d, 3),
                    }
    rows.extend(pi_hits.values())
    rows.extend(hydro_hits.values())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=INTERACTION_COLUMNS)
    frame.to_csv(output_csv, index=False)
    return frame


def _render_matplotlib_fallback(
    complex_pdb: Path,
    interactions: pd.DataFrame,
    png_path: Path,
    pse_path: Path,
    prefix: str,
    png_width: int = 2400,
    png_height: int = 1800,
) -> None:
    status_path = pse_path.with_name(f"{prefix}_render_status.json")
    # PyMOL is unavailable in this branch: remove figures left by an earlier run
    # so a stale session/PNG cannot be mistaken for the current docking result.
    for stale in (pse_path, png_path):
        if stale.is_file():
            stale.unlink()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        receptor, ligand = parse_complex(complex_pdb)
        dpi = 300
        fig = plt.figure(figsize=(png_width / dpi, png_height / dpi), dpi=dpi, facecolor="#F6F1E4")
        ax = fig.add_subplot(111, projection="3d", facecolor="#F6F1E4")
        if ligand:
            lx = [a["coord"][0] for a in ligand]
            ly = [a["coord"][1] for a in ligand]
            lz = [a["coord"][2] for a in ligand]
            # Draw fountain-pen covalent bonds for ligand
            for i in range(len(ligand)):
                for j in range(i + 1, len(ligand)):
                    if distance(ligand[i]["coord"], ligand[j]["coord"]) <= 1.88:
                        ax.plot(
                            [ligand[i]["coord"][0], ligand[j]["coord"][0]],
                            [ligand[i]["coord"][1], ligand[j]["coord"][1]],
                            [ligand[i]["coord"][2], ligand[j]["coord"][2]],
                            color="#2B2520", linewidth=2.0, alpha=0.9,
                        )
            ax.scatter(lx, ly, lz, c="#3B5A75", edgecolors="#2B2520", linewidths=1.2, s=95, label=f"Ligand Sketch ({prefix})")
        pocket = [
            a for a in receptor
            if any(distance(a["coord"], la["coord"]) <= 6.5 for la in ligand)
        ]
        if pocket:
            px = [a["coord"][0] for a in pocket]
            py = [a["coord"][1] for a in pocket]
            pz = [a["coord"][2] for a in pocket]
            ax.scatter(px, py, pz, c="#C9BFA8", edgecolors="#5C5346", linewidths=0.8, s=32, alpha=0.75, label="Pocket Frame (<6.5 A)")
        ax.set_title(f"Plate I: {prefix} Active-Site Sketch ({len(interactions)} contacts)", color="#2B2520", fontfamily="serif")
        ax.legend(loc="upper right", facecolor="#FBF8F1", edgecolor="#2B2520")
        fig.tight_layout()
        fig.savefig(png_path, dpi=dpi, facecolor="#F6F1E4")
        plt.close(fig)
        status_path.write_text(json.dumps({
            "pymol_available": False,
            "fallback_rendered": True,
            "png_present": True,
            "note": "PyMOL was unavailable; a schematic matplotlib figure was generated instead.",
        }, indent=2), encoding="utf-8")
    except Exception as exc:
        status_path.write_text(json.dumps({
            "pymol_available": False,
            "fallback_rendered": False,
            "png_present": png_path.is_file() and png_path.stat().st_size > 0,
            "error": f"{type(exc).__name__}: {exc}",
        }, indent=2), encoding="utf-8")
        LOGGER.debug("Matplotlib fallback figure generation skipped: %s", exc)


def generate_pymol_script(
    complex_pdb: Path,
    interactions: pd.DataFrame,
    figures_dir: Path,
    prefix: str,
    pymol_executable: str = "pymol",
    target_chain: str = "A",
    png_width: int = 2400,
    png_height: int = 1800,
) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    script = figures_dir / f"{prefix}_render.pml"
    pse = figures_dir / f"{prefix}_session.pse"
    png = figures_dir / f"{prefix}_300dpi.png"
    cat_resi = "+".join(
        str(value)
        for value in interactions.loc[
            interactions["Type"] == "Catalytic_Attack", "Receptor_Res"
        ].str.extract(r"(\d+)")[0].dropna().unique()
    ) or "none"
    hbond_resi = "+".join(
        str(value)
        for value in interactions.loc[
            interactions["Type"] == "Hydrogen_Bond", "Receptor_Res"
        ].str.extract(r"(\d+)")[0].dropna().unique()
    ) or "none"
    hydro_resi = "+".join(
        str(value)
        for value in interactions.loc[
            interactions["Type"] == "Hydrophobic_Contact", "Receptor_Res"
        ].str.extract(r"(\d+)")[0].dropna().unique()
    ) or "none"
    complex_path = complex_pdb.resolve()
    pse_path = pse.resolve()
    png_path = png.resolve()
    chain_clause = f"chain {target_chain} and " if target_chain else ""
    script.write_text(
        f'''reinitialize
load "{complex_path.as_posix()}", complex
bg_color white
set ray_opaque_background, off
hide everything, all
show cartoon, polymer.protein
color gray85, polymer.protein
select ligand, resn LIG
show sticks, ligand
show spheres, ligand
color cyan, ligand and elem C
select cofactor, resn NAD+NAP+NDP
show sticks, cofactor
color tv_yellow, cofactor
select cat_cys, {chain_clause}resn CYS+SER and resi {cat_resi}
show sticks, cat_cys
color magenta, cat_cys
select hbond_residues, {chain_clause}polymer.protein and resi {hbond_resi}
show sticks, hbond_residues
color slate, hbond_residues
select hydro_residues, {chain_clause}polymer.protein and resi {hydro_resi}
show sticks, hydro_residues
color smudge, hydro_residues
create hydro_pocket_surf, hydro_residues
show surface, hydro_pocket_surf
set transparency, 72, hydro_pocket_surf
''',
        encoding="utf-8",
    )
    with script.open("a", encoding="utf-8") as handle:
        for index, row in interactions.iterrows():
            if row["Type"] in {"Catalytic_Attack", "Hydrogen_Bond"}:
                resi = "".join(ch for ch in str(row["Receptor_Res"]) if ch.isdigit())
                color = "magenta" if row["Type"] == "Catalytic_Attack" else "tv_orange"
                handle.write(
                    f'distance d{index}, (ligand and name {row["Ligand_Atom"]}), '
                    f'({chain_clause}resi {resi} and name {row["Receptor_Atom"]})\n'
                    f'color {color}, d{index}\n'
                )
        handle.write(
            f"orient ligand\nzoom (ligand or cat_cys), 5.0\nclip slab, 18\n"
            f"save {pse_path.as_posix()}\nray {png_width}, {png_height}\npng {png_path.as_posix()}, dpi=300\nquit\n"
        )
    try:
        resolved = require_executable(pymol_executable)
    except FileNotFoundError:
        _render_matplotlib_fallback(complex_pdb, interactions, png_path, pse_path, prefix, png_width, png_height)
        raise
    executable_path = Path(resolved)
    if executable_path.name.lower() == "pymolwin.exe":
        bundled_python = executable_path.with_name("python.exe")
        if not bundled_python.exists():
            raise FileNotFoundError(f"PyMOL bundled Python was not found beside {resolved}")
        command = [str(bundled_python), "-m", "pymol", "-cq", str(script)]
    else:
        command = [resolved, "-cq", str(script)]
    subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    status_path = figures_dir / f"{prefix}_render_status.json"
    status_path.write_text(json.dumps({
        "pymol_available": True,
        "fallback_rendered": False,
        "pse_present": pse.is_file(),
        "png_present": png.is_file() and png.stat().st_size > 0,
        "pymol_executable": str(resolved),
    }, indent=2), encoding="utf-8")
    return script
