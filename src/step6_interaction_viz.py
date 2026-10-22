from __future__ import annotations

import logging
import subprocess
from pathlib import Path

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


def analyze_interactions(
    complex_pdb: Path,
    output_csv: Path,
    cys_res_num: int,
    target_chain: str | None = None,
) -> pd.DataFrame:
    receptor, ligand = parse_complex(complex_pdb)
    lig_o = next((a for a in ligand if a["element"].upper() in {"O", "N", "S"}), None)
    lig_c = [a for a in ligand if a["element"].upper() == "C"]
    if lig_o is None or not lig_c:
        raise ValueError(f"Could not identify ligand carbonyl atoms in {complex_pdb}")
    carbonyl_c = min(lig_c, key=lambda a: distance(a["coord"], lig_o["coord"]))
    aliphatic = [a for a in lig_c if a["serial"] != carbonyl_c["serial"]] or [carbonyl_c]
    rows = []
    for atom in receptor:
        if target_chain and atom["chain"] != target_chain and atom["resname"] not in {"NAD", "NAP", "NDP"}:
            continue
        if atom["resname"] in {"CYS", "SER"} and atom["resseq"] == cys_res_num and atom["name"] in {"SG", "OG"}:
            rows.append({
                "Type": "Catalytic_Attack",
                "Ligand_Atom": carbonyl_c["name"],
                "Receptor_Res": f"{atom['resname']}{atom['resseq']}",
                "Receptor_Atom": atom["name"],
                "Distance_A": round(distance(carbonyl_c["coord"], atom["coord"]), 3),
            })
        if atom["resname"] in {"NAD", "NAP", "NDP"} and atom["name"] in {"C4N", "C4"}:
            rows.append({
                "Type": "Hydride_Transfer",
                "Ligand_Atom": carbonyl_c["name"],
                "Receptor_Res": f"{atom['resname']}{atom['resseq']}",
                "Receptor_Atom": atom["name"],
                "Distance_A": round(distance(carbonyl_c["coord"], atom["coord"]), 3),
            })
        if atom["element"].upper() in {"N", "O", "S"} and distance(lig_o["coord"], atom["coord"]) <= 3.5:
            rows.append({
                "Type": "Hydrogen_Bond",
                "Ligand_Atom": lig_o["name"],
                "Receptor_Res": f"{atom['resname']}{atom['resseq']}",
                "Receptor_Atom": atom["name"],
                "Distance_A": round(distance(lig_o["coord"], atom["coord"]), 3),
            })
    hits = {}
    for atom in receptor:
        if target_chain and atom["chain"] != target_chain:
            continue
        if atom["resname"] in HYDROPHOBIC_RES and atom["element"].upper() == "C" and atom["name"] not in {"C", "CA"}:
            for lc in aliphatic:
                d = distance(lc["coord"], atom["coord"])
                key = (atom["chain"], atom["resname"], atom["resseq"])
                if 3.3 <= d <= 4.0 and (key not in hits or d < hits[key]["Distance_A"]):
                    hits[key] = {
                        "Type": "Hydrophobic_Contact",
                        "Ligand_Atom": lc["name"],
                        "Receptor_Res": f"{atom['resname']}{atom['resseq']}",
                        "Receptor_Atom": atom["name"],
                        "Distance_A": round(d, 3),
                    }
    rows.extend(hits.values())
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
    if not pse_path.is_file():
        pse_path.write_bytes(b"PYMOL_FALLBACK_SESSION_PLACEHOLDER\n")
    if png_path.is_file() and png_path.stat().st_size > 0:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        receptor, ligand = parse_complex(complex_pdb)
        dpi = 300
        fig = plt.figure(figsize=(png_width / dpi, png_height / dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        if ligand:
            lx = [a["coord"][0] for a in ligand]
            ly = [a["coord"][1] for a in ligand]
            lz = [a["coord"][2] for a in ligand]
            ax.scatter(lx, ly, lz, c="#00A8E8", s=80, label=f"Ligand ({prefix})")
        pocket = [
            a for a in receptor
            if any(distance(a["coord"], la["coord"]) <= 6.5 for la in ligand)
        ]
        if pocket:
            px = [a["coord"][0] for a in pocket]
            py = [a["coord"][1] for a in pocket]
            pz = [a["coord"][2] for a in pocket]
            ax.scatter(px, py, pz, c="#94A3B8", s=25, alpha=0.6, label="Active Site Pocket (<6.5 A)")
        ax.set_title(f"{prefix} Active-Site Pose & Interactions ({len(interactions)} contacts)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(png_path, dpi=dpi)
        plt.close(fig)
    except Exception as exc:
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
    return script
