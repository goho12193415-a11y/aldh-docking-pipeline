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


AROMATIC_RES = {"PHE", "TYR", "TRP", "HIS"}
CHARGED_REC_ATOMS = {
    ("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"),
    ("LYS", "NZ"), ("ARG", "NH1"), ("ARG", "NH2"), ("ARG", "NE"),
    ("HIS", "ND1"), ("HIS", "NE2"),
}
METAL_RES = {"ZN", "MG", "FE", "MN", "CA", "CU", "NI", "CO"}
COFACTOR_RES = {"NAD", "NAP", "NDP", "FAD", "FMN", "HEM", "HEC", "HEA"}


def analyze_interactions(
    complex_pdb: Path,
    output_csv: Path,
    cys_res_num: int,
    target_chain: str | None = None,
) -> pd.DataFrame:
    """Comprehensive 3D protein-ligand interaction profiler across all ligand polar, charged, aromatic, halogen, and aliphatic atoms."""
    receptor, ligand = parse_complex(complex_pdb)
    if not ligand:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame([], columns=INTERACTION_COLUMNS)
        empty.to_csv(output_csv, index=False)
        return empty

    lig_heavy = [a for a in ligand if a["element"].upper() != "H" and a["name"] != "HD"] or ligand
    lig_polar = [a for a in lig_heavy if a["element"].upper() in {"O", "N", "S"} or a["name"] in {"OA", "NA", "SA"}]
    lig_c = [a for a in lig_heavy if a["element"].upper() == "C" or a["name"] == "A"]
    lig_halogens = [a for a in lig_heavy if a["element"].upper() in {"F", "CL", "BR", "I"}]

    if lig_polar and lig_c:
        # Pick the carbon bonded to a polar atom (e.g., carbonyl/warhead carbon)
        pairs = [(c, p, distance(c["coord"], p["coord"])) for c in lig_c for p in lig_polar]
        carbonyl_c = min(pairs, key=lambda item: item[2])[0]
    elif lig_c:
        carbonyl_c = lig_c[0]
    else:
        carbonyl_c = lig_heavy[0]

    aliphatic = ([a for a in lig_c if a["serial"] != carbonyl_c["serial"]] if lig_polar else lig_c) or (lig_c if lig_c else lig_heavy)
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

        # 1. Catalytic attack (CYS, SER, THR, LYS, ASP, GLU, HIS)
        if atom["resseq"] == cys_res_num and atom["name"] in {"SG", "OG", "OG1", "NZ", "OD1", "OE1", "NE2"}:
            _add_row("Catalytic_Attack", carbonyl_c["name"], r_res_label, atom["name"], distance(carbonyl_c["coord"], atom["coord"]))

        # 2. Hydride transfer / cofactor reactive contact (NAD C4N, FAD N5, HEM FE)
        if atom["resname"] in COFACTOR_RES and atom["name"] in {"C4N", "C4", "N5", "C4X", "FE"}:
            _add_row("Hydride_Transfer", carbonyl_c["name"], r_res_label, atom["name"], distance(carbonyl_c["coord"], atom["coord"]))

        # 3. Metal coordination across all ligand polar atoms
        if atom["resname"] in METAL_RES or (atom["resname"] in {"HEM", "HEC"} and atom["name"] == "FE"):
            for lp in lig_polar:
                d_m = distance(lp["coord"], atom["coord"])
                if 1.65 <= d_m <= 3.20:
                    _add_row("Metal_Coordination", lp["name"], r_res_label, atom["name"], d_m)

        # 4. Hydrogen bonds across ALL ligand polar atoms (O, N, S)
        if r_el in {"N", "O", "S"}:
            for lp in lig_polar:
                d_hb = distance(lp["coord"], atom["coord"])
                if 2.20 <= d_hb <= 3.50:
                    _add_row("Hydrogen_Bond", lp["name"], r_res_label, atom["name"], d_hb)

        # 5. Salt bridges between polar/charged ligand groups and charged receptor sidechains
        if (atom["resname"], atom["name"]) in CHARGED_REC_ATOMS:
            for lp in lig_polar:
                d_sb = distance(lp["coord"], atom["coord"])
                if 2.40 <= d_sb <= 4.00 and (abs(float(lp.get("charge") or 0.0)) >= 0.25 or lp["element"].upper() in {"O", "N"}):
                    _add_row("Salt_Bridge", lp["name"], r_res_label, atom["name"], d_sb)

        # 6. Halogen bonds (C-X ... O/N/S)
        if r_el in {"O", "N", "S"} and lig_halogens:
            for lx in lig_halogens:
                d_xb = distance(lx["coord"], atom["coord"])
                if 2.70 <= d_xb <= 3.80:
                    _add_row("Halogen_Bond", lx["name"], r_res_label, atom["name"], d_xb)

    # 7. Pi-Stacking and Hydrophobic contacts (per-residue shortest distance)
    hydro_hits: dict[tuple[str, str, int], dict] = {}
    pi_hits: dict[tuple[str, str, int], dict] = {}
    for atom in receptor:
        if target_chain and atom["chain"] != target_chain:
            continue
        rname = atom["resname"]
        key = (atom["chain"], rname, atom["resseq"])
        if rname in AROMATIC_RES and atom["name"] in {"CG", "CD1", "CD2", "CE1", "CE2", "CZ", "CE3", "CZ2", "CZ3"}:
            for lc in lig_c:
                d_pi = distance(lc["coord"], atom["coord"])
                if 3.35 <= d_pi <= 4.30 and (key not in pi_hits or d_pi < pi_hits[key]["Distance_A"]):
                    pi_hits[key] = {
                        "Type": "Pi_Stacking",
                        "Ligand_Atom": lc["name"],
                        "Receptor_Res": f"{rname}{atom['resseq']}",
                        "Receptor_Atom": atom["name"],
                        "Distance_A": round(d_pi, 3),
                    }
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
