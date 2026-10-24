from __future__ import annotations

import json
import logging
import math
import subprocess
from pathlib import Path

import numpy as np

from .common import FallbackAtom, PDBIO, PDBParser, Select, Superimposer, align_ca_residues, atom_dict, convert_cif_to_pdb, distance, require_executable

LOGGER = logging.getLogger(__name__)
COFACTORS = {"NAD", "NAP", "NDP", "FAD", "FMN", "HEM", "HEC", "HEA"}
CATALYTIC_METALS = {"ZN", "MG", "FE", "MN", "CU", "CO", "NI", "CA"}


def _make_atom(name: str, coord: np.ndarray, element: str, serial: int = 9999):
    fullname = f" {name:<3s}"[:4] if len(name) < 4 else name[:4]
    try:
        from Bio.PDB.Atom import Atom as BioAtom  # type: ignore

        return BioAtom(name, np.array(coord, dtype=float), 20.0, 1.0, " ", fullname, serial, element.upper())
    except ImportError:
        return FallbackAtom(name, np.array(coord, dtype=float), 20.0, 1.0, " ", fullname, serial, element.upper())


class ReceptorSelect(Select):
    """Select target protein chain, cofactors, catalytic metals, and conserved/bridging waters while resolving AltLocs."""

    def __init__(
        self,
        chain_id: str,
        keep_cofactors: set[str],
        cofactor_chain: str | None = None,
        conserved_waters: set[int] | None = None,
        catalytic_coord: np.ndarray | None = None,
        keep_catalytic_metals: bool = True,
        metal_cutoff_A: float = 8.0,
    ) -> None:
        self.chain_id = chain_id
        self.keep_cofactors = keep_cofactors
        self.cofactor_chain = cofactor_chain or chain_id
        self.conserved_waters = conserved_waters or set()
        self.catalytic_coord = catalytic_coord
        self.keep_catalytic_metals = keep_catalytic_metals
        self.metal_cutoff_A = metal_cutoff_A
        self._chosen_altlocs: dict[tuple[str, tuple, str], str] = {}

    def accept_chain(self, chain) -> int:
        return 1

    def accept_residue(self, residue) -> int:
        name = residue.get_resname().strip()
        parent_chain = residue.get_parent().id if residue.get_parent() is not None else ""
        if residue.id[0] == " ":
            return int(parent_chain == self.chain_id)
        if name in self.keep_cofactors and parent_chain == self.cofactor_chain:
            return 1
        if name in {"HOH", "WAT"} and parent_chain in {self.chain_id, self.cofactor_chain} and int(residue.id[1]) in self.conserved_waters:
            return 1
        if self.keep_catalytic_metals and name in CATALYTIC_METALS and parent_chain in {self.chain_id, self.cofactor_chain}:
            if self.catalytic_coord is None:
                return 1
            for atom in residue:
                if distance(np.array(atom.get_coord(), dtype=float), self.catalytic_coord) <= self.metal_cutoff_A:
                    return 1
        return 0

    def accept_atom(self, atom) -> int:
        altloc = atom.get_altloc()
        if altloc in {" ", ""}:
            return 1
        residue = atom.get_parent()
        if residue is None:
            return 1
        chain_id = residue.get_parent().id if residue.get_parent() is not None else ""
        key = (chain_id, residue.id, atom.get_name().strip())
        if key not in self._chosen_altlocs:
            # PDBIO iterates the unpacked altloc children, so build candidates from
            # get_unpacked_list() rather than the residue wrapper to keep identities comparable.
            candidates = [
                a for a in residue.get_unpacked_list()
                if a.get_name().strip() == atom.get_name().strip()
            ]
            candidates.sort(
                key=lambda item: (
                    -float(item.get_occupancy() if item.get_occupancy() is not None else 0.0),
                    0 if item.get_altloc() in {" ", "", "A"} else 1,
                    item.get_altloc(),
                )
            )
            chosen = candidates[0]
            self._chosen_altlocs[key] = chosen.get_altloc() or " "
        if (atom.get_altloc() or " ") == self._chosen_altlocs[key]:
            try:
                atom.set_altloc(" ")
                atom.set_occupancy(1.0)
            except AttributeError:
                pass
            return 1
        return 0


def repair_missing_backbone_oxygens(structure, chain_id: str) -> list[dict]:
    """Deterministically reconstruct missing peptide carbonyl O atoms from N-CA-C geometry."""
    repaired = []
    if chain_id not in structure[0]:
        return repaired
    chain = structure[0][chain_id]
    residues = [r for r in chain if r.id[0] == " "]
    for idx, residue in enumerate(residues):
        if "O" in residue:
            continue
        if not all(name in residue for name in ("N", "CA", "C")):
            continue
        n_coord = np.array(residue["N"].get_coord(), dtype=float)
        ca_coord = np.array(residue["CA"].get_coord(), dtype=float)
        c_coord = np.array(residue["C"].get_coord(), dtype=float)
        if idx + 1 < len(residues) and "N" in residues[idx + 1]:
            next_n = np.array(residues[idx + 1]["N"].get_coord(), dtype=float)
            v1 = ca_coord - c_coord
            v2 = next_n - c_coord
            v1 /= np.linalg.norm(v1) + 1e-8
            v2 /= np.linalg.norm(v2) + 1e-8
            bisector = -(v1 + v2)
        else:
            v1 = ca_coord - c_coord
            v2 = n_coord - ca_coord
            v1 /= np.linalg.norm(v1) + 1e-8
            v2 /= np.linalg.norm(v2) + 1e-8
            bisector = -(v1 - 0.5 * v2)
        norm = np.linalg.norm(bisector)
        if norm < 1e-6:
            continue
        o_coord = c_coord + (bisector / norm) * 1.231
        residue.add(_make_atom("O", o_coord, "O", 9998))
        repaired.append({"chain": chain_id, "residue": int(residue.id[1]), "resname": residue.get_resname().strip(), "reconstructed_atom": "O"})
    return repaired


def reconstruct_missing_sidechains(structure, chain_id: str) -> list[dict]:
    """Deterministically reconstruct missing CB and key catalytic sidechain functional atoms (SG, OG, CG, ND2, OD1, OE1, NZ)."""
    repaired = []
    if chain_id not in structure[0]:
        return repaired
    chain = structure[0][chain_id]
    residues = [r for r in chain if r.id[0] == " "]
    env_coords = [
        np.array(a.get_coord(), dtype=float)
        for r in residues
        for a in r
        if a.get_name().strip() in {"N", "CA", "C", "O"}
    ]

    terminal_specs = {
        "CYS": ("SG", "S", 1.81),
        "SER": ("OG", "O", 1.42),
        "THR": ("OG1", "O", 1.43),
        # These terminal atoms are bonded to CG, not CB; the CB->ND2/OD1
        # through-space distance is ~2.45 A.
        "ASN": ("ND2", "N", 2.45),
        "ASP": ("OD1", "O", 2.45),
        "GLU": ("OE1", "O", 2.45),
        "GLN": ("NE2", "N", 2.45),
        "HIS": ("NE2", "N", 2.40),
        "LYS": ("NZ", "N", 3.60),
    }

    for residue in residues:
        rname = residue.get_resname().strip()
        if rname in {"GLY", "ALA"} or not all(k in residue for k in ("N", "CA", "C")):
            continue
        n_c = np.array(residue["N"].get_coord(), dtype=float)
        ca_c = np.array(residue["CA"].get_coord(), dtype=float)
        c_c = np.array(residue["C"].get_coord(), dtype=float)

        if "CB" not in residue:
            u1 = n_c - ca_c
            u2 = c_c - ca_c
            u1 /= np.linalg.norm(u1) + 1e-8
            u2 /= np.linalg.norm(u2) + 1e-8
            bisect = -(u1 + u2)
            bisect /= np.linalg.norm(bisect) + 1e-8
            perp = np.cross(u1, u2)
            perp /= np.linalg.norm(perp) + 1e-8
            cb_dir = math.cos(0.615) * bisect + math.sin(0.615) * perp
            cb_dir /= np.linalg.norm(cb_dir) + 1e-8
            cb_coord = ca_c + cb_dir * 1.53
            residue.add(_make_atom("CB", cb_coord, "C", 9995))
            repaired.append({"chain": chain_id, "residue": int(residue.id[1]), "resname": rname, "reconstructed_atom": "CB"})

        if rname in terminal_specs:
            atom_name, elem, bond_dist = terminal_specs[rname]
            if atom_name not in residue and "CB" in residue:
                cb_coord = np.array(residue["CB"].get_coord(), dtype=float)
                axis = cb_coord - ca_c
                axis /= np.linalg.norm(axis) + 1e-8
                ref = n_c - ca_c
                perp1 = np.cross(axis, ref)
                if np.linalg.norm(perp1) < 1e-6:
                    perp1 = np.array([1.0, 0.0, 0.0], dtype=float)
                perp1 /= np.linalg.norm(perp1) + 1e-8
                perp2 = np.cross(axis, perp1)
                best_pos = None
                best_clearance = -1.0
                for chi1_deg in (-60.0, 180.0, 60.0):
                    rad = math.radians(chi1_deg)
                    rot_vec = math.cos(rad) * perp1 + math.sin(rad) * perp2
                    direction = 0.35 * axis + 0.936 * rot_vec
                    direction /= np.linalg.norm(direction) + 1e-8
                    cand = cb_coord + direction * bond_dist
                    clearance = min((distance(cand, ec) for ec in env_coords if distance(ec, ca_c) > 2.0), default=5.0)
                    if clearance > best_clearance:
                        best_clearance = clearance
                        best_pos = cand
                if best_pos is not None:
                    residue.add(_make_atom(atom_name, best_pos, elem, 9996))
                    repaired.append({"chain": chain_id, "residue": int(residue.id[1]), "resname": rname, "reconstructed_atom": atom_name})
    return repaired


def detect_bridging_waters(
    structure,
    chain_id: str,
    keep_cofactors: set[str],
    catalytic_coord: np.ndarray | None = None,
    max_waters: int = 3,
) -> list[dict]:
    """Identify crystallographic waters that form >=2 H-bonds bridging distinct active-site protein residues or cofactor atoms."""
    if chain_id not in structure[0]:
        return []
    model = structure[0]
    protein_polar = []
    cofactor_polar = []
    waters = []

    for ch in model:
        for r in ch:
            rname = r.get_resname().strip()
            if rname in {"HOH", "WAT"}:
                for a in r:
                    if (getattr(a, "element", "") or a.get_name()[:1]).strip().upper() == "O":
                        coord = np.array(a.get_coord(), dtype=float)
                        if catalytic_coord is None or distance(coord, catalytic_coord) <= 7.5:
                            waters.append((ch.id, int(r.id[1]), coord))
                            break
            elif r.id[0] == " " and ch.id == chain_id:
                for a in r:
                    elem = (getattr(a, "element", "") or a.get_name()[:1]).strip().upper()
                    if elem in {"N", "O", "S"}:
                        coord = np.array(a.get_coord(), dtype=float)
                        if catalytic_coord is None or distance(coord, catalytic_coord) <= 10.0:
                            protein_polar.append((rname, int(r.id[1]), a.get_name().strip(), coord))
            elif rname in keep_cofactors or rname in CATALYTIC_METALS:
                for a in r:
                    elem = (getattr(a, "element", "") or a.get_name()[:1]).strip().upper()
                    if elem != "H":
                        cofactor_polar.append((rname, int(r.id[1]), a.get_name().strip(), np.array(a.get_coord(), dtype=float)))

    candidates = []
    for w_chain, w_num, w_coord in waters:
        prot_hits = [
            (rn, rnum, an) for rn, rnum, an, pc in protein_polar if 2.2 <= distance(w_coord, pc) <= 3.5
        ]
        cof_hits = [
            f"{rn}{rnum}:{an}" for rn, rnum, an, cc in cofactor_polar if 2.2 <= distance(w_coord, cc) <= 3.6
        ]
        distinct_prot_res = {rnum for _, rnum, _ in prot_hits}
        prot_contacts = [f"{rn}{rnum}:{an}" for rn, rnum, an in prot_hits]
        if (prot_contacts and cof_hits) or len(distinct_prot_res) >= 2:
            dist_cat = round(distance(w_coord, catalytic_coord), 3) if catalytic_coord is not None else 0.0
            candidates.append({
                "chain": w_chain,
                "resseq": w_num,
                "distance_to_catalytic_A": dist_cat,
                "protein_hbonds": prot_contacts[:4],
                "cofactor_hbonds": cof_hits[:4],
                "bridge_score": len(cof_hits) * 2 + len(distinct_prot_res),
            })
    candidates.sort(key=lambda item: (-item["bridge_score"], item["distance_to_catalytic_A"]))
    return candidates[:max_waters]


def optimize_cofactor_conformation(
    structure,
    chain_id: str,
    cofactor_chain: str,
    catalytic_resseq: int,
    target_distance_A: float = 4.4,
) -> dict:
    """Rotate contracted NAD/NAP nicotinamide-ribose moiety around covalent torsion bonds (O5D-C5D and C1D-N1N) without distorting bond lengths."""
    model = structure[0]
    if chain_id not in model or cofactor_chain not in model:
        return {"optimized": False, "reason": "chain_missing"}
    cat_coord = None
    for r in model[chain_id]:
        if r.id[0] == " " and int(r.id[1]) == int(catalytic_resseq) and "SG" in r:
            cat_coord = np.array(r["SG"].get_coord(), dtype=float)
            break
    if cat_coord is None:
        return {"optimized": False, "reason": "catalytic_sg_not_found"}

    nad_res = None
    for r in model[cofactor_chain]:
        if r.get_resname().strip() in {"NAD", "NAP", "NDP"} and ("C4N" in r or "C4" in r):
            nad_res = r
            break
    if nad_res is None:
        return {"optimized": False, "reason": "nad_not_found"}

    c4n_name = "C4N" if "C4N" in nad_res else "C4"
    initial_d = distance(np.array(nad_res[c4n_name].get_coord(), dtype=float), cat_coord)
    if initial_d <= 5.2:
        return {"optimized": False, "initial_d_sg_c4n_A": round(initial_d, 3), "conformation": "already_productive_in"}

    mobile_names = {
        a.get_name().strip() for a in nad_res
        if a.get_name().strip().endswith(("N", "D")) and not a.get_name().strip().startswith("P")
    }
    if not mobile_names:
        return {"optimized": False, "initial_d_sg_c4n_A": round(initial_d, 3), "reason": "no_nicotinamide_atoms"}

    # Use actual covalent bond axis (PN -> O5D) if present so bond lengths across O5D stay strictly invariant
    if "PN" in nad_res and "O5D" in nad_res:
        p1 = np.array(nad_res["PN"].get_coord(), dtype=float)
        pivot = np.array(nad_res["O5D"].get_coord(), dtype=float)
        axis = pivot - p1
    else:
        pivot_atom = nad_res["O5D"] if "O5D" in nad_res else next(iter(nad_res))
        pivot = np.array(pivot_atom.get_coord(), dtype=float)
        c4n_coord = np.array(nad_res[c4n_name].get_coord(), dtype=float)
        axis = np.cross(c4n_coord - pivot, cat_coord - pivot)

    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        return {"optimized": False, "initial_d_sg_c4n_A": round(initial_d, 3)}
    axis /= axis_norm

    protein_coords = np.array(
        [
            a.get_coord()
            for r in model[chain_id]
            if r.id[0] == " " and int(r.id[1]) != int(catalytic_resseq)
            for a in r
            if (getattr(a, "element", "") or a.get_name()[:1]).strip().upper() != "H"
        ],
        dtype=float,
    )

    orig_coords = {a.get_name().strip(): np.array(a.get_coord(), dtype=float).copy() for a in nad_res if a.get_name().strip() in mobile_names}
    best_angle = 0.0
    best_d = initial_d

    for angle_deg in np.linspace(-60.0, 60.0, 25):
        if abs(angle_deg) < 1e-3:
            continue
        theta = math.radians(float(angle_deg))
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        k_mat = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ], dtype=float)
        rot = np.eye(3) + sin_t * k_mat + (1.0 - cos_t) * np.dot(k_mat, k_mat)
        cand_coords = {name: pivot + np.dot(pos - pivot, rot.T) for name, pos in orig_coords.items()}
        cand_arr = np.array(list(cand_coords.values()), dtype=float)
        clashes = _count_clashes(cand_arr, protein_coords, cutoff=2.0)
        new_d = distance(cand_coords[c4n_name], cat_coord)
        if clashes == 0 and 3.6 <= new_d < best_d:
            best_d = new_d
            best_angle = float(angle_deg)
            for a in nad_res:
                an = a.get_name().strip()
                if an in cand_coords:
                    a.coord = cand_coords[an]
            if abs(best_d - target_distance_A) <= 0.4:
                break

    return {
        "optimized": abs(best_angle) > 0.0,
        "rotation_angle_deg": round(best_angle, 2),
        "initial_d_sg_c4n_A": round(initial_d, 3),
        "final_d_sg_c4n_A": round(best_d, 3),
    }


def prepare_flexible_receptor_pdbqt(
    cleaned_pdb: Path,
    full_receptor_pdbqt: Path,
    output_dir: Path,
    chain_id: str,
    flexible_residues: list[int],
) -> tuple[Path, Path | None]:
    """Split receptor PDBQT into rigid receptor_rigid.pdbqt and nested multi-torsion (chi1 CA-CB + chi2 CB-CG) receptor_flex.pdbqt."""
    if not flexible_residues:
        return full_receptor_pdbqt, None
    flex_set = {int(r) for r in flexible_residues}
    rigid_lines = []
    flex_by_res: dict[tuple[str, str, int], list[dict]] = {}

    for line in full_receptor_pdbqt.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            resname = line[17:20].strip()
            ch = line[21:22].strip() or "A"
            resseq = int(line[22:26].strip())
            aname = line[12:16].strip()
        except ValueError:
            rigid_lines.append(line)
            continue
        if ch == chain_id and resseq in flex_set and resname not in {"GLY", "ALA", "PRO"} and aname not in {"N", "C", "O", "H", "HN"}:
            flex_by_res.setdefault((resname, ch, resseq), []).append({"name": aname, "line": line})
            if aname == "CA":
                rigid_lines.append(line)
        else:
            rigid_lines.append(line)

    if not flex_by_res:
        return full_receptor_pdbqt, None

    flex_lines = []
    for (resname, ch, resseq), atoms in flex_by_res.items():
        ca_atom = next((a for a in atoms if a["name"] == "CA"), None)
        cb_atom = next((a for a in atoms if a["name"] == "CB"), None)
        cg_atom = next((a for a in atoms if a["name"] in {"CG", "CG1"}), None)
        side_atoms = [a for a in atoms if a["name"] not in {"CA", "CB"}]
        if not ca_atom or not cb_atom or not side_atoms:
            for a in atoms:
                if a["name"] != "CA":
                    rigid_lines.append(a["line"])
            continue
        ca_serial = int(ca_atom["line"][6:11].strip())
        cb_serial = int(cb_atom["line"][6:11].strip())
        flex_lines.append(f"BEGIN_RES {resname} {ch} {resseq:4d}")
        flex_lines.append("REMARK  status: ('A' for Active; 'I' for Inactive)")
        flex_lines.append(f"REMARK    {ca_serial:4d}  between atoms: CA_1  and  CB_2")
        flex_lines.append("ROOT")
        flex_lines.append(ca_atom["line"])
        flex_lines.append("ENDROOT")
        flex_lines.append(f"BRANCH {ca_serial:3d} {cb_serial:3d}")
        flex_lines.append(cb_atom["line"])

        # Add nested chi2 BRANCH (CB -> CG) if distal sidechain atoms exist beyond CG
        distal_atoms = [a for a in side_atoms if a is not cg_atom]
        if cg_atom is not None and distal_atoms:
            cg_serial = int(cg_atom["line"][6:11].strip())
            flex_lines.append(f"BRANCH {cb_serial:3d} {cg_serial:3d}")
            flex_lines.append(cg_atom["line"])
            for da in distal_atoms:
                flex_lines.append(da["line"])
            flex_lines.append(f"ENDBRANCH {cb_serial:3d} {cg_serial:3d}")
        else:
            for sa in side_atoms:
                flex_lines.append(sa["line"])

        flex_lines.append(f"ENDBRANCH {ca_serial:3d} {cb_serial:3d}")
        flex_lines.append(f"END_RES {resname} {ch} {resseq:4d}")

    if not flex_lines:
        return full_receptor_pdbqt, None

    rigid_path = output_dir / "receptor_rigid.pdbqt"
    flex_path = output_dir / "receptor_flex.pdbqt"
    rigid_path.write_text("\n".join(rigid_lines + ["TER"]) + "\n", encoding="utf-8")
    flex_path.write_text("\n".join(flex_lines) + "\n", encoding="utf-8")
    return rigid_path, flex_path


def transplant_cofactor(
    target_pdb: Path,
    template_pdb: Path,
    output_pdb: Path,
    chain_id: str,
    cofactor_names: set[str],
    max_rmsd_A: float = 4.0,
    audit_json: Path | None = None,
) -> dict:
    """Transplant homologous cofactors using sequence-aligned C-alpha superposition, RMSD validation, and clash relaxation."""
    parser = PDBParser(QUIET=True)
    target = parser.get_structure("target", str(target_pdb))
    template = parser.get_structure("template", str(template_pdb))
    target_chain = target[0][chain_id]
    template_chain = template[0][chain_id] if chain_id in template[0] else next(iter(template[0]))
    if any(residue.get_resname().strip() in cofactor_names for residue in target_chain):
        io = PDBIO()
        io.set_structure(target)
        io.save(str(output_pdb))
        audit = {"status": "existing_cofactor_retained", "transplanted": False}
        if audit_json:
            audit_json.parent.mkdir(parents=True, exist_ok=True)
            audit_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        return audit

    target_res = [r for r in target_chain if "CA" in r and r.id[0] == " "]
    template_res = [r for r in template_chain if "CA" in r and r.id[0] == " "]
    aligned_pairs = align_ca_residues(target_res, template_res)
    if len(aligned_pairs) < 10:
        count = min(len(target_res), len(template_res))
        if count < 10:
            raise ValueError("Not enough common C-alpha atoms for cofactor transplantation.")
        aligned_pairs = list(zip(target_res[:count], template_res[:count]))

    fixed_ca = [r1["CA"] for r1, _ in aligned_pairs]
    moving_ca = [r2["CA"] for _, r2 in aligned_pairs]
    superimposer = Superimposer()
    superimposer.set_atoms(fixed_ca, moving_ca)
    rmsd = float(getattr(superimposer, "rms", 0.0))
    if rmsd > max_rmsd_A:
        raise ValueError(
            f"Template C-alpha superposition RMSD ({rmsd:.2f} A) exceeds safe cofactor transplantation threshold ({max_rmsd_A:.2f} A)."
        )

    # Only cofactors from the template chain that was actually superimposed may
    # be transformed with that chain's rotation/translation.
    template_cofactors = [
        residue for residue in template_chain
        if residue.get_resname().strip() in cofactor_names
    ]
    protein_coords = np.array(
        [atom.get_coord() for r in target_res for atom in r if getattr(atom, "element", "C") != "H"],
        dtype=float,
    )
    transplanted_records = []
    for residue in template_cofactors:
        copied = residue.copy()
        superimposer.apply(copied)
        cof_coords = np.array([a.get_coord() for a in copied], dtype=float)
        min_dist = _min_interatomic_distance(cof_coords, protein_coords)
        clash_count = _count_clashes(cof_coords, protein_coords, cutoff=2.0)
        best_shift = np.zeros(3, dtype=float)
        if clash_count > 0 and len(protein_coords) > 0:
            best_score = (clash_count, -min_dist)
            for dx in (-0.4, -0.2, 0.0, 0.2, 0.4):
                for dy in (-0.4, -0.2, 0.0, 0.2, 0.4):
                    for dz in (-0.4, -0.2, 0.0, 0.2, 0.4):
                        shift = np.array([dx, dy, dz], dtype=float)
                        shifted = cof_coords + shift
                        c_cnt = _count_clashes(shifted, protein_coords, cutoff=2.0)
                        m_dst = _min_interatomic_distance(shifted, protein_coords)
                        if (c_cnt, -m_dst) < best_score:
                            best_score = (c_cnt, -m_dst)
                            best_shift = shift
            if np.any(best_shift != 0):
                for atom in copied:
                    atom.coord = np.array(atom.get_coord(), dtype=float) + best_shift
                cof_coords = np.array([a.get_coord() for a in copied], dtype=float)
                min_dist = _min_interatomic_distance(cof_coords, protein_coords)
                clash_count = _count_clashes(cof_coords, protein_coords, cutoff=2.0)
        if copied.id in {residue.id for residue in target_chain}:
            next_resseq = max((residue.id[1] for residue in target_chain if residue.id[0] != " "), default=9000) + 1
            copied.id = (copied.id[0], next_resseq, " ")
        target_chain.add(copied)
        transplanted_records.append({
            "resname": copied.get_resname().strip(),
            "resseq": int(copied.id[1]),
            "min_receptor_distance_A": round(float(min_dist), 3),
            "severe_clash_count_under_2A": int(clash_count),
            "micro_relaxation_shift_A": [round(float(x), 3) for x in best_shift],
        })
    io = PDBIO()
    io.set_structure(target)
    io.save(str(output_pdb))
    audit = {
        "status": "transplanted_ok" if all(r["severe_clash_count_under_2A"] == 0 for r in transplanted_records) else "transplanted_with_minor_clash_warning",
        "transplanted": True,
        "aligned_ca_count": len(aligned_pairs),
        "superposition_rmsd_A": round(rmsd, 3),
        "cofactors": transplanted_records,
    }
    if audit_json:
        audit_json.parent.mkdir(parents=True, exist_ok=True)
        audit_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _min_interatomic_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    if len(coords_a) == 0 or len(coords_b) == 0:
        return 99.9
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    return float(np.min(dists))


def _count_clashes(coords_a: np.ndarray, coords_b: np.ndarray, cutoff: float = 2.0) -> int:
    if len(coords_a) == 0 or len(coords_b) == 0:
        return 0
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    return int(np.sum(dists < cutoff))


def resolve_cofactor_chain(structure, target_chain_id: str, keep_cofactors: set[str], configured_cofactor_chain: str | None = None, catalytic_number: int | None = None) -> str:
    """Automatically resolve which chain holds the active-site cofactor."""
    preferred = configured_cofactor_chain or target_chain_id
    model = structure[0]
    if preferred in model:
        if any(r.id[0] != " " and r.get_resname().strip() in keep_cofactors for r in model[preferred]):
            return preferred
    cat_coord = None
    if target_chain_id in model and catalytic_number is not None:
        for r in model[target_chain_id]:
            if r.id[0] == " " and int(r.id[1]) == int(catalytic_number):
                for key_atom in ("SG", "OG", "NZ", "OD1", "OE1", "CA"):
                    if key_atom in r:
                        cat_coord = np.array(r[key_atom].get_coord(), dtype=float)
                        break
    best_chain = preferred
    best_dist = float("inf")
    for chain in model:
        for residue in chain:
            if residue.id[0] != " " and residue.get_resname().strip() in keep_cofactors:
                if cat_coord is None:
                    return chain.id
                for atom in residue:
                    d = distance(np.array(atom.get_coord(), dtype=float), cat_coord)
                    if d < best_dist:
                        best_dist = d
                        best_chain = chain.id
    return best_chain


CRYSTAL_BUFFER_ARTIFACTS = {
    "GAI", "GOL", "EDO", "SO4", "PO4", "ACT", "DMS", "PEG", "MPD", "TRS", "EPE", "MES",
    "CL", "NA", "K", "BR", "I", "LI", "CS", "RB", "CIT", "FMT", "ACE",
}


def superimpose_receptor_kabsch(
    mobile_pdb: Path,
    reference_pdb: Path,
    output_pdb: Path,
    mobile_chain: str = "A",
    reference_chain: str = "A",
) -> dict:
    """Superimpose a mobile receptor PDB (and all bound cofactors/waters/metals) onto a reference receptor frame using Needleman-Wunsch C-alpha alignment + Kabsch SVD."""
    parser = PDBParser(QUIET=True)
    ref_struct = parser.get_structure("ref", str(reference_pdb))
    mob_struct = parser.get_structure("mob", str(mobile_pdb))

    ref_ch = ref_struct[0][reference_chain] if reference_chain in ref_struct[0] else next(iter(ref_struct[0]))
    mob_ch = mob_struct[0][mobile_chain] if mobile_chain in mob_struct[0] else next(iter(mob_struct[0]))

    ref_res = [r for r in ref_ch if r.id[0] == " " and "CA" in r]
    mob_res = [r for r in mob_ch if r.id[0] == " " and "CA" in r]
    aligned_pairs = align_ca_residues(ref_res, mob_res)
    if len(aligned_pairs) < 3:
        count = min(len(ref_res), len(mob_res))
        if count < 3:
            raise ValueError("At least 3 C-alpha atoms are required for 3D Kabsch receptor superposition.")
        aligned_pairs = list(zip(ref_res[:count], mob_res[:count]))

    fixed_ca = [r_ref["CA"] for r_ref, _ in aligned_pairs]
    moving_ca = [r_mob["CA"] for _, r_mob in aligned_pairs]
    sup = Superimposer()
    sup.set_atoms(fixed_ca, moving_ca)
    sup.apply(mob_struct)
    rmsd = float(getattr(sup, "rms", 0.0))

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(mob_struct)
    io.save(str(output_pdb))
    return {
        "aligned": True,
        "reference_pdb": str(reference_pdb),
        "mobile_pdb": str(mobile_pdb),
        "output_pdb": str(output_pdb),
        "aligned_ca_count": len(aligned_pairs),
        "ca_rmsd_A": round(rmsd, 4),
    }


def extract_cocrystal_ligand(
    raw_pdb: Path,
    output_pdb: Path,
    chain_id: str = "A",
    keep_cofactors: set[str] | None = None,
    catalytic_coord: np.ndarray | None = None,
    min_heavy_atoms: int = 4,
) -> dict:
    """Detect and extract any bound organic co-crystal substrate/inhibitor prior to receptor cleaning for automated redocking RMSD validation."""
    excluded = set(COFACTORS) | set(CATALYTIC_METALS) | set(CRYSTAL_BUFFER_ARTIFACTS) | {"HOH", "WAT", "TIP", "SOL"}
    if keep_cofactors:
        excluded |= set(keep_cofactors)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("cocrystal_scan", str(raw_pdb))
    candidates = []

    for model in structure:
        for ch in model:
            for r in ch:
                if r.id[0] == " ":
                    continue
                rname = r.get_resname().strip()
                if rname in excluded:
                    continue
                heavy_atoms = [
                    a for a in r
                    if (getattr(a, "element", "") or a.get_name()[:1]).strip().upper() != "H"
                ]
                if len(heavy_atoms) < min_heavy_atoms:
                    continue
                coords = np.array([a.get_coord() for a in heavy_atoms], dtype=float)
                center = np.mean(coords, axis=0)
                dist_cat = distance(center, catalytic_coord) if catalytic_coord is not None else 0.0
                chain_bonus = 0 if ch.id == chain_id else 1
                candidates.append((dist_cat if catalytic_coord is not None else -len(heavy_atoms), chain_bonus, -len(heavy_atoms), ch.id, int(r.id[1]), rname, heavy_atoms))

    if not candidates:
        return {"found": False, "raw_pdb": str(raw_pdb)}

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    dist_val, _, _, c_id, resseq, rname, heavy_atoms = candidates[0]

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for idx, a in enumerate(heavy_atoms, start=9000):
        elem = (getattr(a, "element", "") or a.get_name()[:1]).strip().upper()[:2]
        x, y, z = [float(v) for v in a.get_coord()]
        aname = a.get_name().strip()[:4]
        lines.append(f"HETATM{idx:5d} {aname:>4s} LIG L{999:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {elem:>2s}")
    lines.append("END")
    output_pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "found": True,
        "raw_pdb": str(raw_pdb),
        "cocrystal_pdb": str(output_pdb),
        "resname": rname,
        "chain": c_id,
        "resseq": resseq,
        "heavy_atom_count": len(heavy_atoms),
        "distance_to_catalytic_A": round(float(dist_val), 3) if catalytic_coord is not None else None,
    }


def prepare_receptor(
    input_pdb: Path,
    output_dir: Path,
    chain_id: str,
    keep_cofactors: set[str],
    template_pdb: Path | None = None,
    cofactor_chain: str | None = None,
    conserved_waters: list[int] | None = None,
    catalytic_residue_number: int | None = None,
    auto_retain_bridging_waters: bool = False,
    optimize_cofactor_in_state: bool = False,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if input_pdb.suffix.lower() in {".cif", ".mmcif"}:
        input_pdb = convert_cif_to_pdb(input_pdb, output_dir / f"{input_pdb.stem}_from_cif.pdb")
    cleaned_pdb = output_dir / "receptor_cleaned.pdb"
    source_pdb = input_pdb
    if template_pdb:
        source_pdb = output_dir / "receptor_with_transplanted_cofactor.pdb"
        transplant_cofactor(
            input_pdb,
            template_pdb,
            source_pdb,
            chain_id,
            keep_cofactors,
            audit_json=output_dir.parent.parent / "reports" / "cofactor_transplant_audit.json",
        )

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", str(source_pdb))
    repaired_oxygens = repair_missing_backbone_oxygens(structure, chain_id)
    repaired_sidechains = reconstruct_missing_sidechains(structure, chain_id)
    if repaired_oxygens or repaired_sidechains:
        LOGGER.info(
            "Auto-repaired %d backbone O and %d sidechain atom(s) on chain %s.",
            len(repaired_oxygens),
            len(repaired_sidechains),
            chain_id,
        )

    resolved_cof_chain = resolve_cofactor_chain(
        structure,
        chain_id,
        keep_cofactors,
        cofactor_chain,
        catalytic_residue_number,
    )
    if optimize_cofactor_in_state and catalytic_residue_number is not None:
        opt_info = optimize_cofactor_conformation(structure, chain_id, resolved_cof_chain, int(catalytic_residue_number))
        if opt_info.get("optimized"):
            LOGGER.info("Rotated cofactor into productive in-conformation: %s", opt_info)

    cat_coord = None
    if chain_id in structure[0] and catalytic_residue_number is not None:
        for r in structure[0][chain_id]:
            if r.id[0] == " " and int(r.id[1]) == int(catalytic_residue_number):
                for atom_name in ("SG", "OG", "NZ", "OD1", "OE1", "CA"):
                    if atom_name in r:
                        cat_coord = np.array(r[atom_name].get_coord(), dtype=float)
                        break

    water_set = set(int(w) for w in (conserved_waters or []))
    if auto_retain_bridging_waters:
        bridging = detect_bridging_waters(structure, chain_id, keep_cofactors, cat_coord, max_waters=3)
        water_set.update(int(item["resseq"]) for item in bridging)

    cocrystal_info = extract_cocrystal_ligand(
        input_pdb,
        output_dir / "cocrystal_ligand.pdb",
        chain_id=chain_id,
        keep_cofactors=keep_cofactors,
        catalytic_coord=cat_coord,
    )
    (output_dir / "cocrystal_ligand_manifest.json").write_text(json.dumps(cocrystal_info, indent=2), encoding="utf-8")

    io = PDBIO()
    io.set_structure(structure)
    io.save(
        str(cleaned_pdb),
        ReceptorSelect(
            chain_id=chain_id,
            keep_cofactors=keep_cofactors,
            cofactor_chain=resolved_cof_chain,
            conserved_waters=water_set,
            catalytic_coord=cat_coord,
        ),
    )

    pdbqt_path = output_dir / "receptor.pdbqt"
    try:
        obabel = require_executable("obabel")
        command = [
            obabel, "-ipdb", str(cleaned_pdb), "-opdbqt", "-O", str(pdbqt_path),
            "-xr", "--partialcharge", "gasteiger",
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(f"OpenBabel failed:\n{result.stdout}\n{result.stderr}")
    except FileNotFoundError:
        LOGGER.warning("OpenBabel executable not found in PATH; using built-in rigid receptor PDBQT writer.")
        _write_receptor_pdbqt_fallback(cleaned_pdb, pdbqt_path)
    return cleaned_pdb, pdbqt_path


def _write_receptor_pdbqt_fallback(cleaned_pdb: Path, pdbqt_path: Path) -> Path:
    lines_out = []
    for line in cleaned_pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            element = (line[76:78].strip() if len(line) >= 78 else "") or line[12:16].strip()[:1]
            element = element.upper()
            resname = line[17:20].strip()
            aname = line[12:16].strip()
            ad_type = {"O": "OA", "N": "NA", "S": "SA"}.get(element, element)
            charge = 0.0
            if resname in {"ASP", "GLU"} and aname in {"OD1", "OD2", "OE1", "OE2"}:
                charge = -0.500
            elif resname in {"LYS", "ARG"} and aname in {"NZ", "NH1", "NH2"}:
                charge = +0.500
            elif resname == "CYS" and aname == "SG":
                charge = -0.450
            elif resname in {"ZN", "MG", "FE", "MN", "CA", "CU", "NI", "CO"}:
                charge = +2.000
            elif element == "O":
                charge = -0.350
            elif element == "N":
                charge = -0.250
            base = line[:66].ljust(66)
            lines_out.append(f"{base}    {charge:+6.3f} {ad_type:<2s}")
    lines_out.append("TER")
    pdbqt_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return pdbqt_path
