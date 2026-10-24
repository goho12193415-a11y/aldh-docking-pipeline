from __future__ import annotations

import itertools
import logging
import math
import re
import shutil
import site
import sys
import sysconfig
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

ONE_LETTER_AA = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U",
}

ATOMIC_WEIGHTS = {
    "H": 1.008, "LI": 6.94, "BE": 9.012, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "NA": 22.990, "MG": 24.305,
    "AL": 26.982, "SI": 28.085, "P": 30.974, "S": 32.06, "CL": 35.45,
    "K": 39.098, "CA": 40.078, "ZN": 65.38, "FE": 55.845, "CU": 63.546,
    "MN": 54.938, "CO": 58.933, "NI": 58.693, "BR": 79.904, "I": 126.904,
    "PT": 195.084, "AU": 196.967, "HG": 200.592, "PB": 207.2,
}

DEFAULT_VALENCE = {
    "C": 4, "N": 3, "O": 2, "S": 2, "P": 3, "B": 3, "F": 1, "CL": 1, "BR": 1, "I": 1,
}

# Gasteiger-Marsili PEOE parameters (a_i, b_i, c_i) in eV for orbital electronegativity chi = a + b*q + c*q^2
PEOE_PARAMS = {
    "H": (7.17, 6.24, -0.56),
    "C_sp3": (7.98, 9.18, 1.88),
    "C_sp2": (8.79, 9.32, 1.51),
    "N_sp3": (11.54, 10.82, 1.36),
    "N_sp2": (12.87, 11.15, 0.85),
    "O_sp3": (14.18, 12.92, 1.39),
    "O_sp2": (17.07, 13.79, 0.47),
    "S": (10.14, 9.13, 1.38),
    "P": (8.90, 8.24, 0.96),
    "F": (14.66, 13.85, 2.31),
    "CL": (11.00, 9.69, 1.35),
    "BR": (10.08, 8.47, 1.16),
    "I": (9.90, 7.96, 0.96),
}

MODEL_PKA = {
    "ASP": 3.80,
    "GLU": 4.50,
    "HIS": 6.50,
    "CYS": 8.30,
    "TYR": 10.00,
    "LYS": 10.50,
    "ARG": 12.50,
}

try:
    from rdkit import Chem as _rdkit_Chem  # type: ignore
    HAS_RDKIT = True
except ImportError:
    _rdkit_Chem = None
    HAS_RDKIT = False

try:
    import meeko as _meeko_mod  # type: ignore
    HAS_MEEKO = True
except ImportError:
    _meeko_mod = None
    HAS_MEEKO = False


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def require_executable(name_or_path: str) -> str:
    resolved = shutil.which(name_or_path)
    if resolved:
        return resolved
    path = Path(name_or_path)
    if path.exists():
        return str(path)
    if not path.is_absolute():
        candidates = [
            Path(sysconfig.get_path("scripts")) / name_or_path,
            Path(site.getuserbase()) / "Scripts" / name_or_path,
            Path(sys.executable).parent / name_or_path,
            Path(sys.executable).parent / "Scripts" / name_or_path,
        ]
        candidates.extend(
            base / "Scripts" / name_or_path
            for base in Path(site.getuserbase()).glob("Python*")
        )
        for candidate in candidates:
            for actual in (candidate, candidate.with_suffix(".exe")):
                if actual.exists():
                    return str(actual)
    raise FileNotFoundError(
        f"Required executable was not found: {name_or_path}. "
        "Install the dependency or update pipeline_config.yaml."
    )


def atom_dict(line: str) -> dict[str, Any]:
    """Parse ATOM/HETATM line from either PDB or PDBQT (extracting real partial charge when present)."""
    element = line[76:78].strip() if len(line) >= 78 else ""
    charge: float | None = None
    if len(line) >= 76:
        q_field = line[66:76].strip()
        if q_field:
            try:
                charge = float(q_field)
            except ValueError:
                charge = None
    raw_name = line[12:16].strip()
    inferred_el = element or (raw_name[:1] if raw_name else "C")
    if inferred_el.upper() in {"OA", "NA", "SA", "HD"}:
        inferred_el = {"OA": "O", "NA": "N", "SA": "S", "HD": "H"}[inferred_el.upper()]
    return {
        "record": line[0:6].strip(),
        "serial": int(line[6:11].strip()),
        "name": raw_name,
        "resname": line[17:20].strip(),
        "chain": line[21].strip(),
        "resseq": int(line[22:26].strip()),
        "coord": np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=float,
        ),
        "charge": charge,
        "element": inferred_el,
        "line": line.rstrip("\n"),
    }


def compute_peoe_charges(atoms: list[dict], adj: dict[int, list[tuple[int, int]]], iterations: int = 6) -> list[float]:
    """Iterative Gasteiger-Marsili Partial Equalization of Orbital Electronegativity (PEOE) topological charge solver."""
    charges = [float(a.get("charge", 0.0)) for a in atoms]
    params = []
    for a in atoms:
        idx = a["index"]
        sym = a["symbol"].upper()
        max_order = max((order for _, order in adj.get(idx, [])), default=1)
        is_sp2 = a.get("aromatic", False) or max_order >= 2
        if sym in {"C", "N", "O"}:
            key = f"{sym}_{'sp2' if is_sp2 else 'sp3'}"
            params.append(PEOE_PARAMS[key])
        else:
            params.append(PEOE_PARAMS.get(sym, (8.5, 8.5, 1.0)))

    damping = 0.5
    for _ in range(iterations):
        chi = [
            a_p + b_p * q + c_p * q * q
            for (a_p, b_p, c_p), q in zip(params, charges)
        ]
        dq = [0.0] * len(atoms)
        for u, neighbors in adj.items():
            for v, _ in neighbors:
                if u >= v:
                    continue
                if chi[u] > chi[v]:
                    denom = sum(params[v]) if sum(params[v]) > 1e-3 else 20.0
                    shift = damping * (chi[u] - chi[v]) / denom
                else:
                    denom = sum(params[u]) if sum(params[u]) > 1e-3 else 20.0
                    shift = damping * (chi[u] - chi[v]) / denom
                dq[u] -= shift
                dq[v] += shift
        for idx in range(len(charges)):
            charges[idx] = round(charges[idx] + dq[idx], 4)
        damping *= 0.5
    return charges


class FallbackAtom:
    def __init__(
        self,
        name: str,
        coord: np.ndarray,
        bfactor: float = 0.0,
        occupancy: float = 1.0,
        altloc: str = " ",
        fullname: str | None = None,
        serial_number: int = 1,
        element: str = "",
        record: str = "ATOM",
    ) -> None:
        self.name = name.strip()
        self.fullname = fullname if fullname is not None else f" {self.name:<3s}"[:4]
        self.coord = np.array(coord, dtype=float)
        self.bfactor = float(bfactor)
        self.occupancy = float(occupancy)
        self.altloc = altloc if altloc else " "
        self.serial_number = int(serial_number)
        self.element = (element or self.name[:1]).strip().upper()
        self.record = record.strip() or "ATOM"
        self.parent: FallbackResidue | None = None

    def get_name(self) -> str:
        return self.name

    def get_coord(self) -> np.ndarray:
        return self.coord

    def get_bfactor(self) -> float:
        return self.bfactor

    def get_occupancy(self) -> float:
        return self.occupancy

    def get_altloc(self) -> str:
        return self.altloc

    def set_altloc(self, value: str) -> None:
        self.altloc = value if value else " "

    def set_occupancy(self, value: float) -> None:
        self.occupancy = float(value)

    def get_parent(self) -> FallbackResidue | None:
        return self.parent

    def transform(self, rot: np.ndarray, tran: np.ndarray) -> None:
        self.coord = np.dot(self.coord, rot) + tran

    def copy(self) -> FallbackAtom:
        return FallbackAtom(
            name=self.name,
            coord=self.coord.copy(),
            bfactor=self.bfactor,
            occupancy=self.occupancy,
            altloc=self.altloc,
            fullname=self.fullname,
            serial_number=self.serial_number,
            element=self.element,
            record=self.record,
        )


class FallbackResidue:
    def __init__(self, res_id: tuple[str, int, str], resname: str, segid: str = "    ") -> None:
        self.id = (str(res_id[0]), int(res_id[1]), str(res_id[2]))
        self.resname = resname.strip()
        self.segid = segid
        self._atoms: list[FallbackAtom] = []
        self.parent: FallbackChain | None = None

    def add(self, atom: FallbackAtom) -> None:
        atom.parent = self
        self._atoms.append(atom)

    def get_resname(self) -> str:
        return self.resname

    def get_parent(self) -> FallbackChain | None:
        return self.parent

    def get_atoms(self) -> Iterator[FallbackAtom]:
        return iter(self._atoms)

    def get_unpacked_list(self) -> list[FallbackAtom]:
        return list(self._atoms)

    def __iter__(self) -> Iterator[FallbackAtom]:
        return iter(self._atoms)

    def __contains__(self, atom_name: str) -> bool:
        target = atom_name.strip()
        return any(atom.name == target for atom in self._atoms)

    def __getitem__(self, atom_name: str) -> FallbackAtom:
        target = atom_name.strip()
        selected = [atom for atom in self._atoms if atom.name == target]
        if not selected:
            raise KeyError(atom_name)
        selected.sort(
            key=lambda a: (
                0 if a.altloc in {" ", "", "A"} else 1,
                -a.occupancy,
                a.altloc,
            )
        )
        return selected[0]

    def copy(self) -> FallbackResidue:
        cloned = FallbackResidue(self.id, self.resname, self.segid)
        for atom in self._atoms:
            cloned.add(atom.copy())
        return cloned


class FallbackChain:
    def __init__(self, chain_id: str) -> None:
        self.id = chain_id
        self._residues: list[FallbackResidue] = []
        self._index: dict[tuple[str, int, str], FallbackResidue] = {}
        self.parent: FallbackModel | None = None

    def add(self, residue: FallbackResidue) -> None:
        residue.parent = self
        if residue.id in self._index:
            existing = self._index[residue.id]
            for atom in residue:
                existing.add(atom)
            return
        self._residues.append(residue)
        self._index[residue.id] = residue

    def get_atoms(self) -> Iterator[FallbackAtom]:
        for residue in self._residues:
            for atom in residue:
                yield atom

    def get_residues(self) -> Iterator[FallbackResidue]:
        return iter(self._residues)

    def __iter__(self) -> Iterator[FallbackResidue]:
        return iter(self._residues)

    def __contains__(self, res_id: tuple[str, int, str] | int) -> bool:
        if isinstance(res_id, int):
            return (" ", res_id, " ") in self._index
        return res_id in self._index

    def __getitem__(self, res_id: tuple[str, int, str] | int) -> FallbackResidue:
        if isinstance(res_id, int):
            res_id = (" ", res_id, " ")
        if res_id not in self._index:
            raise KeyError(res_id)
        return self._index[res_id]


class FallbackModel:
    def __init__(self, serial: int = 0) -> None:
        self.id = serial
        self._chains: list[FallbackChain] = []
        self._index: dict[str, FallbackChain] = {}
        self.parent: FallbackStructure | None = None

    def add(self, chain: FallbackChain) -> None:
        chain.parent = self
        self._chains.append(chain)
        self._index[chain.id] = chain

    def get_chains(self) -> Iterator[FallbackChain]:
        return iter(self._chains)

    def get_atoms(self) -> Iterator[FallbackAtom]:
        for chain in self._chains:
            yield from chain.get_atoms()

    def __iter__(self) -> Iterator[FallbackChain]:
        return iter(self._chains)

    def __contains__(self, chain_id: str) -> bool:
        return chain_id in self._index

    def __getitem__(self, chain_id: str) -> FallbackChain:
        if chain_id not in self._index:
            raise KeyError(chain_id)
        return self._index[chain_id]


class FallbackStructure:
    def __init__(self, structure_id: str) -> None:
        self.id = structure_id
        self._models: list[FallbackModel] = []

    def add(self, model: FallbackModel) -> None:
        model.parent = self
        self._models.append(model)

    def get_atoms(self) -> Iterator[FallbackAtom]:
        for model in self._models:
            yield from model.get_atoms()

    def __iter__(self) -> Iterator[FallbackModel]:
        return iter(self._models)

    def __getitem__(self, index: int) -> FallbackModel:
        return self._models[index]


class FallbackSelect:
    def accept_model(self, model) -> int:
        return 1

    def accept_chain(self, chain) -> int:
        return 1

    def accept_residue(self, residue) -> int:
        return 1

    def accept_atom(self, atom) -> int:
        return 1


class FallbackPDBParser:
    def __init__(self, QUIET: bool = True, PERMISSIVE: bool = True) -> None:
        self.quiet = QUIET
        self.permissive = PERMISSIVE

    def get_structure(self, structure_id: str, file_path: str | Path) -> FallbackStructure:
        path = Path(file_path)
        if path.suffix.lower() in {".cif", ".mmcif"}:
            path = convert_cif_to_pdb(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        structure = FallbackStructure(structure_id)
        current_model: FallbackModel | None = None
        model_serial = 0

        for line in text.splitlines():
            record = line[0:6].strip()
            if record == "MODEL":
                current_model = FallbackModel(model_serial)
                structure.add(current_model)
                model_serial += 1
            elif record in {"ATOM", "HETATM"}:
                if current_model is None:
                    current_model = FallbackModel(0)
                    structure.add(current_model)
                try:
                    serial = int(line[6:11].strip() or "1")
                except ValueError:
                    serial = 1
                fullname = line[12:16]
                name = fullname.strip()
                altloc = line[16:17] if len(line) > 16 else " "
                resname = line[17:20].strip()
                chain_id = line[21:22].strip() or "A"
                try:
                    resseq = int(line[22:26].strip())
                except ValueError:
                    continue
                icode = line[26:27] if len(line) > 26 else " "
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                try:
                    occupancy = float(line[54:60].strip()) if len(line) >= 60 and line[54:60].strip() else 1.0
                except ValueError:
                    occupancy = 1.0
                try:
                    bfactor = float(line[60:66].strip()) if len(line) >= 66 and line[60:66].strip() else 0.0
                except ValueError:
                    bfactor = 0.0
                element = line[76:78].strip() if len(line) >= 78 else ""
                if not element:
                    stripped = "".join(ch for ch in name if ch.isalpha())
                    element = stripped[:1].upper() if stripped else "C"
                hetflag = " " if record == "ATOM" else ("W" if resname in {"HOH", "WAT"} else f"H_{resname}")
                res_id = (hetflag, resseq, icode if icode else " ")

                if chain_id not in current_model:
                    current_model.add(FallbackChain(chain_id))
                chain = current_model[chain_id]
                if res_id not in chain:
                    chain.add(FallbackResidue(res_id, resname))
                residue = chain[res_id]
                residue.add(
                    FallbackAtom(
                        name=name,
                        coord=np.array([x, y, z], dtype=float),
                        bfactor=bfactor,
                        occupancy=occupancy,
                        altloc=altloc,
                        fullname=fullname,
                        serial_number=serial,
                        element=element,
                        record=record,
                    )
                )
        if not structure._models:
            structure.add(FallbackModel(0))
        return structure


class FallbackPDBIO:
    def __init__(self) -> None:
        self.structure: FallbackStructure | None = None

    def set_structure(self, structure: FallbackStructure) -> None:
        self.structure = structure

    def save(self, file_path: str | Path, select: FallbackSelect | None = None) -> None:
        if self.structure is None:
            raise ValueError("No structure set on PDBIO.")
        selector = select or FallbackSelect()
        lines: list[str] = []
        serial = 1
        for model in self.structure:
            if hasattr(selector, "accept_model") and not selector.accept_model(model):
                continue
            for chain in model:
                if hasattr(selector, "accept_chain") and not selector.accept_chain(chain):
                    continue
                for residue in chain:
                    if hasattr(selector, "accept_residue") and not selector.accept_residue(residue):
                        continue
                    record = "ATOM" if residue.id[0] == " " else "HETATM"
                    for atom in residue:
                        if hasattr(selector, "accept_atom") and not selector.accept_atom(atom):
                            continue
                        atom_name = atom.get_name()
                        element = (getattr(atom, "element", "") or atom_name[:1]).strip().upper()
                        if len(atom_name) < 4 and len(element) == 1:
                            formatted_name = f" {atom_name:<3s}"
                        else:
                            formatted_name = f"{atom_name:<4s}"[:4]
                        altloc = atom.get_altloc() if atom.get_altloc() else " "
                        icode = residue.id[2] if len(residue.id) > 2 and residue.id[2] else " "
                        x, y, z = [float(v) for v in atom.get_coord()]
                        occ = float(atom.get_occupancy()) if atom.get_occupancy() is not None else 1.0
                        bfac = float(atom.get_bfactor()) if atom.get_bfactor() is not None else 0.0
                        lines.append(
                            f"{record:<6s}{serial:5d} {formatted_name}{altloc:1s}{residue.get_resname():>3s} "
                            f"{chain.id:1s}{int(residue.id[1]):4d}{icode:1s}   "
                            f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}          {element:>2s}  "
                        )
                        serial += 1
        lines.append("END")
        Path(file_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class FallbackSuperimposer:
    def __init__(self) -> None:
        self.rotran: tuple[np.ndarray, np.ndarray] | None = None
        self.rms: float = 0.0

    def set_atoms(self, fixed: Iterable[FallbackAtom], moving: Iterable[FallbackAtom]) -> None:
        fixed_coords = np.array([atom.get_coord() for atom in fixed], dtype=float)
        moving_coords = np.array([atom.get_coord() for atom in moving], dtype=float)
        if len(fixed_coords) != len(moving_coords) or len(fixed_coords) == 0:
            raise ValueError("Fixed and moving atom lists must have the same non-zero length.")
        fixed_center = np.mean(fixed_coords, axis=0)
        moving_center = np.mean(moving_coords, axis=0)
        f_centered = fixed_coords - fixed_center
        m_centered = moving_coords - moving_center
        cov = np.dot(np.transpose(m_centered), f_centered)
        u, _, vt = np.linalg.svd(cov)
        d = np.sign(np.linalg.det(np.dot(u, vt)))
        diag = np.diag([1.0, 1.0, d])
        rot = np.dot(u, np.dot(diag, vt))
        tran = fixed_center - np.dot(moving_center, rot)
        transformed = np.dot(moving_coords, rot) + tran
        diff = fixed_coords - transformed
        self.rms = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        self.rotran = (rot, tran)

    def apply(self, entity) -> None:
        if self.rotran is None:
            raise RuntimeError("Superimposer.set_atoms must be called before apply.")
        rot, tran = self.rotran
        if hasattr(entity, "get_atoms"):
            for atom in entity.get_atoms():
                atom.transform(rot, tran)
        elif hasattr(entity, "__iter__"):
            for atom in entity:
                atom.transform(rot, tran)
        elif hasattr(entity, "transform"):
            entity.transform(rot, tran)


try:
    from Bio.PDB import PDBIO, PDBParser, Select, Superimposer  # type: ignore
    USING_BIOPYTHON = True
except ImportError:
    PDBParser = FallbackPDBParser  # type: ignore
    PDBIO = FallbackPDBIO  # type: ignore
    Select = FallbackSelect  # type: ignore
    Superimposer = FallbackSuperimposer  # type: ignore
    USING_BIOPYTHON = False


def align_ca_residues(target_residues: list, template_residues: list) -> list[tuple[object, object]]:
    """Align two lists of standard amino-acid residues with CA atoms using Needleman-Wunsch."""
    target_seq = [ONE_LETTER_AA.get(r.get_resname().strip(), "X") for r in target_residues]
    template_seq = [ONE_LETTER_AA.get(r.get_resname().strip(), "X") for r in template_residues]
    n, m = len(target_seq), len(template_seq)
    if n == 0 or m == 0:
        return []
    match_score = 3
    mismatch_score = -1
    gap_penalty = -2
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        dp[i, 0] = i * gap_penalty
    for j in range(1, m + 1):
        dp[0, j] = j * gap_penalty
    for i in range(1, n + 1):
        a = target_seq[i - 1]
        for j in range(1, m + 1):
            b = template_seq[j - 1]
            score = match_score if a == b and a != "X" else mismatch_score
            dp[i, j] = max(
                dp[i - 1, j - 1] + score,
                dp[i - 1, j] + gap_penalty,
                dp[i, j - 1] + gap_penalty,
            )
    i, j = n, m
    pairs = []
    while i > 0 and j > 0:
        a, b = target_seq[i - 1], template_seq[j - 1]
        score = match_score if a == b and a != "X" else mismatch_score
        if dp[i, j] == dp[i - 1, j - 1] + score:
            pairs.append((target_residues[i - 1], template_residues[j - 1]))
            i -= 1
            j -= 1
        elif dp[i, j] == dp[i - 1, j] + gap_penalty:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    identical_or_aligned = [
        (r1, r2) for r1, r2 in pairs
        if ONE_LETTER_AA.get(r1.get_resname().strip(), "X") == ONE_LETTER_AA.get(r2.get_resname().strip(), "Y")
    ]
    if len(identical_or_aligned) >= 10:
        return identical_or_aligned
    return pairs


def estimate_receptor_microenvironment_pka(
    pdb_path: Path,
    chain_id: str = "A",
    ph: float = 7.4,
    catalytic_resseq: int | None = None,
) -> dict:
    """Estimate 3D microenvironment pKa shifts (PROPKA-inspired empirical model) for active-site ionizable residues."""
    structure = PDBParser(QUIET=True).get_structure("pka_audit", str(pdb_path))
    if chain_id not in structure[0]:
        chain_id = next(iter(structure[0])).id
    chain = structure[0][chain_id]

    all_heavy = []
    for model in structure:
        for ch in model:
            for r in ch:
                if r.get_resname().strip() in {"HOH", "WAT"}:
                    continue
                for a in r:
                    elem = (getattr(a, "element", "") or a.get_name()[:1]).strip().upper()
                    if elem != "H":
                        all_heavy.append((ch.id, r.get_resname().strip(), int(r.id[1]), a.get_name().strip(), elem, np.array(a.get_coord(), dtype=float)))

    cat_coord = None
    if catalytic_resseq is not None and (" ", int(catalytic_resseq), " ") in chain:
        cat_res = chain[(" ", int(catalytic_resseq), " ")]
        for aname in ("SG", "OG", "NZ", "OD1", "OE1", "CA"):
            if aname in cat_res:
                cat_coord = np.array(cat_res[aname].get_coord(), dtype=float)
                break

    key_atom_map = {
        "ASP": "OD1", "GLU": "OE1", "HIS": "NE2", "CYS": "SG", "TYR": "OH", "LYS": "NZ", "ARG": "CZ",
    }
    acidic_types = {"ASP", "GLU", "CYS", "TYR"}
    records = []

    for r in chain:
        if r.id[0] != " ":
            continue
        rname = r.get_resname().strip()
        if rname not in MODEL_PKA:
            continue
        resseq = int(r.id[1])
        target_aname = key_atom_map[rname]
        if target_aname not in r:
            continue
        coord = np.array(r[target_aname].get_coord(), dtype=float)
        if cat_coord is not None and distance(coord, cat_coord) > 12.0 and resseq != catalytic_resseq:
            continue

        neighbors_6a = [item for item in all_heavy if not (item[0] == chain_id and item[2] == resseq) and distance(item[5], coord) <= 6.0]
        burial_fraction = min(1.0, max(0.0, (len(neighbors_6a) - 12.0) / 28.0))

        hbond_neighbors = [
            item for item in neighbors_6a
            if item[4] in {"N", "O", "S"} and distance(item[5], coord) <= 3.6
        ]

        pos_neighbors = [
            item for item in neighbors_6a
            if (
                item[1] in {"NAD", "NAP", "NDP", "ZN", "MG", "FE", "MN", "CA", "CU", "LYS", "ARG", "HIS"}
                and distance(item[5], coord) <= 5.5
            )
        ]

        pka_model = MODEL_PKA[rname]
        delta_hbond = -0.55 * len(hbond_neighbors) if rname in acidic_types else +0.40 * len(hbond_neighbors)
        delta_elec = -0.65 * min(3, len(pos_neighbors)) if rname in acidic_types else -0.35 * min(2, len(pos_neighbors))
        if rname == "CYS" and resseq == catalytic_resseq:
            delta_elec -= 1.10
        delta_desolv = +0.80 * burial_fraction if rname in {"ASP", "GLU"} else (-0.40 * burial_fraction if rname == "CYS" else 0.0)
        pka_pred = round(pka_model + delta_hbond + delta_elec + delta_desolv, 2)

        if rname in acidic_types:
            deprot_frac = round(1.0 / (1.0 + 10.0 ** (pka_pred - ph)), 4)
            assigned_state = f"{rname}_deprotonated_anion" if deprot_frac >= 0.5 else f"{rname}_neutral"
            if rname == "CYS":
                assigned_state = "CYM_thiolate(-1)" if deprot_frac >= 0.5 else "CYS_thiol(0)"
        else:
            prot_frac = round(1.0 / (1.0 + 10.0 ** (ph - pka_pred)), 4)
            if rname == "HIS":
                if prot_frac >= 0.5:
                    assigned_state = "HIP_protonated(+1)"
                else:
                    nd1_hb = sum(1 for item in hbond_neighbors if "ND1" in r and distance(item[5], np.array(r["ND1"].get_coord(), dtype=float)) <= 3.5)
                    assigned_state = "HID_neutral" if nd1_hb > 0 else "HIE_neutral"
            else:
                assigned_state = f"{rname}_protonated(+1)" if prot_frac >= 0.5 else f"{rname}_neutral(0)"

        records.append({
            "chain": chain_id,
            "residue": rname,
            "number": resseq,
            "is_catalytic": resseq == catalytic_resseq,
            "model_pka": pka_model,
            "predicted_pka": pka_pred,
            "burial_fraction": round(burial_fraction, 3),
            "hbond_neighbor_count": len(hbond_neighbors),
            "electrostatic_stabilizer_count": len(pos_neighbors),
            "assigned_state_at_ph": assigned_state,
            "experimental_ph": float(ph),
        })

    return {
        "pdb": str(pdb_path),
        "chain": chain_id,
        "experimental_ph": float(ph),
        "active_site_residue_count": len(records),
        "residues": records,
    }


def parse_smiles_tokens(smiles: str) -> dict:
    """Parse a SMILES string into atoms, bonds, rings, charges, isotopes, and stereocenters."""
    s = (smiles or "").strip()
    if not s or any(ch.isspace() for ch in s):
        raise ValueError(f"Invalid SMILES: {smiles}")
    fragments = [frag for frag in s.split(".") if frag]
    if not fragments:
        raise ValueError(f"Invalid SMILES: {smiles}")

    atoms: list[dict] = []
    bonds: list[tuple[int, int, int, bool]] = []
    ring_openings: dict[str, tuple[int, int]] = {}
    branch_stack: list[int] = []
    prev_atom: int | None = None
    pending_bond_order = 1

    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == ".":
            prev_atom = None
            pending_bond_order = 1
            i += 1
            continue
        if ch == "(":
            if prev_atom is None:
                raise ValueError(f"Invalid branch opening in SMILES: {smiles}")
            branch_stack.append(prev_atom)
            i += 1
            continue
        if ch == ")":
            if not branch_stack:
                raise ValueError(f"Unbalanced branch parenthesis in SMILES: {smiles}")
            prev_atom = branch_stack.pop()
            i += 1
            continue
        if ch in "-=#:":
            pending_bond_order = {"-": 1, "=": 2, "#": 3, ":": 1}[ch]
            i += 1
            continue
        if ch in "/\\":
            i += 1
            continue
        if ch == "%" and i + 2 < n and s[i + 1:i + 3].isdigit():
            ring_id = s[i + 1:i + 3]
            if prev_atom is None:
                raise ValueError(f"Invalid ring closure in SMILES: {smiles}")
            if ring_id in ring_openings:
                open_atom, open_order = ring_openings.pop(ring_id)
                bonds.append((open_atom, prev_atom, max(open_order, pending_bond_order), True))
            else:
                ring_openings[ring_id] = (prev_atom, pending_bond_order)
            pending_bond_order = 1
            i += 3
            continue
        if ch.isdigit():
            ring_id = ch
            if prev_atom is None:
                raise ValueError(f"Invalid ring closure in SMILES: {smiles}")
            if ring_id in ring_openings:
                open_atom, open_order = ring_openings.pop(ring_id)
                bonds.append((open_atom, prev_atom, max(open_order, pending_bond_order), True))
            else:
                ring_openings[ring_id] = (prev_atom, pending_bond_order)
            pending_bond_order = 1
            i += 1
            continue
        if ch == "[":
            end = s.find("]", i)
            if end == -1:
                raise ValueError(f"Unclosed bracket in SMILES: {smiles}")
            body = s[i + 1:end]
            m = re.match(r"^(\d+)?([A-Za-z][a-z]?)(@@|@)?(H\d*)?([+-]\d*|[+-]{2,})?$", body)
            if not m:
                raise ValueError(f"Invalid bracket atom [{body}] in SMILES: {smiles}")
            isotope_str, sym, chiral, h_str, charge_str = m.groups()
            isotope = int(isotope_str) if isotope_str else 0
            aromatic = sym.islower()
            element = sym.upper()
            if element not in ATOMIC_WEIGHTS:
                raise ValueError(f"Unknown element {element} in SMILES: {smiles}")
            explicit_h = 0
            if h_str:
                explicit_h = int(h_str[1:]) if len(h_str) > 1 else 1
            charge = 0
            if charge_str:
                if charge_str in {"+", "-"}:
                    charge = 1 if charge_str == "+" else -1
                elif charge_str.startswith("+") and charge_str[1:].isdigit():
                    charge = int(charge_str[1:])
                elif charge_str.startswith("-") and charge_str[1:].isdigit():
                    charge = -int(charge_str[1:])
                elif set(charge_str) == {"+"}:
                    charge = len(charge_str)
                elif set(charge_str) == {"-"}:
                    charge = -len(charge_str)
            idx = len(atoms)
            atoms.append({
                "index": idx,
                "symbol": element,
                "aromatic": aromatic,
                "isotope": isotope,
                "chiral": chiral or "",
                "explicit_h": explicit_h,
                "charge": charge,
                "bracket": True,
                "radical": 0,
            })
            if prev_atom is not None:
                bonds.append((prev_atom, idx, pending_bond_order, False))
            prev_atom = idx
            pending_bond_order = 1
            i = end + 1
            continue
        if s[i:i + 2] in {"Cl", "Br"}:
            sym = s[i:i + 2]
            step = 2
        elif ch in "BCNOPSFIbcnosp":
            sym = ch
            step = 1
        else:
            raise ValueError(f"Unrecognized character '{ch}' in SMILES: {smiles}")
        aromatic = sym.islower()
        element = sym.upper()
        idx = len(atoms)
        atoms.append({
            "index": idx,
            "symbol": element,
            "aromatic": aromatic,
            "isotope": 0,
            "chiral": "",
            "explicit_h": None,
            "charge": 0,
            "bracket": False,
            "radical": 0,
        })
        if prev_atom is not None:
            order = 1 if not (aromatic and atoms[prev_atom]["aromatic"]) else 1
            bonds.append((prev_atom, idx, pending_bond_order if pending_bond_order > 1 else order, False))
        prev_atom = idx
        pending_bond_order = 1
        i += step

    if branch_stack or ring_openings:
        raise ValueError(f"Unclosed rings or branches in SMILES: {smiles}")
    if not atoms:
        raise ValueError(f"No atoms parsed from SMILES: {smiles}")

    adj: dict[int, list[tuple[int, int]]] = {a["index"]: [] for a in atoms}
    for u, v, order, _ in bonds:
        adj[u].append((v, order))
        adj[v].append((u, order))

    total_h = 0
    for a in atoms:
        idx = a["index"]
        bond_sum = sum(order for _, order in adj[idx]) + (1 if a["aromatic"] else 0)
        if a["explicit_h"] is not None:
            h_count = a["explicit_h"]
        else:
            val = DEFAULT_VALENCE.get(a["symbol"], 0)
            h_count = max(0, val - bond_sum + a["charge"])
        a["implicit_h"] = h_count
        total_h += h_count

    heavy_atoms = [a for a in atoms if a["symbol"] != "H"]
    ring_count = sum(1 for _, _, _, is_ring in bonds if is_ring)

    unassigned_chiral: list[int] = []
    assigned_chiral: list[int] = []
    for a in atoms:
        if a["chiral"] in {"@", "@@"}:
            assigned_chiral.append(a["index"])
        elif a["symbol"] == "C" and not a["aromatic"]:
            neighbors = adj[a["index"]]
            if all(order == 1 for _, order in neighbors) and (
                (len(neighbors) == 3 and a["implicit_h"] == 1)
                or (len(neighbors) == 4 and a["implicit_h"] == 0)
            ):
                branch_sigs = []
                for nb, _ in neighbors:
                    nb_atom = atoms[nb]
                    second_shell = tuple(
                        sorted(
                            (atoms[n2]["symbol"], int(o2), int(atoms[n2]["implicit_h"]))
                            for n2, o2 in adj[nb]
                            if n2 != a["index"]
                        )
                    )
                    branch_sigs.append((nb_atom["symbol"], int(nb_atom["implicit_h"]), int(nb_atom["charge"]), second_shell))
                if len(set(branch_sigs)) == len(branch_sigs):
                    unassigned_chiral.append(a["index"])
                elif len(neighbors) >= 3 and ring_count >= 2:
                    neighbor_syms = sorted(atoms[nb]["symbol"] + str(nb) for nb, _ in neighbors)
                    if len(set(neighbor_syms)) == len(neighbor_syms) and any(atoms[nb]["symbol"] == "O" for nb, _ in neighbors):
                        unassigned_chiral.append(a["index"])

    rotatable_bonds = 0
    for u, v, order, is_ring in bonds:
        if is_ring or order != 1:
            continue
        if atoms[u]["symbol"] == "H" or atoms[v]["symbol"] == "H":
            continue
        u_heavy_deg = sum(1 for nb, _ in adj[u] if atoms[nb]["symbol"] != "H")
        v_heavy_deg = sum(1 for nb, _ in adj[v] if atoms[nb]["symbol"] != "H")
        if u_heavy_deg > 1 and v_heavy_deg > 1:
            rotatable_bonds += 1

    mw = sum(ATOMIC_WEIGHTS.get(a["symbol"], 12.0) for a in atoms) + total_h * ATOMIC_WEIGHTS["H"]

    return {
        "smiles": s,
        "fragments": fragments,
        "atoms": atoms,
        "bonds": bonds,
        "adj": adj,
        "heavy_atom_count": len(heavy_atoms),
        "ring_count": ring_count,
        "rotatable_bond_count": rotatable_bonds,
        "formal_charge": sum(a["charge"] for a in atoms),
        "molecular_weight": round(mw, 3),
        "assigned_chiral": assigned_chiral,
        "unassigned_chiral": unassigned_chiral,
    }


def standardize_and_desalt_smiles(smiles: str) -> dict:
    """Automatically strip counterions/salts and return the largest organic parent fragment."""
    raw = (smiles or "").strip()
    parts = [p.strip() for p in raw.split(".") if p.strip()]
    if len(parts) <= 1:
        return {
            "original_smiles": raw,
            "desalted_smiles": raw,
            "stripped_counterions": [],
            "fragment_count": len(parts),
            "was_desalted": False,
        }

    if HAS_RDKIT:
        from rdkit import Chem
        scored_parts = []
        stripped = []
        for part in parts:
            mol = Chem.MolFromSmiles(part)
            if mol is None:
                continue
            heavy = mol.GetNumHeavyAtoms()
            has_carbon = any(a.GetSymbol() == "C" for a in mol.GetAtoms())
            scored_parts.append((int(has_carbon), heavy, part))
        if scored_parts:
            scored_parts.sort(key=lambda item: (item[0], item[1]), reverse=True)
            best = scored_parts[0][2]
            for _, _, p in scored_parts[1:]:
                stripped.append(p)
            return {
                "original_smiles": raw,
                "desalted_smiles": best,
                "stripped_counterions": stripped,
                "fragment_count": len(parts),
                "was_desalted": True,
            }

    scored_fallback = []
    for part in parts:
        parsed = parse_smiles_tokens(part)
        has_c = any(a["symbol"] == "C" for a in parsed["atoms"])
        scored_fallback.append((int(has_c), parsed["heavy_atom_count"], part))
    scored_fallback.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_part = scored_fallback[0][2]
    stripped_parts = [item[2] for item in scored_fallback[1:]]
    return {
        "original_smiles": raw,
        "desalted_smiles": best_part,
        "stripped_counterions": stripped_parts,
        "fragment_count": len(parts),
        "was_desalted": True,
    }


def _prot_penalty_kcal(fraction: float) -> float:
    f = max(1e-6, min(1.0, float(fraction)))
    return round(-0.592 * math.log(f), 3)


def enumerate_ph_protonation_states(smiles: str, ph: float = 7.4, max_states: int = 8) -> list[dict]:
    """Generate single-site and coupled multi-site (zwitterionic/multiprotic) protonation microstates with Boltzmann penalties."""
    desalted = standardize_and_desalt_smiles(smiles)["desalted_smiles"]
    states: list[dict] = [{
        "label": "parent",
        "smiles": desalted,
        "ph": float(ph),
        "rule": "input_or_desalted_parent",
        "estimated_fraction": 1.0,
        "protonation_penalty_kcal_mol": 0.0,
    }]
    seen = {desalted}

    # Build per-site transformation options for combinatorial coupled microstate enumeration
    site_transforms: list[list[tuple[str, str, str, float]]] = []

    has_ewg = any(hal in desalted for hal in ("F", "Cl", "Br", "[N+](=O)[O-]"))
    pka_cooh = 2.8 if has_ewg else 4.2
    frac_coo_neg = round(1.0 / (1.0 + 10 ** (pka_cooh - ph)), 4)
    frac_cooh_neu = round(1.0 - frac_coo_neg, 4)
    if "C(=O)O" in desalted and not desalted.endswith("C(=O)OC"):
        site_transforms.append([
            ("cooh_neutral", "C(=O)O", "C(=O)O", frac_cooh_neu),
            ("carboxylate_deprotonated", "C(=O)O", "C(=O)[O-]", frac_coo_neg),
        ])
    elif "C(=O)[O-]" in desalted:
        site_transforms.append([
            ("carboxylate_deprotonated", "C(=O)[O-]", "C(=O)[O-]", frac_coo_neg),
            ("carboxylic_acid_neutral", "C(=O)[O-]", "C(=O)O", frac_cooh_neu),
        ])

    frac_nh3_pos = round(1.0 / (1.0 + 10 ** (ph - 9.8)), 4)
    frac_nh2_neu = round(1.0 - frac_nh3_pos, 4)
    if "[NH3+]" in desalted:
        site_transforms.append([
            ("ammonium_protonated", "[NH3+]", "[NH3+]", frac_nh3_pos),
            ("amine_neutral_microstate", "[NH3+]", "N", frac_nh2_neu),
        ])
    elif desalted.startswith("NC") or desalted.endswith("CN"):
        target_tok = "NC" if desalted.startswith("NC") else "CN"
        repl_tok = "[NH3+]C" if desalted.startswith("NC") else "C[NH3+]"
        site_transforms.append([
            ("amine_neutral_microstate", target_tok, target_tok, frac_nh2_neu),
            ("ammonium_protonated", target_tok, repl_tok, frac_nh3_pos),
        ])

    if "C(=O)NO" in desalted:
        frac_hyd = round(1.0 / (1.0 + 10 ** (8.8 - ph)), 4)
        site_transforms.append([
            ("hydroxamic_neutral", "C(=O)NO", "C(=O)NO", round(1.0 - frac_hyd, 4)),
            ("hydroxamate_chelator_anion", "C(=O)NO", "C(=O)N[O-]", frac_hyd),
        ])

    if 6.5 <= ph <= 10.0 and ("CS" in desalted and not re.search(r"CS[C(]", desalted)):
        frac_thiolate = round(1.0 / (1.0 + 10 ** (8.3 - ph)), 4)
        site_transforms.append([
            ("thiol_neutral", "CS", "CS", round(1.0 - frac_thiolate, 4)),
            ("thiolate_microstate", "CS", "C[S-]", frac_thiolate),
        ])

    if site_transforms:
        for combo in itertools.product(*site_transforms):
            cand = desalted
            labels = []
            joint_prob = 1.0
            for lbl, src_pat, dst_pat, prob in combo:
                cand = cand.replace(src_pat, dst_pat, 1)
                labels.append(lbl)
                joint_prob *= prob
            joint_prob = round(joint_prob, 4)
            if cand not in seen and joint_prob >= 1e-4:
                try:
                    parse_smiles_tokens(cand)
                    seen.add(cand)
                    combined_label = "__".join(l for l in labels if not l.endswith("_neutral") or len(labels) == 1) or "neutral_microstate"
                    states.append({
                        "label": combined_label,
                        "smiles": cand,
                        "ph": float(ph),
                        "rule": "coupled_henderson_hasselbalch_microstate",
                        "estimated_fraction": joint_prob,
                        "protonation_penalty_kcal_mol": _prot_penalty_kcal(joint_prob),
                    })
                except ValueError:
                    pass

    if re.search(r"CC=O$", desalted):
        enol = re.sub(r"CC=O$", "C=CO", desalted)
        if enol not in seen:
            seen.add(enol)
            states.append({
                "label": "enol_tautomer_minor",
                "smiles": enol,
                "ph": float(ph),
                "rule": "carbonyl_enol_tautomer",
                "estimated_fraction": 0.01,
                "protonation_penalty_kcal_mol": _prot_penalty_kcal(0.01),
            })

    return states[:max_states]


def enumerate_stereoisomers_and_tautomers(smiles: str, max_isomers: int = 8) -> list[dict]:
    """Enumerate unassigned R/S tetrahedral stereocenters, E/Z alkenes, and prototropic tautomers (keto-enol, imidazole, amide-imidic)."""
    desalted = standardize_and_desalt_smiles(smiles)["desalted_smiles"]
    results: list[dict] = [{
        "label": "parent_isomer",
        "smiles": desalted,
        "category": "parent",
        "rule": "input_or_desalted_parent",
    }]
    seen = {desalted}

    # 1. Tautomer enumeration (RDKit + deterministic chemical tautomer transforms)
    if HAS_RDKIT:
        try:
            from rdkit import Chem
            from rdkit.Chem.MolStandardize import rdMolStandardize

            mol = Chem.MolFromSmiles(desalted)
            if mol is not None:
                enumerator = rdMolStandardize.TautomerEnumerator()
                for idx, taut in enumerate(enumerator.Enumerate(mol), 1):
                    cand = Chem.MolToSmiles(taut, isomericSmiles=True, canonical=True)
                    if cand not in seen and len(results) < max_isomers:
                        seen.add(cand)
                        results.append({
                            "label": f"tautomer_{idx}",
                            "smiles": cand,
                            "category": "tautomer",
                            "rule": "rdkit_tautomer_enumerator",
                        })
        except Exception:
            pass

    tautomer_rules = [
        ("keto_enol_tautomer", r"CC=O$", "C=CO"),
        ("keto_enol_internal", r"CC\(=O\)C", "C=C(O)C"),
        ("imidazole_1H_3H_tautomer", r"c1c\[nH\]cn1", "c1cnc[nH]1"),
        ("imidazole_3H_1H_tautomer", r"c1cnc\[nH\]1", "c1c[nH]cn1"),
        ("amide_imidic_tautomer", r"C\(=O\)N([A-Za-z0-9(\[\]]*)", r"C(O)=N"),
        ("thioamide_imidothiol_tautomer", r"C\(=S\)N", "C(S)=N"),
    ]
    for rule_name, pattern, replacement in tautomer_rules:
        if re.search(pattern, desalted):
            cand = re.sub(pattern, replacement, desalted, count=1)
            if cand not in seen and len(results) < max_isomers:
                try:
                    parse_smiles_tokens(cand)
                    seen.add(cand)
                    results.append({
                        "label": rule_name,
                        "smiles": cand,
                        "category": "tautomer",
                        "rule": rule_name,
                    })
                except ValueError:
                    pass

    # 2. Unassigned R/S chiral center & E/Z double-bond stereoisomer enumeration
    if HAS_RDKIT:
        try:
            from rdkit import Chem
            from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

            mol = Chem.MolFromSmiles(desalted)
            if mol is not None:
                opts = StereoEnumerationOptions(onlyUnassigned=True, maxIsomers=max_isomers, unique=True)
                for idx, iso in enumerate(EnumerateStereoisomers(mol, options=opts), 1):
                    cand = Chem.MolToSmiles(iso, isomericSmiles=True, canonical=True)
                    if cand not in seen and len(results) < max_isomers:
                        seen.add(cand)
                        results.append({
                            "label": f"stereoisomer_{idx}",
                            "smiles": cand,
                            "category": "stereoisomer",
                            "rule": "rdkit_unassigned_stereoisomer",
                        })
        except Exception:
            pass

    try:
        parsed = parse_smiles_tokens(desalted)
    except ValueError:
        return results[:max_isomers]

    if parsed["unassigned_chiral"]:
        # Locate unassigned tetrahedral carbon branch sites like C(X) and expand into [C@H](X) / [C@@H](X) or [C@](X) / [C@@](X)
        m = re.search(r"(?<!\[)C(\([^()]+\))", desalted)
        if m:
            prefix = desalted[:m.start()]
            branch_and_rest = desalted[m.start() + 1:]
            chiral_idx = parsed["unassigned_chiral"][0]
            imp_h = parsed["atoms"][chiral_idx].get("implicit_h", 1)
            h_tok = "H" if imp_h >= 1 else ""
            for tag, stereo_lbl in (("@", "stereoisomer_R_or_at"), ("@@", "stereoisomer_S_or_atat")):
                cand = f"{prefix}[C{tag}{h_tok}]{branch_and_rest}"
                if cand not in seen and len(results) < max_isomers:
                    try:
                        parse_smiles_tokens(cand)
                        seen.add(cand)
                        results.append({
                            "label": stereo_lbl,
                            "smiles": cand,
                            "category": "stereoisomer",
                            "rule": "tetrahedral_chiral_expansion",
                        })
                    except ValueError:
                        pass

    return results[:max_isomers]


def generate_fallback_3d_coords(parsed: dict, seed: int = 0xA1D0, conformer_index: int = 0) -> list[dict]:
    """Generate realistic 3D atomic coordinates with Gasteiger-Marsili PEOE partial charges when RDKit is unavailable."""
    rng = np.random.default_rng(seed + conformer_index * 101)
    atoms = parsed["atoms"]
    adj = parsed["adj"]
    peoe_q = compute_peoe_charges(atoms, adj, iterations=6)
    coords: dict[int, np.ndarray] = {}

    for start_idx in range(len(atoms)):
        if start_idx in coords:
            continue
        coords[start_idx] = np.array([0.0, 0.0, 0.0], dtype=float) + rng.normal(0.0, 0.02, size=3)
        queue = [start_idx]
        step_counter = 0
        while queue:
            u = queue.pop(0)
            u_pos = coords[u]
            neighbors = adj[u]
            for branch_idx, (v, order) in enumerate(neighbors):
                if v in coords:
                    continue
                step_counter += 1
                u_sym = atoms[u]["symbol"]
                v_sym = atoms[v]["symbol"]
                if {u_sym, v_sym} == {"C", "O"} and order == 2:
                    bond_len = 1.225
                elif order == 2:
                    bond_len = 1.34
                elif order == 3:
                    bond_len = 1.20
                else:
                    bond_len = 1.52 if {u_sym, v_sym} == {"C"} else 1.43

                chiral_tag = atoms[u].get("chiral", "")
                chiral_sign = -1.0 if chiral_tag == "@@" else 1.0
                torsion = (step_counter * 1.0472) + (branch_idx * 2.0944 * chiral_sign) + (conformer_index * 0.65)
                tilt = 0.615 if order == 1 else 0.523
                direction = np.array([
                    math.cos(tilt),
                    math.sin(tilt) * math.cos(torsion),
                    chiral_sign * math.sin(tilt) * math.sin(torsion),
                ], dtype=float)
                direction /= np.linalg.norm(direction)
                coords[v] = u_pos + direction * bond_len
                queue.append(v)

    atom_records = []
    for idx, a in enumerate(atoms):
        pos = coords.get(idx, np.zeros(3, dtype=float))
        sym = a["symbol"]
        if sym == "O":
            ad_type = "OA"
        elif sym == "N":
            ad_type = "NA" if a["implicit_h"] == 0 else "N"
        elif sym == "S":
            ad_type = "SA" if a["implicit_h"] == 0 else "S"
        elif sym == "C" and a["aromatic"]:
            ad_type = "A"
        else:
            ad_type = sym
        atom_records.append({
            "serial": idx + 1,
            "name": f"{sym}{idx + 1}"[:4],
            "symbol": sym,
            "ad_type": ad_type,
            "coord": pos,
            "charge": float(peoe_q[idx]),
        })
    return atom_records


def write_fallback_sdf(name: str, atom_records: list[dict], bonds: list[tuple[int, int, int, bool]], sdf_path: Path) -> Path:
    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        name,
        "  ALDH_Docking_Pipeline_3D",
        "",
        f"{len(atom_records):3d}{len(bonds):3d}  0  0  0  0  0  0  0  0999 V2000",
    ]
    for rec in atom_records:
        x, y, z = rec["coord"]
        lines.append(f"{x:10.4f}{y:10.4f}{z:10.4f} {rec['symbol']:<3s} 0  0  0  0  0  0  0  0  0  0  0  0")
    for u, v, order, _ in bonds:
        lines.append(f"{u + 1:3d}{v + 1:3d}{order:3d}  0  0  0  0")
    lines.append("M  END")
    lines.append("$$$$")
    sdf_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sdf_path


def _format_pdbqt_hetatm(rec: dict, serial_num: int) -> str:
    x, y, z = rec["coord"]
    return (
        f"HETATM{serial_num:5d} {rec['name']:>4s} UNL     1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    {rec['charge']:+6.3f} {rec['ad_type']:<2s}"
    )


def write_fallback_pdbqt(
    name: str,
    atom_records: list[dict],
    pdbqt_path: Path,
    bonds: list[tuple[int, int, int, bool]] | None = None,
) -> Path:
    """Compile a valid AutoDock Vina PDBQT with a spanning-tree ROOT + nested BRANCH/ENDBRANCH torsion tree and accurate TORSDOF count."""
    pdbqt_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(atom_records)
    if n == 0:
        pdbqt_path.write_text(f"REMARK  Name = {name}\nROOT\nENDROOT\nTORSDOF 0\n", encoding="utf-8")
        return pdbqt_path

    if bonds is None:
        inferred: list[tuple[int, int, int, bool]] = []
        adj_tmp: dict[int, list[int]] = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                d = distance(np.array(atom_records[i]["coord"], dtype=float), np.array(atom_records[j]["coord"], dtype=float))
                if 0.85 <= d <= 1.88:
                    order = 2 if d < 1.38 else 1
                    inferred.append((i, j, order, False))
                    adj_tmp[i].append(j)
                    adj_tmp[j].append(i)
        bonds = []
        for u, v, order, _ in inferred:
            visited = {u}
            queue = [u]
            in_ring = False
            while queue:
                curr = queue.pop(0)
                if curr == v:
                    in_ring = True
                    break
                for nb in adj_tmp[curr]:
                    if (curr == u and nb == v) or (curr == v and nb == u):
                        continue
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            bonds.append((u, v, order, in_ring))

    adj: dict[int, list[tuple[int, int, bool]]] = {i: [] for i in range(n)}
    for u, v, order, is_ring in bonds:
        if 0 <= u < n and 0 <= v < n:
            adj[u].append((v, order, is_ring))
            adj[v].append((u, order, is_ring))

    rotatable_edges: set[tuple[int, int]] = set()
    for u, v, order, is_ring in bonds:
        if is_ring or order != 1 or not (0 <= u < n and 0 <= v < n):
            continue
        if atom_records[u]["symbol"].upper() == "H" or atom_records[v]["symbol"].upper() == "H":
            continue
        deg_u = sum(1 for nb, _, _ in adj[u] if atom_records[nb]["symbol"].upper() != "H")
        deg_v = sum(1 for nb, _, _ in adj[v] if atom_records[nb]["symbol"].upper() != "H")
        if deg_u > 1 and deg_v > 1:
            vis = {u}
            q = [u]
            while q:
                cur = q.pop(0)
                for nb, _, _ in adj[cur]:
                    if (cur == u and nb == v) or (cur == v and nb == u):
                        continue
                    if nb not in vis:
                        vis.add(nb)
                        q.append(nb)
            if v not in vis:
                rotatable_edges.add((min(u, v), max(u, v)))

    frag_id_of: dict[int, int] = {}
    fragments: list[list[int]] = []
    for i in range(n):
        if i in frag_id_of:
            continue
        fid = len(fragments)
        comp = []
        q = [i]
        frag_id_of[i] = fid
        while q:
            cur = q.pop(0)
            comp.append(cur)
            for nb, _, _ in adj[cur]:
                if (min(cur, nb), max(cur, nb)) in rotatable_edges:
                    continue
                if nb not in frag_id_of:
                    frag_id_of[nb] = fid
                    q.append(nb)
        fragments.append(comp)

    root_fid = max(range(len(fragments)), key=lambda fid: (len(fragments[fid]), -min(fragments[fid])))

    frag_adj: dict[int, list[tuple[int, int, int]]] = {fid: [] for fid in range(len(fragments))}
    for u, v in sorted(rotatable_edges):
        fu, fv = frag_id_of[u], frag_id_of[v]
        if fu != fv:
            frag_adj[fu].append((fv, u, v))
            frag_adj[fv].append((fu, v, u))

    serial_map: dict[int, int] = {}
    next_serial = 1

    def _assign_serials(fid: int, parent_fid: int | None, entry_atom: int | None) -> None:
        nonlocal next_serial
        atoms_in_frag = list(fragments[fid])
        if entry_atom is not None and entry_atom in atoms_in_frag:
            atoms_in_frag.remove(entry_atom)
            atoms_in_frag.insert(0, entry_atom)
        for a_idx in atoms_in_frag:
            serial_map[a_idx] = next_serial
            next_serial += 1
        for child_fid, u_atom, v_atom in frag_adj[fid]:
            if child_fid == parent_fid:
                continue
            _assign_serials(child_fid, fid, v_atom)

    _assign_serials(root_fid, None, None)
    for i in range(n):
        if i not in serial_map:
            serial_map[i] = next_serial
            next_serial += 1

    lines: list[str] = [f"REMARK  Name = {name}", "ROOT"]
    for a_idx in fragments[root_fid]:
        lines.append(_format_pdbqt_hetatm(atom_records[a_idx], serial_map[a_idx]))
    for i in range(n):
        if frag_id_of[i] != root_fid and not frag_adj[frag_id_of[i]]:
            lines.append(_format_pdbqt_hetatm(atom_records[i], serial_map[i]))
    lines.append("ENDROOT")

    emitted_branches = 0

    def _emit_branches(fid: int, parent_fid: int | None) -> None:
        nonlocal emitted_branches
        for child_fid, u_atom, v_atom in frag_adj[fid]:
            if child_fid == parent_fid:
                continue
            emitted_branches += 1
            su = serial_map[u_atom]
            sv = serial_map[v_atom]
            lines.append(f"BRANCH {su:3d} {sv:3d}")
            atoms_in_child = list(fragments[child_fid])
            if v_atom in atoms_in_child:
                atoms_in_child.remove(v_atom)
                atoms_in_child.insert(0, v_atom)
            for a_idx in atoms_in_child:
                lines.append(_format_pdbqt_hetatm(atom_records[a_idx], serial_map[a_idx]))
            _emit_branches(child_fid, fid)
            lines.append(f"ENDBRANCH {su:3d} {sv:3d}")

    _emit_branches(root_fid, None)
    lines.append(f"TORSDOF {emitted_branches}")
    pdbqt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pdbqt_path


def parse_mol2_block(mol2_text: str, default_name: str = "Ligand", auto_desalt: bool = True) -> dict:
    """Parse a single Tripos @<TRIPOS>MOLECULE block into heavy-atom 3D records, bonds, SYBYL types, and partial charges."""
    lines = mol2_text.splitlines()
    section = ""
    mol_name = default_name
    mol_line_idx = 0
    raw_atoms: list[dict] = []
    raw_bonds: list[tuple[int, int, int, bool]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("@<TRIPOS>"):
            section = stripped.upper()
            mol_line_idx = 0
            continue

        if section == "@<TRIPOS>MOLECULE":
            mol_line_idx += 1
            if mol_line_idx == 1 and stripped and stripped != "*****":
                mol_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", stripped).strip("_") or default_name
        elif section == "@<TRIPOS>ATOM":
            parts = stripped.split()
            if len(parts) < 6:
                continue
            try:
                atom_id = int(parts[0])
                atom_name = parts[1]
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                sybyl_type = parts[5]
                charge = float(parts[8]) if len(parts) >= 9 else 0.0
            except ValueError:
                continue
            base_sym = sybyl_type.split(".")[0].strip()
            element = base_sym.upper()
            if element not in ATOMIC_WEIGHTS:
                alpha = "".join(ch for ch in atom_name if ch.isalpha())
                element = (alpha[:2].upper() if alpha[:2].upper() in {"CL", "BR", "NA", "MG", "ZN", "FE", "CA", "CU", "MN"} else alpha[:1].upper()) or "C"
            is_aromatic = ".ar" in sybyl_type.lower()
            raw_atoms.append({
                "mol2_id": atom_id,
                "name": atom_name[:4],
                "symbol": element,
                "sybyl_type": sybyl_type,
                "aromatic": is_aromatic,
                "coord": np.array([x, y, z], dtype=float),
                "charge": charge,
            })
        elif section == "@<TRIPOS>BOND":
            parts = stripped.split()
            if len(parts) < 4:
                continue
            try:
                u_id = int(parts[1])
                v_id = int(parts[2])
                b_type = parts[3].lower()
            except ValueError:
                continue
            order = 2 if b_type == "2" else (3 if b_type == "3" else 1)
            is_arom_bond = b_type == "ar"
            raw_bonds.append((u_id, v_id, order, is_arom_bond))

    if not raw_atoms:
        raise ValueError(f"No valid @<TRIPOS>ATOM records found in MOL2 block for {default_name}")

    id_to_raw_idx = {a["mol2_id"]: idx for idx, a in enumerate(raw_atoms)}
    raw_adj: dict[int, list[tuple[int, int, bool]]] = {i: [] for i in range(len(raw_atoms))}
    for u_id, v_id, order, is_ar in raw_bonds:
        if u_id in id_to_raw_idx and v_id in id_to_raw_idx:
            ui, vi = id_to_raw_idx[u_id], id_to_raw_idx[v_id]
            raw_adj[ui].append((vi, order, is_ar))
            raw_adj[vi].append((ui, order, is_ar))

    # Fold non-polar/explicit H partial charges onto their parent heavy atoms and keep heavy atoms for PDBQT
    heavy_indices = [i for i, a in enumerate(raw_atoms) if a["symbol"] != "H"]
    if not heavy_indices:
        heavy_indices = list(range(len(raw_atoms)))

    for i, a in enumerate(raw_atoms):
        if a["symbol"] == "H" and raw_adj[i]:
            parent_idx = raw_adj[i][0][0]
            raw_atoms[parent_idx]["charge"] += a["charge"]

    # Optional desalting of disconnected counterion fragments (keep largest organic heavy-atom fragment)
    selected_heavy_set = set(heavy_indices)
    stripped_counterions: list[str] = []
    if auto_desalt and len(heavy_indices) > 1:
        visited: set[int] = set()
        components: list[list[int]] = []
        for hi in heavy_indices:
            if hi in visited:
                continue
            comp = []
            q = [hi]
            visited.add(hi)
            while q:
                cur = q.pop(0)
                comp.append(cur)
                for nb, _, _ in raw_adj[cur]:
                    if nb in selected_heavy_set and nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            components.append(comp)
        if len(components) > 1:
            components.sort(
                key=lambda comp_list: (
                    int(any(raw_atoms[idx]["symbol"] == "C" for idx in comp_list)),
                    len(comp_list),
                ),
                reverse=True,
            )
            selected_heavy_set = set(components[0])
            for salt_comp in components[1:]:
                stripped_counterions.append("".join(raw_atoms[idx]["symbol"] for idx in salt_comp))

    kept_raw_indices = [i for i in heavy_indices if i in selected_heavy_set]
    raw_to_new = {old_i: new_i for new_i, old_i in enumerate(kept_raw_indices)}

    atoms_for_peoe: list[dict] = []
    bonds_new: list[tuple[int, int, int, bool]] = []
    adj_new: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(kept_raw_indices))}

    # Build heavy-atom bond graph and detect ring closures via BFS
    temp_edges: list[tuple[int, int, int, bool]] = []
    temp_adj: dict[int, list[int]] = {i: [] for i in range(len(kept_raw_indices))}
    for u_id, v_id, order, is_ar in raw_bonds:
        if u_id in id_to_raw_idx and v_id in id_to_raw_idx:
            uo, vo = id_to_raw_idx[u_id], id_to_raw_idx[v_id]
            if uo in raw_to_new and vo in raw_to_new:
                un, vn = raw_to_new[uo], raw_to_new[vo]
                temp_edges.append((un, vn, order, is_ar))
                temp_adj[un].append(vn)
                temp_adj[vn].append(un)

    for un, vn, order, is_ar in temp_edges:
        # Check if (un, vn) lies on a cycle
        vis = {un}
        q = [un]
        in_ring = False
        while q:
            cur = q.pop(0)
            if cur == vn:
                in_ring = True
                break
            for nb in temp_adj[cur]:
                if (cur == un and nb == vn) or (cur == vn and nb == un):
                    continue
                if nb not in vis:
                    vis.add(nb)
                    q.append(nb)
        bonds_new.append((un, vn, order, in_ring or is_ar))
        adj_new[un].append((vn, order))
        adj_new[vn].append((un, order))

    for new_i, old_i in enumerate(kept_raw_indices):
        ra = raw_atoms[old_i]
        h_attached = sum(1 for nb, _, _ in raw_adj[old_i] if raw_atoms[nb]["symbol"] == "H")
        formal_q = 0
        st_low = ra["sybyl_type"].lower()
        if st_low in {"n.4", "n.cat", "c.cat"}:
            formal_q = 1
        elif st_low == "o.co2":
            formal_q = -1
        atoms_for_peoe.append({
            "index": new_i,
            "symbol": ra["symbol"],
            "aromatic": ra["aromatic"],
            "implicit_h": h_attached,
            "charge": formal_q,
        })

    # Use MOL2 partial charges if non-zero; otherwise compute 6-iteration Gasteiger-Marsili PEOE charges
    has_mol2_charges = any(abs(raw_atoms[old_i]["charge"]) > 1e-4 for old_i in kept_raw_indices)
    peoe_charges = compute_peoe_charges(atoms_for_peoe, adj_new, iterations=6) if not has_mol2_charges else [
        round(float(raw_atoms[old_i]["charge"]), 4) for old_i in kept_raw_indices
    ]

    atom_records: list[dict] = []
    for new_i, old_i in enumerate(kept_raw_indices):
        ra = raw_atoms[old_i]
        sym = ra["symbol"]
        st_low = ra["sybyl_type"].lower()
        h_attached = atoms_for_peoe[new_i]["implicit_h"]
        if sym == "O":
            ad_type = "OA"
        elif sym == "N":
            ad_type = "NA" if (h_attached == 0 and st_low not in {"n.am", "n.pl3", "n.4"}) else "N"
        elif sym == "S":
            ad_type = "SA" if h_attached == 0 else "S"
        elif sym == "C" and ra["aromatic"]:
            ad_type = "A"
        else:
            ad_type = sym
        atom_records.append({
            "serial": new_i + 1,
            "name": ra["name"] or f"{sym}{new_i + 1}"[:4],
            "symbol": sym,
            "sybyl_type": ra["sybyl_type"],
            "aromatic": ra["aromatic"],
            "implicit_h": h_attached,
            "ad_type": ad_type,
            "coord": ra["coord"],
            "charge": float(peoe_charges[new_i]),
        })

    ring_count = max(0, len(bonds_new) - len(atom_records) + 1)
    rotatable_bonds = 0
    for u, v, order, is_ring in bonds_new:
        if is_ring or order != 1:
            continue
        if len(adj_new[u]) > 1 and len(adj_new[v]) > 1:
            rotatable_bonds += 1

    total_h = sum(a["implicit_h"] for a in atoms_for_peoe)
    mw = sum(ATOMIC_WEIGHTS.get(a["symbol"], 12.0) for a in atom_records) + total_h * ATOMIC_WEIGHTS["H"]

    return {
        "name": mol_name,
        "atoms": atom_records,
        "peoe_atoms": atoms_for_peoe,
        "bonds": bonds_new,
        "adj": adj_new,
        "heavy_atom_count": len(atom_records),
        "ring_count": ring_count,
        "rotatable_bond_count": rotatable_bonds,
        "formal_charge": int(round(sum(a["charge"] for a in atom_records))),
        "molecular_weight": round(mw, 3),
        "was_desalted": bool(stripped_counterions),
        "stripped_counterions": stripped_counterions,
        "has_mol2_charges": has_mol2_charges,
    }


def parse_mol2_file(mol2_path: Path, auto_desalt: bool = True) -> list[dict]:
    """Parse a single-molecule or multi-molecule Tripos .mol2 file into a list of parsed molecule dicts."""
    text = mol2_path.read_text(encoding="utf-8", errors="replace")
    raw_blocks = re.split(r"(?=@<TRIPOS>MOLECULE)", text)
    blocks = [b for b in raw_blocks if "@<TRIPOS>ATOM" in b]
    if not blocks:
        raise ValueError(f"No @<TRIPOS>MOLECULE / @<TRIPOS>ATOM sections found in {mol2_path}")
    results: list[dict] = []
    seen_names: set[str] = set()
    for idx, block in enumerate(blocks, 1):
        fallback_name = mol2_path.stem if len(blocks) == 1 else f"{mol2_path.stem}_{idx}"
        parsed = parse_mol2_block(block, default_name=fallback_name, auto_desalt=auto_desalt)
        name = parsed["name"]
        if name in seen_names or name.lower() in {"ligand", "mol", "unl", "none"}:
            name = fallback_name if fallback_name not in seen_names else f"{mol2_path.stem}_{idx}"
            parsed["name"] = name
        seen_names.add(name)
        parsed["source_path"] = str(mol2_path)
        parsed["mol2_block"] = block
        results.append(parsed)
    return results


def convert_cif_to_pdb(cif_path: Path, output_pdb_path: Path | None = None) -> Path:
    """Convert a PDBx/mmCIF (.cif/.mmcif) receptor structure into a standard 80-column PDB file."""
    if output_pdb_path is None:
        output_pdb_path = cif_path.with_suffix(".converted.pdb")
    text = cif_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    headers: list[str] = []
    pdb_lines: list[str] = []
    in_atom_loop = False
    serial = 1
    first_model_num: str | None = None

    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if stripped == "loop_":
            j = i + 1
            loop_cols: list[str] = []
            while j < n and lines[j].strip().startswith("_"):
                loop_cols.append(lines[j].strip())
                j += 1
            if loop_cols and all(c.startswith("_atom_site.") for c in loop_cols):
                headers = [c.split(".", 1)[1].strip() for c in loop_cols]
                in_atom_loop = True
                i = j
                continue
            else:
                in_atom_loop = False
        elif in_atom_loop:
            if not stripped or stripped.startswith(("#", "loop_", "_")) or stripped.startswith("data_"):
                in_atom_loop = False
            else:
                # Tokenize preserving quoted strings
                tokens = [m.group(1) or m.group(2) or m.group(0) for m in re.finditer(r"'([^']*)'|\"([^\"]*)\"|(\S+)", stripped)]
                if len(tokens) >= len(headers):
                    row = dict(zip(headers, tokens))
                    model_num = row.get("pdbx_PDB_model_num", "1")
                    if first_model_num is None:
                        first_model_num = model_num
                    elif model_num != first_model_num:
                        i += 1
                        continue
                    record = row.get("group_PDB", "ATOM").upper()
                    if record not in {"ATOM", "HETATM"}:
                        record = "ATOM"
                    atom_name = (row.get("auth_atom_id") or row.get("label_atom_id") or "C").strip()
                    element = (row.get("type_symbol") or atom_name[:1] or "C").strip().upper()
                    element = "".join(ch for ch in element if ch.isalpha())[:2] or "C"
                    altloc = row.get("label_alt_id", ".")
                    altloc = " " if altloc in {".", "?"} else altloc[:1]
                    resname = (row.get("auth_comp_id") or row.get("label_comp_id") or "UNK").strip()[:3]
                    chain_id = (row.get("auth_asym_id") or row.get("label_asym_id") or "A").strip()[:1] or "A"
                    seq_str = (row.get("auth_seq_id") or row.get("label_seq_id") or "1").strip()
                    try:
                        resseq = int(seq_str)
                    except ValueError:
                        resseq = 1
                    icode = row.get("pdbx_PDB_ins_code", ".")
                    icode = " " if icode in {".", "?"} else icode[:1]
                    try:
                        x = float(row.get("Cartn_x", "0.0"))
                        y = float(row.get("Cartn_y", "0.0"))
                        z = float(row.get("Cartn_z", "0.0"))
                    except ValueError:
                        i += 1
                        continue
                    try:
                        occ = float(row.get("occupancy", "1.0"))
                    except ValueError:
                        occ = 1.0
                    try:
                        bfac = float(row.get("B_iso_or_equiv", "20.0"))
                    except ValueError:
                        bfac = 20.0

                    if len(atom_name) < 4 and len(element) == 1:
                        fmt_name = f" {atom_name:<3s}"
                    else:
                        fmt_name = f"{atom_name:<4s}"[:4]
                    pdb_lines.append(
                        f"{record:<6s}{serial:5d} {fmt_name}{altloc:1s}{resname:>3s} "
                        f"{chain_id:1s}{resseq:4d}{icode:1s}   "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}          {element:>2s}  "
                    )
                    serial += 1
        i += 1

    if not pdb_lines:
        raise ValueError(f"No valid _atom_site records found in CIF file: {cif_path}")
    pdb_lines.append("END")
    output_pdb_path.parent.mkdir(parents=True, exist_ok=True)
    output_pdb_path.write_text("\n".join(pdb_lines) + "\n", encoding="utf-8")
    return output_pdb_path


def parse_sdf_block(sdf_text: str, default_name: str = "Ligand", auto_desalt: bool = True) -> dict:
    """Parse a V2000 .sdf / .mol structure block into 3D heavy-atom records, bonds, PEOE charges, and topological descriptors."""
    lines = sdf_text.splitlines()
    if len(lines) < 4:
        raise ValueError(f"Invalid SDF/MOL block for {default_name}")
    raw_title = lines[0].strip()
    mol_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_title).strip("_") if raw_title else default_name
    if not mol_name or mol_name.lower() in {"ligand", "mol", "none", "unl"}:
        mol_name = default_name

    counts_line = lines[3]
    try:
        n_atoms = int(counts_line[0:3].strip())
        n_bonds = int(counts_line[3:6].strip())
    except ValueError:
        parts = counts_line.split()
        if len(parts) < 2:
            raise ValueError(f"Cannot parse SDF counts line: {counts_line}")
        n_atoms, n_bonds = int(parts[0]), int(parts[1])

    raw_atoms: list[dict] = []
    for idx in range(4, min(len(lines), 4 + n_atoms)):
        line = lines[idx]
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            sym = parts[3].strip().upper()
        except ValueError:
            continue
        if sym not in ATOMIC_WEIGHTS:
            sym = sym[:1] if sym[:1] in ATOMIC_WEIGHTS else "C"
        raw_atoms.append({
            "sdf_id": len(raw_atoms) + 1,
            "name": f"{sym}{len(raw_atoms) + 1}"[:4],
            "symbol": sym,
            "coord": np.array([x, y, z], dtype=float),
            "formal_charge": 0,
        })

    raw_bonds: list[tuple[int, int, int]] = []
    bond_start = 4 + n_atoms
    for idx in range(bond_start, min(len(lines), bond_start + n_bonds)):
        line = lines[idx]
        try:
            u_id = int(line[0:3].strip())
            v_id = int(line[3:6].strip())
            b_type = int(line[6:9].strip())
        except ValueError:
            parts = line.split()
            if len(parts) < 3:
                continue
            u_id, v_id, b_type = int(parts[0]), int(parts[1]), int(parts[2])
        order = 2 if b_type in {2, 4} else (3 if b_type == 3 else 1)
        raw_bonds.append((u_id - 1, v_id - 1, order))

    for line in lines[bond_start + n_bonds:]:
        if line.startswith("M  CHG"):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    cnt = int(parts[2])
                    for k in range(cnt):
                        a_idx = int(parts[3 + 2 * k]) - 1
                        chg = int(parts[4 + 2 * k])
                        if 0 <= a_idx < len(raw_atoms):
                            raw_atoms[a_idx]["formal_charge"] = chg
                except (ValueError, IndexError):
                    pass

    if not raw_atoms:
        raise ValueError(f"No atoms parsed from SDF block for {default_name}")

    raw_adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(raw_atoms))}
    for u, v, order in raw_bonds:
        if 0 <= u < len(raw_atoms) and 0 <= v < len(raw_atoms):
            raw_adj[u].append((v, order))
            raw_adj[v].append((u, order))

    heavy_indices = [i for i, a in enumerate(raw_atoms) if a["symbol"] != "H"] or list(range(len(raw_atoms)))
    selected_heavy = set(heavy_indices)
    stripped_counterions: list[str] = []

    if auto_desalt and len(heavy_indices) > 1:
        visited: set[int] = set()
        comps: list[list[int]] = []
        for hi in heavy_indices:
            if hi in visited:
                continue
            comp = []
            q = [hi]
            visited.add(hi)
            while q:
                cur = q.pop(0)
                comp.append(cur)
                for nb, _ in raw_adj[cur]:
                    if nb in selected_heavy and nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            comps.append(comp)
        if len(comps) > 1:
            comps.sort(key=lambda c: (int(any(raw_atoms[i]["symbol"] == "C" for i in c)), len(c)), reverse=True)
            selected_heavy = set(comps[0])
            for sc in comps[1:]:
                stripped_counterions.append("".join(raw_atoms[i]["symbol"] for i in sc))

    kept = [i for i in heavy_indices if i in selected_heavy]
    old_to_new = {old_i: new_i for new_i, old_i in enumerate(kept)}

    temp_adj: dict[int, list[int]] = {i: [] for i in range(len(kept))}
    temp_edges: list[tuple[int, int, int]] = []
    for u, v, order in raw_bonds:
        if u in old_to_new and v in old_to_new:
            un, vn = old_to_new[u], old_to_new[v]
            temp_edges.append((un, vn, order))
            temp_adj[un].append(vn)
            temp_adj[vn].append(un)

    bonds_new: list[tuple[int, int, int, bool]] = []
    adj_new: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(kept))}
    for un, vn, order in temp_edges:
        vis = {un}
        q = [un]
        in_ring = False
        while q:
            cur = q.pop(0)
            if cur == vn:
                in_ring = True
                break
            for nb in temp_adj[cur]:
                if (cur == un and nb == vn) or (cur == vn and nb == un):
                    continue
                if nb not in vis:
                    vis.add(nb)
                    q.append(nb)
        bonds_new.append((un, vn, order, in_ring))
        adj_new[un].append((vn, order))
        adj_new[vn].append((un, order))

    peoe_atoms: list[dict] = []
    for new_i, old_i in enumerate(kept):
        ra = raw_atoms[old_i]
        explicit_h = sum(1 for nb, _ in raw_adj[old_i] if raw_atoms[nb]["symbol"] == "H")
        bond_sum = sum(order for _, order in adj_new[new_i]) + explicit_h
        val = DEFAULT_VALENCE.get(ra["symbol"], 0)
        imp_h = explicit_h if explicit_h > 0 else max(0, val - bond_sum + ra["formal_charge"])
        is_arom = any(order == 2 and is_r for u, v, order, is_r in bonds_new if u == new_i or v == new_i)
        peoe_atoms.append({
            "index": new_i,
            "symbol": ra["symbol"],
            "aromatic": is_arom,
            "implicit_h": imp_h,
            "charge": ra["formal_charge"],
        })

    peoe_q = compute_peoe_charges(peoe_atoms, adj_new, iterations=6)
    atom_records: list[dict] = []
    for new_i, old_i in enumerate(kept):
        ra = raw_atoms[old_i]
        pa = peoe_atoms[new_i]
        sym = ra["symbol"]
        if sym == "O":
            ad_type = "OA"
        elif sym == "N":
            ad_type = "NA" if pa["implicit_h"] == 0 else "N"
        elif sym == "S":
            ad_type = "SA" if pa["implicit_h"] == 0 else "S"
        elif sym == "C" and pa["aromatic"]:
            ad_type = "A"
        else:
            ad_type = sym
        atom_records.append({
            "serial": new_i + 1,
            "name": ra["name"],
            "symbol": sym,
            "aromatic": pa["aromatic"],
            "implicit_h": pa["implicit_h"],
            "ad_type": ad_type,
            "coord": ra["coord"],
            "charge": float(peoe_q[new_i]),
        })

    ring_count = max(0, len(bonds_new) - len(atom_records) + 1)
    rotatable_bonds = sum(
        1 for u, v, order, is_ring in bonds_new
        if not is_ring and order == 1 and len(adj_new[u]) > 1 and len(adj_new[v]) > 1
    )
    total_h = sum(a["implicit_h"] for a in peoe_atoms)
    mw = sum(ATOMIC_WEIGHTS.get(a["symbol"], 12.0) for a in atom_records) + total_h * ATOMIC_WEIGHTS["H"]

    return {
        "name": mol_name,
        "atoms": atom_records,
        "peoe_atoms": peoe_atoms,
        "bonds": bonds_new,
        "adj": adj_new,
        "heavy_atom_count": len(atom_records),
        "ring_count": ring_count,
        "rotatable_bond_count": rotatable_bonds,
        "formal_charge": sum(a["charge"] for a in peoe_atoms),
        "molecular_weight": round(mw, 3),
        "was_desalted": bool(stripped_counterions),
        "stripped_counterions": stripped_counterions,
        "has_mol2_charges": False,
    }


def parse_sdf_file(sdf_path: Path, auto_desalt: bool = True) -> list[dict]:
    """Parse a single- or multi-molecule .sdf / .mol file into a list of parsed structure dicts."""
    text = sdf_path.read_text(encoding="utf-8", errors="replace")
    raw_blocks = [b.strip() for b in text.split("$$$$") if "V2000" in b or len(b.strip().splitlines()) >= 5]
    if not raw_blocks:
        raise ValueError(f"No valid molecule blocks found in {sdf_path}")
    results: list[dict] = []
    seen: set[str] = set()
    for idx, block in enumerate(raw_blocks, 1):
        fallback_name = sdf_path.stem if len(raw_blocks) == 1 else f"{sdf_path.stem}_{idx}"
        parsed = parse_sdf_block(block, default_name=fallback_name, auto_desalt=auto_desalt)
        name = parsed["name"]
        if name in seen:
            name = f"{sdf_path.stem}_{idx}"
            parsed["name"] = name
        seen.add(name)
        parsed["source_path"] = str(sdf_path)
        parsed["sdf_block"] = block
        results.append(parsed)
    return results


def parse_pdb_ligand_file(pdb_path: Path, default_name: str | None = None, auto_desalt: bool = True) -> dict:
    """Parse a 3D ligand .pdb file into heavy-atom records, inferred covalent bonds, PEOE charges, and topological descriptors."""
    name = default_name or pdb_path.stem
    raw_atoms: list[dict] = []
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            a = atom_dict(line)
            if a["element"].upper() != "H":
                raw_atoms.append(a)
    if not raw_atoms:
        raise ValueError(f"No heavy atoms found in ligand PDB: {pdb_path}")

    n = len(raw_atoms)
    temp_adj: dict[int, list[int]] = {i: [] for i in range(n)}
    temp_edges: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = distance(raw_atoms[i]["coord"], raw_atoms[j]["coord"])
            if 0.85 <= d <= 1.88:
                order = 2 if d < 1.38 else 1
                temp_edges.append((i, j, order))
                temp_adj[i].append(j)
                temp_adj[j].append(i)

    bonds_new: list[tuple[int, int, int, bool]] = []
    adj_new: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
    for u, v, order in temp_edges:
        vis = {u}
        q = [u]
        in_ring = False
        while q:
            cur = q.pop(0)
            if cur == v:
                in_ring = True
                break
            for nb in temp_adj[cur]:
                if (cur == u and nb == v) or (cur == v and nb == u):
                    continue
                if nb not in vis:
                    vis.add(nb)
                    q.append(nb)
        bonds_new.append((u, v, order, in_ring))
        adj_new[u].append((v, order))
        adj_new[v].append((u, order))

    peoe_atoms: list[dict] = []
    for i, ra in enumerate(raw_atoms):
        sym = ra["element"].upper()
        bond_sum = sum(order for _, order in adj_new[i])
        imp_h = max(0, DEFAULT_VALENCE.get(sym, 0) - bond_sum)
        is_arom = any(order == 2 and is_r for u, v, order, is_r in bonds_new if u == i or v == i)
        peoe_atoms.append({
            "index": i,
            "symbol": sym,
            "aromatic": is_arom,
            "implicit_h": imp_h,
            "charge": 0,
        })

    peoe_q = compute_peoe_charges(peoe_atoms, adj_new, iterations=6)
    atom_records: list[dict] = []
    for i, ra in enumerate(raw_atoms):
        sym = ra["element"].upper()
        pa = peoe_atoms[i]
        ad_type = "OA" if sym == "O" else ("NA" if sym == "N" and pa["implicit_h"] == 0 else ("A" if sym == "C" and pa["aromatic"] else sym))
        atom_records.append({
            "serial": i + 1,
            "name": ra["name"][:4] or f"{sym}{i + 1}"[:4],
            "symbol": sym,
            "aromatic": pa["aromatic"],
            "implicit_h": pa["implicit_h"],
            "ad_type": ad_type,
            "coord": ra["coord"],
            "charge": float(peoe_q[i]),
        })

    ring_count = max(0, len(bonds_new) - n + 1)
    rotatable_bonds = sum(
        1 for u, v, order, is_ring in bonds_new
        if not is_ring and order == 1 and len(adj_new[u]) > 1 and len(adj_new[v]) > 1
    )
    total_h = sum(a["implicit_h"] for a in peoe_atoms)
    mw = sum(ATOMIC_WEIGHTS.get(a["symbol"], 12.0) for a in atom_records) + total_h * ATOMIC_WEIGHTS["H"]
    return {
        "name": name,
        "atoms": atom_records,
        "peoe_atoms": peoe_atoms,
        "bonds": bonds_new,
        "adj": adj_new,
        "heavy_atom_count": n,
        "ring_count": ring_count,
        "rotatable_bond_count": rotatable_bonds,
        "formal_charge": 0,
        "molecular_weight": round(mw, 3),
        "was_desalted": False,
        "stripped_counterions": [],
        "has_mol2_charges": False,
    }


def compute_admet_druglikeness(
    name: str,
    smiles: str | None = None,
    parsed_structure: dict | None = None,
) -> dict:
    """Compute Lipinski Rule-of-5, Veber oral bioavailability, Ertl TPSA, Wildman-Crippen LogP, HBD, HBA, and QED proxy for SMILES or 3D (.mol2/.sdf/.pdb) ligands."""
    if parsed_structure is not None:
        atoms = parsed_structure.get("peoe_atoms") or parsed_structure.get("atoms", [])
        adj = parsed_structure.get("adj", {})
        mw = float(parsed_structure.get("molecular_weight", 0.0))
        rot_bonds = int(parsed_structure.get("rotatable_bond_count", 0))
        heavy_count = int(parsed_structure.get("heavy_atom_count", len(atoms)))
        ring_count = int(parsed_structure.get("ring_count", 0))
    elif smiles:
        desalt = standardize_and_desalt_smiles(smiles)["desalted_smiles"]
        p = parse_smiles_tokens(desalt)
        atoms = p["atoms"]
        adj = p["adj"]
        mw = float(p["molecular_weight"])
        rot_bonds = int(p["rotatable_bond_count"])
        heavy_count = int(p["heavy_atom_count"])
        ring_count = int(p["ring_count"])
    else:
        raise ValueError(f"Either smiles or parsed_structure is required for ADMET calculation of {name}")

    hbd = 0
    hba = 0
    tpsa = 0.0
    logp = 0.0

    for idx, a in enumerate(atoms):
        sym = a["symbol"].upper()
        imp_h = int(a.get("implicit_h") or 0)
        chg = int(a.get("charge") or 0)
        is_arom = bool(a.get("aromatic", False))
        nbs = adj.get(idx, [])

        if sym in {"N", "O"} and imp_h > 0:
            hbd += imp_h
        if sym in {"N", "O"}:
            hba += 1

        # Ertl group contributions to TPSA (in A^2)
        if sym == "O":
            if imp_h >= 1:
                tpsa += 20.23
            elif chg < 0:
                tpsa += 23.06
            elif any(order == 2 for _, order in nbs):
                tpsa += 17.07
            else:
                tpsa += 9.23
        elif sym == "N":
            if chg > 0 and imp_h >= 3:
                tpsa += 27.64
            elif imp_h >= 2:
                tpsa += 26.02
            elif imp_h == 1:
                tpsa += 12.03 if not is_arom else 15.79
            else:
                tpsa += 12.36 if is_arom else 3.24
        elif sym == "S":
            if imp_h >= 1:
                tpsa += 38.80
            elif any(order == 2 for _, order in nbs):
                tpsa += 17.07
            else:
                tpsa += 25.30
        elif sym == "P":
            tpsa += 9.81

        # Wildman-Crippen atomic LogP contributions
        if sym == "C":
            has_het_nb = any(atoms[nb]["symbol"].upper() in {"N", "O", "S", "F", "CL", "BR", "I"} for nb, _ in nbs if 0 <= nb < len(atoms))
            if is_arom:
                logp += 0.295 if not has_het_nb else 0.158
            else:
                logp += (0.144 + 0.123 * imp_h) if not has_het_nb else (-0.042 + 0.080 * imp_h)
        elif sym == "O":
            logp += -0.289 if imp_h >= 1 else (-0.150 if any(order == 2 for _, order in nbs) else -0.112)
        elif sym == "N":
            logp += -0.520 if imp_h >= 1 else -0.310
        elif sym == "S":
            logp += 0.210
        elif sym == "F":
            logp += 0.420
        elif sym == "CL":
            logp += 0.680
        elif sym == "BR":
            logp += 0.840
        elif sym == "I":
            logp += 1.120

    tpsa = round(tpsa, 2)
    logp = round(logp, 2)

    violations = 0
    if mw > 500.0:
        violations += 1
    if logp > 5.0:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1

    lipinski_pass = violations <= 1
    veber_pass = (rot_bonds <= 10) and (tpsa <= 140.0)

    # Quantitative Estimate of Drug-likeness (QED) smooth desirability score (0.0 to 1.0)
    d_mw = math.exp(-((mw - 320.0) / 180.0) ** 2) if 120.0 <= mw <= 500.0 else 0.45
    d_logp = math.exp(-((logp - 2.5) / 2.5) ** 2)
    d_tpsa = math.exp(-((tpsa - 65.0) / 60.0) ** 2)
    d_rot = 1.0 if rot_bonds <= 8 else max(0.2, 1.0 - 0.08 * (rot_bonds - 8))
    qed_proxy = round(float((d_mw * d_logp * d_tpsa * d_rot) ** 0.25), 3)

    return {
        "Ligand": name,
        "Molecular_Weight_Da": round(mw, 2),
        "LogP_Crippen": logp,
        "TPSA_A2": tpsa,
        "HBD_Count": hbd,
        "HBA_Count": hba,
        "Rotatable_Bonds": rot_bonds,
        "Heavy_Atom_Count": heavy_count,
        "Ring_Count": ring_count,
        "Lipinski_Violations": violations,
        "Lipinski_Rule_of_5_Pass": lipinski_pass,
        "Veber_Oral_Pass": veber_pass,
        "QED_DrugLikeness_Score": qed_proxy,
        "Plain_Chinese_Summary": (
            f"符合类药五原则 (QED={qed_proxy})" if (lipinski_pass and veber_pass) else f"存在 {violations} 项越界需关注口服吸收"
        ),
    }
