from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

ONE_LETTER_AA = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U",
}


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
