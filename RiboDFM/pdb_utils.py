from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser


ATOM_NAMES = ("OP1", "OP2", "P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'")
ATOM_INDEX = {atom: index for index, atom in enumerate(ATOM_NAMES)}
REQUIRED_ATOMS = ("C4'", "C3'", "C2'", "C1'")
BASE_INDEX = {"A": 0, "U": 1, "C": 2, "G": 3}
RESIDUE_BASE = {"A": "A", "RA": "A", "ADE": "A", "U": "U", "RU": "U", "URA": "U", "C": "C", "RC": "C", "CYT": "C", "G": "G", "RG": "G", "GUA": "G"}


@dataclass
class ChainLayout:
    chain_id: str
    length: int
    model_indices: list
    fasta_positions: list


@dataclass
class StructureInput:
    name: str
    coords: np.ndarray
    residue_idx: np.ndarray
    chains: list


def _atom_coord(residue, atom_name):
    aliases = {
        "OP1": ("OP1", "O1P"),
        "OP2": ("OP2", "O2P"),
        "O5'": ("O5'", "O5*"),
        "C5'": ("C5'", "C5*"),
        "C4'": ("C4'", "C4*"),
        "O4'": ("O4'", "O4*"),
        "C3'": ("C3'", "C3*"),
        "O3'": ("O3'", "O3*"),
        "C2'": ("C2'", "C2*"),
        "O2'": ("O2'", "O2*"),
        "C1'": ("C1'", "C1*"),
    }
    for candidate in aliases.get(atom_name, (atom_name,)):
        if candidate in residue:
            return np.asarray(residue[candidate].coord, dtype=np.float32)
    return None


def _is_rna_residue(residue):
    return any(_atom_coord(residue, atom) is not None for atom in ATOM_NAMES)


def _pdb_chains(path, selected_chains=None):
    model = PDBParser(QUIET=True).get_structure("", str(path))[0]
    available_chains = list(model.get_chains())
    if not available_chains:
        raise ValueError("No chains were found in the PDB file")
    chain_lookup = {(chain.id.strip() or "_"): chain for chain in available_chains}
    if selected_chains:
        missing = sorted(set(selected_chains) - set(chain_lookup))
        if missing:
            raise ValueError(f"RNA chain not found: {','.join(missing)}")
        chains = [chain_lookup[chain_id] for chain_id in selected_chains]
    else:
        chains = [available_chains[0]]
    return chains


def _base_name(residue):
    name = residue.resname.strip().upper()
    if name in RESIDUE_BASE:
        return RESIDUE_BASE[name]
    if len(name) == 2 and name[-1] in BASE_INDEX:
        return name[-1]
    return None


def _coordinate_row(residue):
    coordinates = [_atom_coord(residue, atom) for atom in ATOM_NAMES]
    if not all(coordinates[ATOM_INDEX[atom]] is not None for atom in REQUIRED_ATOMS):
        return None
    row = np.full((len(ATOM_NAMES), 3), np.nan, dtype=np.float32)
    for atom_index, coordinate in enumerate(coordinates):
        if coordinate is not None:
            row[atom_index] = coordinate
    return row


def parse_pdb(path, selected_chains=None, chain_gap=32):
    path = Path(path)
    chains = _pdb_chains(path, selected_chains)
    coord_rows = []
    residue_indices = []
    layouts = []
    model_index = 0
    global_offset = 0
    for chain in chains:
        chain_id = chain.id.strip() or "_"
        residues_by_position = {}
        for residue in chain.get_residues():
            residue_number = int(residue.id[1])
            if residue_number < 1 or not _is_rna_residue(residue):
                continue
            residues_by_position[residue_number - 1] = residue
        residues = sorted(residues_by_position.items())
        if not residues:
            continue
        chain_length = residues[-1][0] + 1
        valid_model_indices = []
        valid_fasta_positions = []
        for local_position, residue in residues:
            row = _coordinate_row(residue)
            if row is None:
                continue
            coord_rows.append(row)
            residue_indices.append(global_offset + local_position + 1)
            valid_model_indices.append(model_index)
            valid_fasta_positions.append(local_position)
            model_index += 1
        layouts.append(ChainLayout(chain_id, chain_length, valid_model_indices, valid_fasta_positions))
        global_offset += chain_length + chain_gap
    if not coord_rows:
        raise ValueError("No RNA residues with C4', C3', C2' and C1' coordinates were found")
    if len(coord_rows) < 6:
        raise ValueError("At least six structurally complete RNA residues are required")
    return StructureInput(path.stem, np.stack(coord_rows), np.asarray(residue_indices, dtype=np.int64), layouts)


def parse_qa_pdb(path, selected_chains=None):
    path = Path(path)
    coords = []
    sequence = []
    labels = []
    for chain in _pdb_chains(path, selected_chains):
        chain_id = chain.id.strip() or "_"
        for residue in chain.get_residues():
            base = _base_name(residue)
            row = _coordinate_row(residue)
            if base is None or row is None:
                continue
            coords.append(row)
            sequence.append(BASE_INDEX[base])
            labels.append((chain_id, int(residue.id[1]), residue.id[2].strip(), base))
    if len(coords) < 6:
        raise ValueError("At least six valid RNA residues are required")
    return np.stack(coords), np.asarray(sequence, dtype=np.int64), labels


def sequences_with_gaps(prediction, structure):
    records = []
    for layout in structure.chains:
        sequence = ["-"] * layout.length
        for model_index, fasta_position in zip(layout.model_indices, layout.fasta_positions):
            sequence[fasta_position] = prediction[model_index]
        records.append((layout.chain_id, "".join(sequence)))
    return records


def write_fasta(path, structure, predictions, confidences=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for sample_index, prediction in enumerate(predictions, start=1):
            for chain_id, sequence in sequences_with_gaps(prediction, structure):
                confidence = confidences[sample_index - 1] if confidences is not None else None
                confidence_text = f"|confidence={confidence:.6f}" if confidence is not None and np.isfinite(confidence) else ""
                handle.write(f">{structure.name}|sample={sample_index}|chain={chain_id}{confidence_text}\n")
                handle.write(sequence + "\n")
    return path
