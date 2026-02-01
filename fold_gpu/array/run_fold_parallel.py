#!/usr/bin/env python
import fcntl
import os
import sys
from pathlib import Path
from typing import List, Union, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForProteinFolding

np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)


def list_files_with_extension(directory, extension):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]


def load_fasta_entries(fasta_path: Union[str, Path]) -> Tuple[List[str], List[str]]:
    path = Path(fasta_path)
    if not path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    descs_list = []
    seqs_list = []

    current_desc = None
    current_chunks = []

    # Read as text; universal newlines handle \r, \n, \r\n transparently.
    with open(fasta_path, "r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            # Skip blank and comment lines
            if not line or line.startswith(";") or line.startswith("#"):
                continue

            if line.startswith(">"):
                # Flush any previous record
                if current_desc is not None:
                    seq = "".join(current_chunks).replace(" ", "")
                    descs_list.append(current_desc)
                    seqs_list.append(seq)
                # Start new record
                current_desc = line[1:].strip()
                current_chunks = []
            else:
                # Sequence line
                current_chunks.append(line)

        # Flush last record at EOF
        if current_desc is not None:
            seq = "".join(current_chunks).replace(" ", "")
            descs_list.append(current_desc)
            seqs_list.append(seq)

    return seqs_list, descs_list


def write_to_shared_file(message: str, shared_file: str) -> None:
    """
    Write a message to a shared file with an exclusive lock.

    Parameters
    ----------
    message : str
        The message to write to the file.
    shared_file : str
        The path to the shared file.

    Returns
    -------
    None
        This function does not return a value.
    """
    with open(shared_file, 'a') as f:
        # Acquire an exclusive lock before writing
        fcntl.flock(f, fcntl.LOCK_EX)
        # Write the message to the file
        f.write(message)
        # Release the lock after writing
        fcntl.flock(f, fcntl.LOCK_UN)
    return None


def fold_sequence_esm(sequence: str, fileout: str = 'tmp.pdb') -> tuple[np.ndarray, float]:
    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        low_cpu_mem_usage=True,
    ).to(device).eval()

    inputs = tok(
        [sequence],
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Updated: use torch.amp.autocast and fix 'enabled' condition
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == device):
        out = model(**inputs)
        pdb_str = model.infer_pdb(sequence)

    # Write the folded structure to the output PDB file
    with open(fileout, "w") as f:
        f.write(pdb_str)

    # Extract pLDDT (per-residue confidence) and pTM (predicted TM-score)
    plddt = out.plddt[0].cpu().numpy()
    ptm = out.ptm.cpu().numpy()

    return plddt, ptm


if __name__ == "__main__":
    i = int(sys.argv[1])
    j = int(sys.argv[2])
    pdb_tmp = f"tmp_{i}_{j}.pdb"
    file = list_files_with_extension("../data/", ".faa")[j]
    entries, descriptions = load_fasta_entries(file)
    entry, description = entries[i], descriptions[i]
    plddt, ptm = fold_sequence_esm(entry, fileout=pdb_tmp)

    with open(pdb_tmp, "r") as f:
        pdb_str = f.read()
    os.remove(pdb_tmp)

    out_str = (f"{file};;; "
               f"{description};;; "
               f"{entry};;; "
               f"{ptm};;; "
               f"{np.array2string(plddt)};;; "
               f"{plddt.mean()};;; "
               f"{pdb_str} ~~~\n")
    write_to_shared_file(out_str, f"{file}.fdat")
