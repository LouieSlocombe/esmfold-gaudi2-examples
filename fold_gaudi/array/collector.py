#!/usr/bin/env python
import os
import re
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

import monsterproteinstability as mps


def _safe_first(df: pd.DataFrame, column: str, default=np.nan):
    """
    Safely retrieve the first value of a specified column in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to retrieve the value from.
    column : str
        Name of the column to access.
    default : any, optional
        Value to return if the column does not exist or the DataFrame is empty.
        Default is ``numpy.nan``.

    Returns
    -------
    any
        The first value of ``column`` if the column exists in ``df`` and ``df`` is not empty;
        otherwise returns ``default``.

    Notes
    -----
    The function checks ``column in df`` and ``not df.empty`` before accessing
    ``df[column].iloc[0]`` to avoid ``KeyError`` or ``IndexError`` when the
    column is missing or the DataFrame has no rows.
    """
    if column in df and not df.empty:
        return df[column].iloc[0]
    return default


def _process_entry(item):
    """
    Processes a single entry to extract relevant data.

    Parameters
    ----------
    item : tuple
        A tuple containing the following elements:
        - j (int): Index of the current entry.
        - entry (str): Sequence data for the entry.
        - description (str): Description of the entry.
        - faa_file (str or Path): Path to the FAA file.
        - matching_pdb_files (list of str): List of matching PDB file paths.
        - matching_csv_files (list of str): List of matching CSV file paths.

    Returns
    -------
    dict
        A dictionary containing the processed data with the following keys:
        - "Description" (str): The description of the entry.
        - "Sequence" (str): The sequence data of the entry.
        - "pTM_Score" (float or any): The pTM score from the analysis CSV, or NaN if unavailable.
        - "pLDDT_Score" (float or any): The pLDDT score from the analysis CSV, or NaN if unavailable.
        - "PDB" (str or None): The content of the PDB file, or None if unavailable.
        - "FAA_File" (str): The path to the FAA file as a string.
    """
    j, entry, description, faa_file, matching_pdb_files, matching_csv_files = item
    pdb_str = None
    if j < len(matching_pdb_files):
        with open(matching_pdb_files[j], "r", encoding="utf-8", errors="replace") as f:
            pdb_str = f.read()
    if j < len(matching_csv_files):
        df_analysis = pd.read_csv(matching_csv_files[j])
    else:
        df_analysis = pd.DataFrame()
    return {
        "Description": description,
        "Sequence": entry,
        "pTM_Score": _safe_first(df_analysis, "pTM_Score"),
        "pLDDT_Score": _safe_first(df_analysis, "pLDDT_Score"),
        "PDB": pdb_str,
        "FAA_File": str(faa_file),
    }


def _process_faa_item(item):
    """
    Processes a single FAA file to extract and organize relevant data.

    Parameters
    ----------
    item : tuple
        A tuple containing:
        - i (int): Index of the FAA file.
        - faa_file (str or Path): Path to the FAA file.

    Returns
    -------
    list
        A list of dictionaries, each containing processed data for an entry in the FAA file.
        Each dictionary includes:
        - "Description" (str): Description of the entry.
        - "Sequence" (str): Sequence data of the entry.
        - "pTM_Score" (float or any): pTM score from the analysis CSV, or NaN if unavailable.
        - "pLDDT_Score" (float or any): pLDDT score from the analysis CSV, or NaN if unavailable.
        - "PDB" (str or None): Content of the PDB file, or None if unavailable.
        - "FAA_File" (str): Path to the FAA file as a string.
    """
    i, faa_file = item
    entries, descriptions = mps.load_fasta_entries(faa_file)
    if not entries:
        return []

    # Regular expressions to match PDB and CSV files for the current FAA index
    pattern_pdb = rf"folded_.*_{i}\.pdb$"
    pattern_csv = rf"tmp_.*_{i}\.csv$"

    # Select PDB files matching the current FAA index and sort them by a numeric identifier
    matching_pdb_files = sorted(
        (f for f in pdb_files if re.search(pattern_pdb, f)),
        key=lambda x: int(re.search(r"_(\d+)_\d+\.pdb$", x).group(1)),
    )
    # Select CSV files matching the current FAA index and sort them by a numeric identifier
    matching_csv_files = sorted(
        (f for f in csv_files if re.search(pattern_csv, f)),
        key=lambda x: int(re.search(r"_(\d+)_\d+\.csv$", x).group(1)),
    )

    # Prepare items for processing, pairing entries and descriptions with their respective files
    inner_items = [
        (j, entry, description, faa_file, matching_pdb_files, matching_csv_files)
        for j, (entry, description) in enumerate(zip(entries, descriptions))
    ]

    # Process the items in parallel and collect the results
    chunked_results = list(mps.mp_calc(_process_entry, inner_items))
    print(
        f"Processed FAA file {i + 1}/{len(faa_files)}: {faa_file} "
        f"with {len(chunked_results)} entries",
        flush=True,
    )
    return chunked_results


if __name__ == "__main__":
    data_out = "collected_results"

    cwd = os.getcwd()
    print(f"Current working directory: {cwd}", flush=True)

    dir_faa = os.path.join(cwd, "data")

    # All subdirectories except the FAA data directory
    sub_directories = [
        d for d in mps.list_directories(cwd) if d != Path(dir_faa)
    ]

    # Collect all PDB/CSV files from subdirectories
    pdb_files = list(
        chain.from_iterable(
            mps.list_files_with_extension(sub_dir, ".pdb")
            for sub_dir in sub_directories
        )
    )
    csv_files = list(
        chain.from_iterable(
            mps.list_files_with_extension(sub_dir, ".csv")
            for sub_dir in sub_directories
        )
    )

    # FAA files live in the data directory
    faa_files = mps.list_files_with_extension(dir_faa, ".faa")

    print(f"Total folded PDB Files: {len(pdb_files)}", flush=True)
    print(f"Total folded CSV Files: {len(csv_files)}", flush=True)
    print(f"Total folded FAA Files: {len(faa_files)}", flush=True)

    if len(pdb_files) != len(csv_files):
        raise ValueError(
            f"Mismatched number of PDB ({len(pdb_files)}) and "
            f"CSV ({len(csv_files)}) files"
        )

    # Process all FAA files
    results = list(
        chain.from_iterable(
            _process_faa_item(item) for item in enumerate(faa_files)
        )
    )
    print("Data gathering done!", flush=True)
    df_results = pd.DataFrame(results)
    print("Compressing collected results...", flush=True)
    df_results.to_csv(f"{data_out}.csv.gz", index=False, compression="gzip")
    print(f"Collected results saved to {data_out}", flush=True)
