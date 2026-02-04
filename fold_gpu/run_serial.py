import os

import pandas as pd

import monsterproteinstability as mps


def bulk_fold(file):
    entries, descriptions = mps.load_fasta_entries(file)
    print(f"FASTA entries: {entries}", flush=True)

    results = []
    for j, entry in enumerate(entries):
        print(f"Working on entry # {j}", flush=True)
        # Fold the sequence using ESM
        plddt, ptm = mps.fold_sequence_esm(entry)

        # Load the pdb file and add it to the results
        with open("tmp.pdb", "r") as f:
            pdb_str = f.read()

        # Collect the results in a list
        results.append({
            "Description": descriptions[j],
            "Sequence": entry,
            "pTM_Score": ptm,
            "pLDDT_Score": plddt,
            "Mean_pLDDT_Score": plddt.mean(),
            "PDB": pdb_str,
        })
        # Safely remove the temporary file
        if os.path.exists("tmp.pdb"):
            os.remove("tmp.pdb")

    # Create the dataframe after processing all entries
    df = pd.DataFrame(results)

    # Save the dataframe to a csv file
    df.to_csv(f"results_entry.csv", index=False)
    return df


def bulk_analysis(file):
    df = pd.read_csv(file)

    # Add columns for analysis results
    df["Radius_of_Gyration"] = None
    df["RMSD"] = None
    df["PDB_MD"] = None
    df["SASA"] = None
    df["3DI"] = None
    df["P-SEA"] = None
    df["Protein_Blocks"] = None
    df["DSSP"] = None

    for i in range(len(df)):
        print(f"Analyzing entry # {i}", flush=True)
        pdb_str = df.loc[i, "PDB"]
        with open("tmp.pdb", "w") as f:
            f.write(pdb_str)
        mps.md_workflow("tmp.pdb")

        rad_gyr = mps.get_radius_gyration_time("output.pdb")
        rmsd = mps.get_rmsd_time("output.pdb", "tmp.pdb")

        df.at[i, "Radius_of_Gyration"] = rad_gyr
        df.at[i, "RMSD"] = rmsd

        # Save a stripped version of the pdb
        mps.save_stripped_pdb("output.pdb", f"output.pdb")

        # Load the pdb file and add it to the results
        with open("output.pdb", "r") as f:
            pdb_str = f.read()

        df.at[i, "PDB_MD"] = pdb_str
        df.at[i, "SASA"] = mps.calc_sasa("output.pdb")
        df.at[i, "3DI"] = mps.get_3di_sequence("output.pdb")
        df.at[i, "P-SEA"] = mps.get_p_sea_sequence("output.pdb")
        df.at[i, "Protein_Blocks"] = mps.get_protein_blocks_sequence("output.pdb")
        df.at[i, "DSSP"] = mps.get_dssp_sequence("output.pdb")

        # Safely remove temporary files
        for temp_file in ["tmp.pdb", "output.pdb", "md_log.txt"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # Save the updated dataframe to a new csv file
    df.to_csv(f"analyzed_{file}", index=False)
    return df


if __name__ == "__main__":
    print(flush=True)
    # List all the files in the current directory that end with .faa
    files = mps.list_files_with_extension("data/", ".faa")
    for i, file in enumerate(files):
        print(f"Working with {file}", flush=True)
        bulk_fold(file)
        break

    # load the result file
    bulk_analysis("results_entry_0.csv")
