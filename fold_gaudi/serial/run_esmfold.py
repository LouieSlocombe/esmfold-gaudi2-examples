#!/usr/bin/env python
import os

import pandas as pd
import torch
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
from optimum.habana.utils import HabanaGenerationTime
from transformers import AutoTokenizer, EsmForProteinFolding

os.environ["PT_HPU_ENABLE_H2D_DYNAMIC_SLICE"] = "0"
os.environ["PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES"] = "1"

if __name__ == "__main__":
    adapt_transformers_to_gaudi()
    device = torch.device("hpu")
    sequence = "MGAGASAEEKHSRELEKKLK"
    fileout = "tmp.pdb"
    resout = "tmp.csv"

    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        low_cpu_mem_usage=True,
    ).to(device).eval()

    inputs = tok([sequence],
                 return_tensors="pt",
                 add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == device):
        out = model(**inputs)
        pdb_str = model.infer_pdb(sequence)

    # Write the folded structure to the output PDB file
    with open(fileout, "w") as f:
        f.write(pdb_str)

    # Extract pLDDT (per-residue confidence) and pTM (predicted TM-score)
    plddt = out.plddt[0].cpu().numpy()
    ptm = out.ptm.cpu().numpy()
    # convert to dataframe and save to csv
    df = pd.DataFrame([{"sequence": sequence,
                        "pLDDT": plddt,
                        "pTM": float(ptm)}])
    df.to_csv(resout, index=False)
