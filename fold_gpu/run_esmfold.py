#!/usr/bin/env python
import torch
from transformers import AutoTokenizer, EsmForProteinFolding

if __name__ == "__main__":
    device = torch.device("gpu")
    sequence = "MGAGASAEEKHSRELEKKLK"
    fileout = "tmp.pdb"

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
    print(plddt.mean(), ptm)
