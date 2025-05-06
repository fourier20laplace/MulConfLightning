from torchdrug import data
# import torch
import os
# import glob
# import torch.nn.functional as F
import pandas as pd
# process_dir = "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_mutated/af3_sp2_output"
process_dir = "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_sp2_output"
def process_subdir(subdir):
    # print(f"processing {subdir}")
    gather_dir = os.path.join(process_dir, subdir, "gather")
    ref_pdb = os.path.join(gather_dir, "ref.pdb")
    ref_pt_G = data.Protein.from_pdb(
            ref_pdb, atom_feature=None, bond_feature=None, residue_feature=None)
    
    chain_id=set(ref_pt_G.chain_id.tolist())
    num_chains=len(chain_id)
    print(f"{subdir} has {num_chains} chains")
    return num_chains

if __name__ == "__main__":
    subdirs = os.listdir(process_dir)
    results = []
    for subdir in subdirs:
        num_chains = process_subdir(subdir)
        results.append({"subdir": subdir, "num_chains": num_chains})
    # Create DataFrame from list of dictionaries
    df = pd.DataFrame(results)
    # Sort by num_chains in descending order
    df = df.sort_values(by="num_chains", ascending=False)
    df.to_csv("/home/lmh/projects_dir/MulConf0420/notebooks/max_chains_sorted.csv", index=False)


