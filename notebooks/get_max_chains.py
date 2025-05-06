from torchdrug import data
# import torch
import os
# import glob
# import torch.nn.functional as F
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
# process_dir = "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_mutated/af3_sp2_output"
process_dir = "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_sp2_output"
def process_subdir(subdir):
    #*一开始总是卡住主要还是因为1kbh没办法很好的建模
    if subdir.startswith("1kbh") or subdir.startswith("3vr6"):
        return 0
    # 每个子进程内独立初始化模型
    print(f"processing {subdir}")
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
    # 设置并行最大线程数，比如这里是4
    max_workers = 96  # 可根据实际CPU核数调整
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_subdir = {executor.submit(process_subdir, subdir): subdir for subdir in subdirs}
        for future in as_completed(future_to_subdir,timeout=30):
            subdir = future_to_subdir[future]
            try:
                num_chains = future.result()
                results.append({"subdir": subdir, "num_chains": num_chains})
            except Exception as exc:
                print(f"{subdir} generated an exception: {exc}")
    # Create DataFrame from list of dictionaries
    df = pd.DataFrame(results)
    # Sort by num_chains in descending order
    df = df.sort_values(by="num_chains", ascending=False)
    df.to_csv("/home/lmh/projects_dir/MulConf0420/notebooks/max_chains_sorted.csv", index=False)


