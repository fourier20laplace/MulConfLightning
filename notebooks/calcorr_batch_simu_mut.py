from torchdrug import layers
from torchdrug.layers import geometry
from torchdrug import data
import torch
import os
import glob
from concurrent.futures import ProcessPoolExecutor
process_dir = "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_mutated/af3_sp2_output"
# process_dir = "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_sp2_output"
def process_subdir(subdir):
    # 每个子进程内独立初始化模型
    graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()])
    
    gather_dir = os.path.join(process_dir, subdir, "gather")
    ref_pdb = os.path.join(gather_dir, "ref.pdb")
    # if  os.path.exists(os.path.join(gather_dir, "af_cov_tensor.pt")):
    #     print(f"skip {subdir}")
    #     return
    if  os.path.exists(os.path.join(gather_dir, "corr_tensor.pt")):
        print(f"skip {subdir}")
        return

    try:
        ref_pt_G = data.Protein.from_pdb(
            ref_pdb, atom_feature=None, bond_feature="length", residue_feature="symbol")
        ref_pt_G.view = "residue"
        refG = graph_construction_model(ref_pt_G)

        pdb_files = glob.glob(os.path.join(gather_dir, "aligned_*.pdb"))
        dist_list = []
        for pdb in pdb_files:
            conf_pt_G = data.Protein.from_pdb(
                pdb, atom_feature=None, bond_feature="length", residue_feature="symbol")
            conf_pt_G.view = "residue"
            confG = graph_construction_model(conf_pt_G)

            dist = torch.norm(refG.node_position - confG.node_position, dim=1)
            dist_list.append(dist)

        dist_tensor = torch.stack(dist_list, dim=1)
        # cov_tensor = torch.cov(dist_tensor)
        corr_tensor = torch.corrcoef(dist_tensor)
        # torch.save(cov_tensor, os.path.join(gather_dir, "af_cov_tensor.pt"))
        torch.save(corr_tensor, os.path.join(gather_dir, "corr_tensor.pt"))
        print(f"process {subdir} done")
    except Exception as e:
        print(f"Error processing {subdir}: {e}")

if __name__ == "__main__":
    subdirs = os.listdir(process_dir)
    # 设置并行最大线程数，比如这里是4
    max_workers = 96
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_subdir, subdirs)

