from torchdrug import layers
from torchdrug.layers import geometry
from torchdrug import data
import torch
import os
import glob
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor
process_dir = "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_mutated/af3_sp2_output"
# process_dir = "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_sp2_output"
def process_subdir(subdir):
    # 每个子进程内独立初始化模型
    graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()])
    
    gather_dir = os.path.join(process_dir, subdir, "gather")
    ref_pdb = os.path.join(gather_dir, "ref.pdb")
    try:
        ref_pt_G = data.Protein.from_pdb(
            ref_pdb, atom_feature=None, bond_feature="length", residue_feature="symbol")
        ref_pt_G.view = "residue"
        refG = graph_construction_model(ref_pt_G)
        num_chains=refG.chain_id.max()
        
        
        pdb_files = glob.glob(os.path.join(gather_dir, "aligned_*.pdb"))
        dist_list = []
        for pdb in pdb_files:
            conf_pt_G = data.Protein.from_pdb(
                pdb, atom_feature=None, bond_feature="length", residue_feature="symbol")
            conf_pt_G.view = "residue"
            confG = graph_construction_model(conf_pt_G)
            
            dist_within_pt=torch.cdist(confG.node_position,confG.node_position)
            dist_list.append(dist_within_pt)
            
            onehot_chain = F.one_hot(confG.chain_id-1, num_classes=num_chains)
            L=confG.num_residue
            assert L==dist_within_pt.shape[0]
            onehot_i = onehot_chain.unsqueeze(1).expand(L, L, num_chains)
            onehot_j = onehot_chain.unsqueeze(0).expand(L, L, num_chains)
            indicator = torch.cat([onehot_i, onehot_j], dim=-1)
            break
        dist_tensor = torch.stack(dist_list, dim=0)

        # torch.save(dist_tensor, os.path.join(gather_dir, "dist_within_pt_tensor.pt"))
        print(f"process {subdir} done")
    except Exception as e:
        print(f"Error processing {subdir}: {e}")

if __name__ == "__main__":
    subdirs = os.listdir(process_dir)
    # 设置并行最大线程数，比如这里是4
    # max_workers = 96
    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #     executor.map(process_subdir, subdirs)
    
    process_subdir(subdirs[0])

