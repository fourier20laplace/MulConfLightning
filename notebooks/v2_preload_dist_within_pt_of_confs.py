from torchdrug import layers
from torchdrug.layers import geometry
from torchdrug import data
import torch
import os
import glob
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor
# process_dir = "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_mutated/af3_sp2_output"
process_dir = "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_sp2_output"

def mark_thresh_sym_min(matrix: torch.Tensor, threshold: float, mark_value: int = 1,indicator_matrix: torch.Tensor = None) -> torch.Tensor:
    #todo: 可能需要进一步改进 目前的想法是 如果Cα距离小于10A,且在不同的链上 则认为这样的位置是重要的
    #todo: 比如对于1a22这个蛋白来说 实际上最近的两个氨基酸为178-300 178这个氨基酸位于一条蛋白的末端 感觉这种作用是不是就不稳定?
    #* :另一方面 这段代码其实给出了如何去找interaction face处的氨基酸 某种程度上来讲 这里额外提供的就是interaction face信息?
    x = matrix.clone()
    if indicator_matrix is not None:
        x[indicator_matrix] = float('inf')  # Simplified boolean indexing

    L = x.size(0)
    # 获取上三角（不含对角线）索引
    triu_indices = torch.triu_indices(L, L, offset=1)
    selected_mask = x[triu_indices[0], triu_indices[1]] < threshold

    # 筛选出满足条件的位置
    i_indices = triu_indices[0][selected_mask]
    j_indices = triu_indices[1][selected_mask]

    # 构造输出矩阵
    out = torch.zeros_like(matrix, dtype=torch.int32)
    out[i_indices, j_indices] = mark_value
    out[j_indices, i_indices] = mark_value  # 保持对称性
    return out



def process_subdir(subdir):
    if subdir.startswith("1kbh") or subdir.startswith("3vr6"):
        return
    
    gather_dir = os.path.join(process_dir, subdir, "gather")
    
    # if os.path.exists(os.path.join(gather_dir, "repr_within_pt_tensor.pt")):
    #     print(f"Found {subdir}")
    #     return
    # else:
    #     print(f"Not found {subdir}")
    #     return
    
    # ref_pdb = os.path.join(gather_dir, "ref.pdb")
    # 每个子进程内独立初始化模型
    graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()])
    
    pdb_files = glob.glob(os.path.join(gather_dir, "af_aligned_*.pdb"))
    dist_list = []
    for pdb in pdb_files:
        conf_pt_G = data.Protein.from_pdb(
            pdb, atom_feature=None, bond_feature="length", residue_feature="symbol")
        conf_pt_G.view = "residue"
        confG = graph_construction_model(conf_pt_G)
        
        dist_within_pt=torch.cdist(confG.node_position,confG.node_position)
        dist_list.append(dist_within_pt)
    dist_tensor = torch.stack(dist_list, dim=-1)
    
    chain_id=confG.chain_id
    L=confG.num_residue
    assert L==dist_within_pt.shape[0]
    chain_id_i=chain_id.unsqueeze(1).expand(L, L)
    chain_id_j=chain_id.unsqueeze(0).expand(L, L)
    indicator_bool = chain_id_i==chain_id_j
    #bool to 01 and unsqueeze
    indicator=indicator_bool.float()
    indicator=indicator.unsqueeze(-1)
    
    min_dist_matrix=torch.min(dist_tensor,dim=-1)[0].unsqueeze(-1)
    max_dist_matrix=torch.max(dist_tensor,dim=-1)[0].unsqueeze(-1)
    mean_dist_matrix=torch.mean(dist_tensor,dim=-1).unsqueeze(-1)
    std_dist_matrix=torch.std(dist_tensor,dim=-1).unsqueeze(-1)
    
    my_importance_matrix=mark_thresh_sym_min(min_dist_matrix.squeeze(-1),10,1,indicator_bool).unsqueeze(-1)
    
    residue_type=F.one_hot(confG.residue_type,num_classes=21)
    restype_i=residue_type.unsqueeze(1).expand(L, L,21)
    restype_j=residue_type.unsqueeze(0).expand(L, L,21)
    restype_i2j=torch.cat([restype_i,restype_j],dim=-1)

    fin_tensor=torch.cat([dist_tensor,indicator,my_importance_matrix,min_dist_matrix,max_dist_matrix,mean_dist_matrix,std_dist_matrix,restype_i2j],dim=-1)
    # hdim=fin_tensor.shape[-1]
    # fin_tensor=fin_tensor.view(-1,hdim)
    torch.save(fin_tensor, os.path.join(gather_dir, "repr_within_pt_tensor.pt"))
    # torch.save(dist_tensor, os.path.join(gather_dir, "dist_within_pt_tensor.pt"))
    print(f"process {subdir} done")


if __name__ == "__main__":
    subdirs = os.listdir(process_dir)
    # 设置并行最大线程数，比如这里是4
    max_workers = 96
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_subdir, subdirs)
    
    # process_subdir(subdirs[0])

