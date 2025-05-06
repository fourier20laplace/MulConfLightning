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

    roi_indices = torch.unique(torch.cat([i_indices, j_indices]))
    # 构造输出矩阵
    out = torch.zeros_like(matrix, dtype=torch.int32)
    out[i_indices, j_indices] = mark_value
    out[j_indices, i_indices] = mark_value  # 保持对称性
    return out,roi_indices



def process_subdir(subdir):
    myset = {"3vr6", "1kbh", "4nm8", "4pwx", "4gxu", "4lrx"}
    if any(subdir.lower().startswith(prefix) for prefix in myset):
        return
    
    gather_dir = os.path.join(process_dir, subdir, "gather")
    
    # 每个子进程内独立初始化模型
    graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()])
    
    # pdb_files = glob.glob(os.path.join(gather_dir, "aligned_*.pdb"))
    dist_list = []
    vct_list = []

    ref_pdb = os.path.join(gather_dir, "af_ref.pdb")
    ref_pt_G = data.Protein.from_pdb(
        ref_pdb, atom_feature=None, bond_feature="length", residue_feature="symbol")
    ref_pt_G.view = "residue"
    refG = graph_construction_model(ref_pt_G)
    L=refG.num_residue
    pos_diff = refG.node_position.unsqueeze(
        0) - refG.node_position.unsqueeze(1)
    norms = pos_diff.norm(dim=2, keepdim=True)
    dist_within_pt=norms.squeeze(-1)
    dist_list.append(norms.squeeze(-1))
    norms = norms + torch.eye(L).unsqueeze(-1)  # 加上一个单位矩阵，确保对角线为非零
    unit_vectors = pos_diff / norms
    vct_list.append(unit_vectors)

    dist_list = dist_list*5
    vct_list = vct_list*5
        

    vct_tensor = torch.cat(vct_list, dim=-1)
    dist_tensor = torch.stack(dist_list, dim=-1)
    
    chain_id=refG.chain_id
    #假如有多个链时是否要指示分属哪两条链呢 可是链的数目并不统一 用onehot感觉会无意义 用序号则会引起不必要的数值差异
    #todo:进一步增强信息？
    chain_id_i=chain_id.unsqueeze(1).expand(L, L)
    chain_id_j=chain_id.unsqueeze(0).expand(L, L)
    indicator_bool = chain_id_i==chain_id_j
    #bool to 01 and unsqueeze
    indicator=indicator_bool.float()
    indicator=indicator.unsqueeze(-1)
    
    min_dist_matrix = dist_within_pt.unsqueeze(-1)
    max_dist_matrix = dist_within_pt.unsqueeze(-1)
    mean_dist_matrix = dist_within_pt.unsqueeze(-1)
    std_dist_matrix = torch.zeros_like(mean_dist_matrix)
    
    my_importance_matrix,roi_indices=mark_thresh_sym_min(min_dist_matrix.squeeze(-1),10,1,indicator_bool)
    my_importance_matrix=my_importance_matrix.unsqueeze(-1)

    # * indicator和importance matrix感觉还是要拼接的
    fin_tensor=torch.cat([dist_tensor,vct_tensor,indicator,my_importance_matrix,min_dist_matrix,max_dist_matrix,mean_dist_matrix,std_dist_matrix],dim=-1)
    roi_tensor=fin_tensor[roi_indices][:,roi_indices]
    mask_roi=torch.zeros(L)
    mask_roi[roi_indices]=1
    torch.save(mask_roi, os.path.join(gather_dir, "rand_mask_roi.pt"))
    torch.save(roi_tensor, os.path.join(gather_dir, "rand_roi_tensor.pt"))
    
    print(f"process {subdir} done")


if __name__ == "__main__":
    subdirs = os.listdir(process_dir)
    # 设置并行最大线程数，比如这里是4
    max_workers = 32
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_subdir, subdirs)
    
    # process_subdir("3r9a_sp2_rand0.2")
    # process the first one
    # process_subdir(subdirs[0])

