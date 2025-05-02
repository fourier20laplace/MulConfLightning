import torch
import os
from concurrent.futures import ProcessPoolExecutor
# process_dir = "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_mutated/af3_sp2_output"
process_dir = "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_sp2_output"
def process_subdir(subdir):

    gather_dir = os.path.join(process_dir, subdir, "gather")
    if  os.path.exists(os.path.join(gather_dir, "af_corr_tensor.pt")):
        #生成一个与corr_tensor.pt大小相同，对角线为1 其余随机在-1，1之间的对称矩阵 rand.pt
        corr_tensor = torch.load(os.path.join(gather_dir, "af_corr_tensor.pt"))
        N = corr_tensor.size(0)
        rand_mat = torch.zeros(N, N)

        # 只填上三角（不包括对角线），然后复制到下三角
        upper_indices = torch.triu_indices(N, N, offset=1)
        random_vals = torch.rand(upper_indices.size(1)) * 2 - 1  # [-1, 1]
        rand_mat[upper_indices[0], upper_indices[1]] = random_vals
        rand_mat = rand_mat + rand_mat.T  # 对称填充

        # 对角线为 1
        rand_mat.fill_diagonal_(1.0)
        torch.save(rand_mat, os.path.join(gather_dir, "rand.pt"))
        print(f"process {subdir} done")
    else:
        raise ValueError(f"corr_tensor.pt not found in {gather_dir}")


if __name__ == "__main__":
    subdirs = os.listdir(process_dir)
    # 设置并行最大线程数，比如这里是4
    max_workers = 96
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_subdir, subdirs)

