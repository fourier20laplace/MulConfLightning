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

def process_subdir(subdir):
    myset = {"3vr6", "1kbh", "4nm8", "4pwx", "4gxu", "4lrx"}
    if any(subdir.lower().startswith(prefix) for prefix in myset):
        return
    
    gather_dir = os.path.join(process_dir, subdir, "gather")
    

    mask_roi=torch.load( os.path.join(gather_dir, "rand_mask_roi.pt"))
    roi_tensor=torch.load( os.path.join(gather_dir, "rand_roi_tensor.pt"))
    
    
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

