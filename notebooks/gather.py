import os
import shutil
# conformers_dir="/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_sp2_output/"
conformers_dir="/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_mutated/af3_sp2_output/"
for dir in os.listdir(conformers_dir):
    conformers_path = os.path.join(conformers_dir, dir)
    os.makedirs(os.path.join(conformers_path, "gather"), exist_ok=True)
    count = 0
    max_files = 5

    for subdir in os.listdir(conformers_path):
        subdir_path = os.path.join(conformers_path, subdir)
        if not os.path.isdir(subdir_path) or subdir_path.startswith("gather"):
            continue

        for file in os.listdir(subdir_path):
            if file.endswith(".cif"):
                source_path = os.path.join(subdir_path, file)
                target_path = os.path.join(conformers_path,"gather", file)
                shutil.copy(source_path, target_path)
                count += 1
                if count >= max_files:
                    break
        if count >= max_files:
            break
