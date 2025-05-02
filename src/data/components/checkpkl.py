path = "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_output/1a22/1a22_MulConf.pkl"
# read file
import pickle
with open(path, 'rb') as file:
    data = pickle.load(file)
print()