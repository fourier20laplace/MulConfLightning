from torchdrug import layers
from torchdrug.layers import geometry
from torchdrug import data
import torch
import os
import glob
graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()])

ref_pdb="/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/1A4Y.pdb"
confs_fold="/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_sp2_output/1a4y_sp2_rand0.2/gather/"

ref_pt_G = data.Protein.from_pdb(
            ref_pdb, atom_feature=None, bond_feature="length", residue_feature="symbol")
ref_pt_G.view = "residue"
refG = graph_construction_model(ref_pt_G)
#获取confs_fold下的所有pdb文件
pdb_files=glob.glob(os.path.join(confs_fold,"*.pdb"))
#遍历每个pdb文件
dist_list=[]
for pdb in pdb_files:
    #读取pdb文件
    conf_pt_G = data.Protein.from_pdb(pdb, atom_feature=None, bond_feature="length", residue_feature="symbol")
    conf_pt_G.view = "residue"
    confG = graph_construction_model(conf_pt_G)
    
    #计算每个坐标的欧式距离
    dist=torch.norm(refG.node_position-confG.node_position,dim=1)
    dist_list.append(dist)
#dist_list 2 tensor
dist_tensor=torch.stack(dist_list,dim=1)
cov_tensor=torch.cov(dist_tensor)
#将cov保存为csv
import pandas as pd
pd.DataFrame(cov_tensor.cpu().numpy()).to_csv("/home/lmh/projects_dir/af3/alphafold3/MyAnalyze/analyze_dir/1a4y_sp2_rand0.2_cov.csv")

print(dist_tensor.shape)
print(dist_tensor)
print(cov_tensor.shape)
print(cov_tensor)
print("end")

