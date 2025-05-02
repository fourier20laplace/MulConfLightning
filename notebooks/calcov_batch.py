from torchdrug import layers
from torchdrug.layers import geometry
from torchdrug import data
import torch
import os
import glob
graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()])

# ref_pdb="/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/1A4Y.pdb"
# confs_fold="/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_sp2_output/1a4y_sp2_rand0.2/gather/"
process_dir="/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_mutated/af3_sp2_output"
for dir in os.listdir(process_dir):
    print(dir)
    gather_dir=os.path.join(process_dir,dir,"gather")
    ref_pdb=os.path.join(gather_dir,"ref.pdb")
    ref_pt_G = data.Protein.from_pdb(
                ref_pdb, atom_feature=None, bond_feature="length", residue_feature="symbol")
    ref_pt_G.view = "residue"
    refG = graph_construction_model(ref_pt_G)
    #获取confs_fold下的所有pdb文件
    pdb_files=glob.glob(os.path.join(gather_dir,"aligned_*.pdb"))
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
    # 保存cov_tensor
    torch.save(cov_tensor, os.path.join(gather_dir, "cov_tensor.pt"))
    # read cov_tensor
    # cov_tensor=torch.load(os.path.join(gather_dir, "cov_tensor.pt"))
    # print(cov_tensor.shape)
    # print(cov_tensor)
    # print("end")
    # break
    # print(dist_tensor.shape)
    # print(dist_tensor)
    # print(cov_tensor.shape)
    # print(cov_tensor)
    # print("end")
    # break

