import numpy as np
import os
from torch.utils.data.dataset import Dataset
from .protein.read_pdbs import parse_pdb, KnnResidue, KnnAgnet
from torchdrug import data, utils
from torchdrug.tasks import MultipleBinaryClassification
import pickle
from tqdm import tqdm
from pathlib import Path
class SKEMPIV2Dataset(Dataset):
    def __init__(self, data_df, is_train, knn_num, knn_agents_num):
        super(SKEMPIV2Dataset, self).__init__()

        self.data_df = data_df
        self.data_batches = len(data_df['PDB_id'])
        self.is_train = is_train
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num

    def __len__(self):
        return self.data_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def __getitem__(self, index):
        index = index % self.data_batches
        sample_info = self.data_df.iloc[index].values
        PDB_id, chain1, chain2, mutate_info, _, ddG = sample_info

        PDB_id = PDB_id.replace('+', '')
        PDB_id = PDB_id.replace('.00', '')

        mutate_info = mutate_info.replace(',', '_')

        wt_name=str(PDB_id).lower()
        wt_file_path='/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_output/'+wt_name
        PDB_wt_file_path = wt_file_path+f"/{wt_name}_model.pdb"
        pt_wt_G=data.Protein.from_pdb(PDB_wt_file_path, atom_feature=None, bond_feature="length", residue_feature="symbol")
        # 字符串转小写
        mt_name=wt_name+'_'+mutate_info.lower()
        mut_file_path='/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_mutated/af3_output/'+mt_name
        PDB_mut_file_path = mut_file_path+f"/{mt_name}_model.pdb"
        pt_mt_G=data.Protein.from_pdb(PDB_mut_file_path, atom_feature=None, bond_feature="length", residue_feature="symbol")
        complex_wt_info = parse_pdb(PDB_wt_file_path)
        complex_mut_info = parse_pdb(PDB_mut_file_path)

        transform = KnnResidue(num_neighbors=self.knn_num)
        agent_select = KnnAgnet(num_neighbors=self.knn_agents_num)

        if len(complex_wt_info['aa']) != len(complex_mut_info['aa']):
            print(sample_info)
            return 0

        mutation_mask = (complex_wt_info['aa'] != complex_mut_info['aa'])

        agent_mask = agent_select({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})

        complex_wt_info['agent_mask'] = agent_mask
        complex_wt_info['PDB_id'] = PDB_id
        complex_wt_info['mutate_info'] = mutate_info

        complex_mut_info['agent_mask'] = agent_mask
        complex_mut_info['PDB_id'] = PDB_id
        complex_mut_info['mutate_info'] = mutate_info

        # batch = transform({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})
        batch ={'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask}
        batch['ddG'] = ddG
        pt_mt_G.view="residue"
        pt_wt_G.view="residue"
        batch['wtG']=pt_wt_G
        batch['mtG']=pt_mt_G
        return batch

import torch
class SKEMPIV2Dataset0429(Dataset):
    def __init__(self, data_df, is_train, knn_num, knn_agents_num,rand):
        super(SKEMPIV2Dataset0429, self).__init__()

        self.data_df = data_df
        self.data_batches = len(data_df['PDB_id'])
        self.is_train = is_train
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num
        self.rand=rand
        if self.rand:
            print(">>>>>>>>>>>>>>>>>>rand")

    def __len__(self):
        return self.data_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def __getitem__(self, index):
        index = index % self.data_batches
        sample_info = self.data_df.iloc[index].values
        PDB_id, chain1, chain2, mutate_info, _, ddG = sample_info

        PDB_id = PDB_id.replace('+', '')
        PDB_id = PDB_id.replace('.00', '')

        mutate_info = mutate_info.replace(',', '_')

        wt_name=str(PDB_id).lower()

        wt_file_path="/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_sp2_output/"+f"{wt_name}_sp2_rand0.2/gather"
        #*0430 update 只有fix的model变了
        #*0430 update cov2corr
        PDB_wt_file_path=os.path.join(wt_file_path,"af_ref.pdb")
        # cov_wt_tensor=torch.load(os.path.join(wt_file_path,"af_cov_tensor.pt"))
        # cov_wt_tensor=torch.load(os.path.join(wt_file_path,"af_corr_tensor.pt"))
        if self.rand:
            cov_wt_tensor=torch.load(os.path.join(wt_file_path,"rand.pt"))
        else:
            cov_wt_tensor=torch.load(os.path.join(wt_file_path,"af_corr_tensor.pt"))
        
        pt_wt_G=data.Protein.from_pdb(PDB_wt_file_path, atom_feature=None, bond_feature="length", residue_feature="symbol")

        mt_name=wt_name+'_'+mutate_info.lower()
        mut_file_path="/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_mutated/af3_sp2_output/"+f"{mt_name}_sp2_rand0.2/gather"
        PDB_mut_file_path=os.path.join(mut_file_path,"ref.pdb")
        # cov_mut_tensor=torch.load(os.path.join(mut_file_path,"cov_tensor.pt"))
        # cov_mut_tensor=torch.load(os.path.join(mut_file_path,"corr_tensor.pt"))
        if self.rand:
            cov_mut_tensor=torch.load(os.path.join(mut_file_path,"rand.pt"))
        else:
            cov_mut_tensor=torch.load(os.path.join(mut_file_path,"corr_tensor.pt"))
        pt_mt_G=data.Protein.from_pdb(PDB_mut_file_path, atom_feature=None, bond_feature="length", residue_feature="symbol")
        
        complex_wt_info = parse_pdb(PDB_wt_file_path)
        complex_mut_info = parse_pdb(PDB_mut_file_path)

        transform = KnnResidue(num_neighbors=self.knn_num)
        agent_select = KnnAgnet(num_neighbors=self.knn_agents_num)

        if len(complex_wt_info['aa']) != len(complex_mut_info['aa']):
            print(sample_info)
            return 0

        mutation_mask = (complex_wt_info['aa'] != complex_mut_info['aa'])

        agent_mask = agent_select({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})

        complex_wt_info['agent_mask'] = agent_mask
        complex_wt_info['PDB_id'] = PDB_id
        complex_wt_info['mutate_info'] = mutate_info

        complex_mut_info['agent_mask'] = agent_mask
        complex_mut_info['PDB_id'] = PDB_id
        complex_mut_info['mutate_info'] = mutate_info

        # batch = transform({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})
        batch ={'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask}
        batch['ddG'] = ddG
        pt_mt_G.view="residue"
        pt_wt_G.view="residue"
        batch['wtG']=pt_wt_G
        batch['mtG']=pt_mt_G
        # 获取cov_tensor
        batch['cov_wt_tensor']=cov_wt_tensor
        batch['cov_mut_tensor']=cov_mut_tensor
        return batch

class SKEMPIV2Dataset0429_mode1(Dataset):
    def __init__(self, data_df, is_train, knn_num, knn_agents_num):
        super(SKEMPIV2Dataset0429_mode1, self).__init__()

        self.data_df = data_df
        self.data_batches = len(data_df['PDB_id'])
        self.is_train = is_train
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num

    def __len__(self):
        return self.data_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def __getitem__(self, index):
        index = index % self.data_batches
        sample_info = self.data_df.iloc[index].values
        PDB_id, chain1, chain2, mutate_info, _, ddG = sample_info

        PDB_id = PDB_id.replace('+', '')
        PDB_id = PDB_id.replace('.00', '')

        mutate_info = mutate_info.replace(',', '_')

        wt_name=str(PDB_id).lower()
        #*0430 update 只有fix的model变了
        wt_file_path="/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_sp2_output/"+f"{wt_name}_sp2_rand0.2/gather"
        PDB_wt_file_path=os.path.join(wt_file_path,"af_ref.pdb")
        pt_wt_G=data.Protein.from_pdb(PDB_wt_file_path, atom_feature=None, bond_feature="length", residue_feature="symbol")

        mt_name=wt_name+'_'+mutate_info.lower()
        mut_file_path="/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_mutated/af3_sp2_output/"+f"{mt_name}_sp2_rand0.2/gather"
        PDB_mut_file_path=os.path.join(mut_file_path,"ref.pdb")
        pt_mt_G=data.Protein.from_pdb(PDB_mut_file_path, atom_feature=None, bond_feature="length", residue_feature="symbol")
        
        complex_wt_info = parse_pdb(PDB_wt_file_path)
        complex_mut_info = parse_pdb(PDB_mut_file_path)

        transform = KnnResidue(num_neighbors=self.knn_num)
        agent_select = KnnAgnet(num_neighbors=self.knn_agents_num)

        if len(complex_wt_info['aa']) != len(complex_mut_info['aa']):
            print(sample_info)
            return 0

        mutation_mask = (complex_wt_info['aa'] != complex_mut_info['aa'])

        agent_mask = agent_select({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})

        complex_wt_info['agent_mask'] = agent_mask
        complex_wt_info['PDB_id'] = PDB_id
        complex_wt_info['mutate_info'] = mutate_info

        complex_mut_info['agent_mask'] = agent_mask
        complex_mut_info['PDB_id'] = PDB_id
        complex_mut_info['mutate_info'] = mutate_info

        # batch = transform({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})
        batch ={'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask}
        batch['ddG'] = ddG
        pt_mt_G.view="residue"
        pt_wt_G.view="residue"
        batch['wtG']=pt_wt_G
        batch['mtG']=pt_mt_G
        return batch

class SKEMPIV2Dataset0429TST(Dataset):
    def __init__(self, data_df, is_train, knn_num, knn_agents_num):
        super(SKEMPIV2Dataset0429TST, self).__init__()

        self.data_df = data_df
        self.data_batches = len(data_df['PDB_id'])
        self.is_train = is_train
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num
        self.graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                                 edge_layers=[geometry.SequentialEdge(max_distance=1),
                                                                              geometry.SpatialEdge(
                                                                     radius=7.0, min_distance=6),
            geometry.KNNEdge(k=6, min_distance=6)], edge_feature="gearnet")

    def __len__(self):
        return self.data_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def __getitem__(self, index):
        index = index % self.data_batches
        sample_info = self.data_df.iloc[index].values
        PDB_id, chain1, chain2, mutate_info, _, ddG = sample_info

        PDB_id = PDB_id.replace('+', '')
        PDB_id = PDB_id.replace('.00', '')

        mutate_info = mutate_info.replace(',', '_')

        wt_name=str(PDB_id).lower()

        wt_file_path="/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_sp2_output/"+f"{wt_name}_sp2_rand0.2/gather"
        PDB_wt_file_path=os.path.join(wt_file_path,"ref.pdb")
        # cov_wt_tensor=torch.load(os.path.join(wt_file_path,"cov_tensor.pt"))

        mt_name=wt_name+'_'+mutate_info.lower()
        mut_file_path="/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_mutated/af3_sp2_output/"+f"{mt_name}_sp2_rand0.2/gather"
        PDB_mut_file_path=os.path.join(mut_file_path,"ref.pdb")
        # cov_mut_tensor=torch.load(os.path.join(mut_file_path,"cov_tensor.pt"))
        
        #检查文件是否存在
        if not os.path.exists(PDB_wt_file_path):
            print(f"Warning: Wild type PDB file not found: {PDB_wt_file_path}")
        if not os.path.exists(PDB_mut_file_path):
            print(f"Warning: Mutant PDB file not found: {PDB_mut_file_path}")
        cov_wt_tensor_path=os.path.join(wt_file_path,"cov_tensor.pt")
        cov_mut_tensor_path=os.path.join(mut_file_path,"cov_tensor.pt")
        if not os.path.exists(cov_wt_tensor_path):
            print(f"Warning: Wild type cov_tensor file not found: {cov_wt_tensor_path}")
        if not os.path.exists(cov_mut_tensor_path):
            print(f"Warning: Mutant cov_tensor file not found: {cov_mut_tensor_path}")
        
        return 1

class SKEMPIV2DatasetTST(Dataset):
    def __init__(self, data_df, is_train, knn_num, knn_agents_num):
        super(SKEMPIV2DatasetTST, self).__init__()

        self.data_df = data_df
        self.data_batches = len(data_df['PDB_id'])
        self.is_train = is_train
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num

    def __len__(self):
        return self.data_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def __getitem__(self, index):
        index = index % self.data_batches
        sample_info = self.data_df.iloc[index].values
        PDB_id, chain1, chain2, mutate_info, _, ddG = sample_info

        PDB_id = PDB_id.replace('+', '')
        PDB_id = PDB_id.replace('.00', '')

        mutate_info = mutate_info.replace(',', '_')

        wt_name = str(PDB_id).lower()
        wt_file_path = '/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_output/'+wt_name
        PDB_wt_file_path = wt_file_path+f"/{wt_name}_model.pdb"
        
        mt_name = wt_name+'_'+mutate_info.lower()
        mut_file_path = '/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_mutated/af3_output/'+mt_name
        PDB_mut_file_path = mut_file_path+f"/{mt_name}_model.pdb"
        # 检查文件是否存在
        if not os.path.exists(PDB_wt_file_path):
            raise FileNotFoundError(f"Wild type PDB file not found: {PDB_wt_file_path}")
        if not os.path.exists(PDB_mut_file_path):
            raise FileNotFoundError(f"Mutant PDB file not found: {PDB_mut_file_path}")
        
        return 1


