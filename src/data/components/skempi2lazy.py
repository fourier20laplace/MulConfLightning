import numpy as np
import os
from torch.utils.data.dataset import Dataset
from .protein.read_pdbs import parse_pdb, KnnResidue, KnnAgnet
from torchdrug import data, utils
from torchdrug.tasks import MultipleBinaryClassification
import pickle
from tqdm import tqdm
from pathlib import Path
class SKEMPIV2DatasetMulConf(Dataset):
    def __init__(self, data_df, is_train, knn_num, knn_agents_num):
        super(SKEMPIV2DatasetMulConf, self).__init__()

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
        pt_wt_G = data.Protein.from_pdb(
            PDB_wt_file_path, atom_feature=None, bond_feature="length", residue_feature="symbol")

        # 加载MulConf
        prefix = "seed-114514_sample-"
        WTConfDict = {}
        for i in range(5):
            pth = os.path.join(wt_file_path, prefix+str(i), "model.pdb")
            pt = data.Protein.from_pdb(
                pth, atom_feature=None, bond_feature="length", residue_feature="symbol")
            WTConfDict["Conf" + str(i)] = pt

        # 字符串转小写
        mt_name = wt_name+'_'+mutate_info.lower()
        mut_file_path = '/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_mutated/af3_output/'+mt_name
        PDB_mut_file_path = mut_file_path+f"/{mt_name}_model.pdb"
        pt_mt_G = data.Protein.from_pdb(
            PDB_mut_file_path, atom_feature=None, bond_feature="length", residue_feature="symbol")
        MTConfDict = {}
        for i in range(5):
            pth = os.path.join(mut_file_path, prefix+str(i), "model.pdb")
            pt = data.Protein.from_pdb(
                pth, atom_feature=None, bond_feature="length", residue_feature="symbol")
            MTConfDict["Conf" + str(i)] = pt

        # listProtMT=list_conf
        # PackedProteinMT = data.Protein.pack(list_conf)

        complex_wt_info = parse_pdb(PDB_wt_file_path)
        complex_mut_info = parse_pdb(PDB_mut_file_path)

        transform = KnnResidue(num_neighbors=self.knn_num)
        agent_select = KnnAgnet(num_neighbors=self.knn_agents_num)

        if len(complex_wt_info['aa']) != len(complex_mut_info['aa']):
            print(sample_info)
            return 0

        mutation_mask = (complex_wt_info['aa'] != complex_mut_info['aa'])

        agent_mask = agent_select(
            {'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})

        complex_wt_info['agent_mask'] = agent_mask
        complex_wt_info['PDB_id'] = PDB_id
        complex_wt_info['mutate_info'] = mutate_info

        complex_mut_info['agent_mask'] = agent_mask
        complex_mut_info['PDB_id'] = PDB_id
        complex_mut_info['mutate_info'] = mutate_info

        # batch = transform({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})
        batch = {'wt': complex_wt_info, 'mut': complex_mut_info,
                 'mutation_mask': mutation_mask}
        batch['ddG'] = ddG
        pt_mt_G.view = "residue"
        pt_wt_G.view = "residue"
        batch['wtG'] = pt_wt_G
        batch['mtG'] = pt_mt_G

        batch['wtDict'] = WTConfDict
        batch['mtDict'] = MTConfDict
        return batch


class SKEMPIV2DatasetMulConfLazy(Dataset):
    def __init__(self, data_df, is_train, knn_num, knn_agents_num):
        super(SKEMPIV2DatasetMulConfLazy, self).__init__()
        self.data_df = data_df
        self.data_batches = len(data_df['PDB_id'])
        self.is_train = is_train
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num
        # uneffective ! it can be 44G ,so large~
        # self.pklPath =  os.path.join(
        #     os.path.dirname(self.path), f'{self.name}_preprocessed_data.pkl')
        # if not os.path.exists(self.pklPath):
        #     self.preprocess()
        # else:
        #     print("---------------Loading Dataset--------------")
        #     with open(self.pklPath, 'rb') as f:
        #         self.preprocessed_data = pickle.load(f)
        #     print("---------------Dataset Loaded--------------")
        # self.Lazy=True

    def __len__(self):
        return self.data_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    # def preprocess(self):
    #     preprocessed_data = []
    #     for idx in tqdm(range(len(self.data_df)), desc="Preprocessing data"):
    #         sample_info = self.__Mygetitem__(idx)
    #         preprocessed_data.append(sample_info)
    #     with open(self.pklPath, 'wb') as f:
    #         pickle.dump(preprocessed_data, f)

    def __getitem__(self, index, save=False):
        index = index % self.data_batches
        sample_info = self.data_df.iloc[index].values
        PDB_id, chain1, chain2, mutate_info, _, ddG = sample_info

        PDB_id = PDB_id.replace('+', '')
        PDB_id = PDB_id.replace('.00', '')

        mutate_info = mutate_info.replace(',', '_')

        wt_name = str(PDB_id).lower()
        wt_file_path = '/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_output/'+wt_name

        wt_pkl_file_path = wt_file_path+f"/{wt_name}_model.pkl"
        PDB_wt_file_path = wt_file_path+f"/{wt_name}_model.pdb"
        if os.path.exists(wt_pkl_file_path):
            # 加载pkl
            with open(wt_pkl_file_path, 'rb') as f:
                pt_wt_G = pickle.load(f)
            # pass
        else:
            
            pt_wt_G = data.Protein.from_pdb(
                PDB_wt_file_path, atom_feature=None, bond_feature="length", residue_feature="symbol")
            # 存到pkl
            with open(wt_pkl_file_path, 'wb') as f:
                pickle.dump(pt_wt_G, f)
        
        wtConf_pkl_file_path = wt_file_path+f"/{wt_name}_MulConf.pkl"
        if os.path.exists(wtConf_pkl_file_path):
            with open(wtConf_pkl_file_path, 'rb') as f:
                WTConfDict = pickle.load(f)
            # pass
        else:
            # 加载MulConf
            prefix = "seed-114514_sample-"
            WTConfDict = {}
            for i in range(5):
                pth = os.path.join(wt_file_path, prefix+str(i), "model.pdb")
                pt = data.Protein.from_pdb(
                    pth, atom_feature=None, bond_feature="length", residue_feature="symbol")
                WTConfDict["Conf" + str(i)] = pt
            with open(wtConf_pkl_file_path, 'wb') as f:
                pickle.dump(WTConfDict, f)

        # 字符串转小写
        mt_name = wt_name+'_'+mutate_info.lower()
        mut_file_path = '/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_mutated/af3_output/'+mt_name
        
        
        mut_pkl_file_path = mut_file_path+f"/{mt_name}_model.pkl"
        PDB_mut_file_path = mut_file_path+f"/{mt_name}_model.pdb"
        if os.path.exists(mut_pkl_file_path):
            # 加载pkl
            with open(mut_pkl_file_path, 'rb') as f:
                pt_mt_G = pickle.load(f)
            # pass
        else:
            
            pt_mt_G = data.Protein.from_pdb(
            PDB_mut_file_path, atom_feature=None, bond_feature="length", residue_feature="symbol")
            # 存到pkl
            with open(mut_pkl_file_path, 'wb') as f:
                pickle.dump(pt_mt_G, f)
        
        mtConf_pkl_file_path = mut_file_path+f"/{mt_name}_MulConf.pkl"
        if os.path.exists(mtConf_pkl_file_path):
            with open(mtConf_pkl_file_path, 'rb') as f:
                MTConfDict = pickle.load(f)
            # pass
        else:
            # 加载MulConf
            prefix = "seed-114514_sample-"
            MTConfDict = {}
            for i in range(5):
                pth = os.path.join(mut_file_path, prefix+str(i), "model.pdb")
                pt = data.Protein.from_pdb(
                    pth, atom_feature=None, bond_feature="length", residue_feature="symbol")
                MTConfDict["Conf" + str(i)] = pt
            with open(mtConf_pkl_file_path, 'wb') as f:
                pickle.dump(MTConfDict, f)

        complex_wt_info = parse_pdb(PDB_wt_file_path)
        complex_mut_info = parse_pdb(PDB_mut_file_path)

        transform = KnnResidue(num_neighbors=self.knn_num)
        agent_select = KnnAgnet(num_neighbors=self.knn_agents_num)

        if len(complex_wt_info['aa']) != len(complex_mut_info['aa']):
            print(sample_info)
            return 0

        mutation_mask = (complex_wt_info['aa'] != complex_mut_info['aa'])

        agent_mask = agent_select(
            {'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})

        complex_wt_info['agent_mask'] = agent_mask
        complex_wt_info['PDB_id'] = PDB_id
        complex_wt_info['mutate_info'] = mutate_info

        complex_mut_info['agent_mask'] = agent_mask
        complex_mut_info['PDB_id'] = PDB_id
        complex_mut_info['mutate_info'] = mutate_info

        # batch = transform({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})
        batch = {'wt': complex_wt_info, 'mut': complex_mut_info,
                 'mutation_mask': mutation_mask}
        batch['ddG'] = ddG
        pt_mt_G.view = "residue"
        pt_wt_G.view = "residue"
        batch['wtG'] = pt_wt_G
        batch['mtG'] = pt_mt_G

        batch['wtDict'] = WTConfDict
        batch['mtDict'] = MTConfDict
        # print(wt_name, mt_name)
        return batch

    # def __getitem__(self, index):
    #     if self.Lazy:
    #         return self.preprocessed_data[index]
    #     else:
    #         return self.__Mygetitem__(index)
