import numpy as np
import os
from torch.utils.data.dataset import Dataset
from .protein.read_pdbs import parse_pdb, KnnResidue, KnnAgnet
from torchdrug import data

import torch
 #* 版本更新 使用repr
 #* 0504 版本更新 residues of ionterest(roi)
class SKEMPIV2Dataset0504(Dataset):
    def __init__(self, data_root_path,data_df, is_train, knn_num, knn_agents_num,rand):
        super(SKEMPIV2Dataset0504, self).__init__()
        self.data_root_path=data_root_path
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
#-------------------wt load
        wt_name=str(PDB_id).lower()
        wt_root_path=os.path.join(self.data_root_path,"Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_sp2_output")
        wt_file_path=os.path.join(wt_root_path,f"{wt_name}_sp2_rand0.2/gather")

        if self.rand:
            # repr_wt_tensor=torch.load(os.path.join(wt_file_path,"rand_repr_within_pt_tensor.pt"))
            # raise NotImplementedError(
            #     "rand_repr_within_pt_tensor.pt not implemented")
            repr_wt_tensor = torch.load(
                os.path.join(wt_file_path, "rand_roi_tensor.pt"))
            wt_mask_roi = torch.load(os.path.join(wt_file_path, "rand_mask_roi.pt"))
        else:
            repr_wt_tensor = torch.load(
                os.path.join(wt_file_path, "roi_tensor.pt"))
            wt_mask_roi = torch.load(os.path.join(wt_file_path, "mask_roi.pt"))

        PDB_wt_file_path=os.path.join(wt_file_path,"af_ref.pdb")
        pt_wt_G=data.Protein.from_pdb(PDB_wt_file_path, atom_feature=None, bond_feature="length", residue_feature="symbol")
#-------------------mut load
        mt_name=wt_name+'_'+mutate_info.lower()
        mut_root_path=os.path.join(self.data_root_path,"Antibody_Mutation/data/SKEMPIv2/PDBs_mutated/af3_sp2_output")
        mut_file_path=os.path.join(mut_root_path,f"{mt_name}_sp2_rand0.2/gather")
        
        if self.rand:
            # repr_mut_tensor=torch.load(os.path.join(mut_file_path,"rand_repr_within_pt_tensor.pt"))
            # raise NotImplementedError("rand_repr_within_pt_tensor.pt not implemented")
            repr_mut_tensor = torch.load(
                os.path.join(mut_file_path, "rand_roi_tensor.pt"))
            mut_mask_roi = torch.load(os.path.join(mut_file_path, "rand_mask_roi.pt"))
        else:
            repr_mut_tensor=torch.load(os.path.join(mut_file_path,"roi_tensor.pt"))
            mut_mask_roi=torch.load(os.path.join(mut_file_path,"mask_roi.pt"))

        PDB_mut_file_path=os.path.join(mut_file_path,"ref.pdb")
        pt_mt_G=data.Protein.from_pdb(PDB_mut_file_path, atom_feature=None, bond_feature="length", residue_feature="symbol")
#------------------------pass to item
        complex_wt_info = parse_pdb(PDB_wt_file_path)
        complex_mut_info = parse_pdb(PDB_mut_file_path)

        # transform = KnnResidue(num_neighbors=self.knn_num)
        # agent_select = KnnAgnet(num_neighbors=self.knn_agents_num)

        if len(complex_wt_info['aa']) != len(complex_mut_info['aa']):
            print(sample_info)
            return 0

        mutation_mask = (complex_wt_info['aa'] != complex_mut_info['aa'])

        # agent_mask = agent_select({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})

        # complex_wt_info['agent_mask'] = agent_mask
        complex_wt_info['PDB_id'] = PDB_id
        complex_wt_info['mutate_info'] = mutate_info

        # complex_mut_info['agent_mask'] = agent_mask
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
        batch['roi_repr_wt_tensor'] = repr_wt_tensor
        batch['roi_repr_mut_tensor']=repr_mut_tensor
        batch['wt_mask_roi']=wt_mask_roi
        batch['mut_mask_roi']=mut_mask_roi
        
        return batch


class SKEMPIV2Dataset0504TST(Dataset):
    def __init__(self, data_df, is_train, knn_num, knn_agents_num, rand):
        super(SKEMPIV2Dataset0504TST, self).__init__()

        self.data_df = data_df
        self.data_batches = len(data_df['PDB_id'])
        self.is_train = is_train
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num
        self.rand = rand
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
# -------------------wt load
        wt_name = str(PDB_id).lower()
        wt_file_path = "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_fixed/af3_sp2_output/" + \
            f"{wt_name}_sp2_rand0.2/gather"

        if self.rand:
            # repr_wt_tensor=torch.load(os.path.join(wt_file_path,"rand_repr_within_pt_tensor.pt"))
            raise NotImplementedError(
                "rand_repr_within_pt_tensor.pt not implemented")
        else:
            repr_wt_tensor = torch.load(
                os.path.join(wt_file_path, "roi_tensor.pt"))
        wt_mask_roi = torch.load(os.path.join(wt_file_path, "mask_roi.pt"))

        PDB_wt_file_path = os.path.join(wt_file_path, "af_ref.pdb")
        # pt_wt_G = data.Protein.from_pdb(
        #     PDB_wt_file_path, atom_feature=None, bond_feature="length", residue_feature="symbol")
# -------------------mut load
        mt_name = wt_name+'_'+mutate_info.lower()
        mut_file_path = "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/PDBs_mutated/af3_sp2_output/" + \
            f"{mt_name}_sp2_rand0.2/gather"

        if self.rand:
            # repr_mut_tensor=torch.load(os.path.join(mut_file_path,"rand_repr_within_pt_tensor.pt"))
            raise NotImplementedError(
                "rand_repr_within_pt_tensor.pt not implemented")
        else:
            repr_mut_tensor = torch.load(
                os.path.join(mut_file_path, "roi_tensor.pt"))
        mut_mask_roi = torch.load(os.path.join(mut_file_path, "mask_roi.pt"))

        PDB_mut_file_path = os.path.join(mut_file_path, "ref.pdb")
        # pt_mt_G = data.Protein.from_pdb(
        #     PDB_mut_file_path, atom_feature=None, bond_feature="length", residue_feature="symbol")
# ------------------------pass to item
        # complex_wt_info = parse_pdb(PDB_wt_file_path)
        # complex_mut_info = parse_pdb(PDB_mut_file_path)
        complex_wt_info = {}
        complex_mut_info = {}

        # transform = KnnResidue(num_neighbors=self.knn_num)
        # agent_select = KnnAgnet(num_neighbors=self.knn_agents_num)

        # if len(complex_wt_info['aa']) != len(complex_mut_info['aa']):
        #     print(sample_info)
        #     return 0

        # mutation_mask = (complex_wt_info['aa'] != complex_mut_info['aa'])

        # agent_mask = agent_select({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})

        # complex_wt_info['agent_mask'] = agent_mask
        complex_wt_info['PDB_id'] = PDB_id
        complex_wt_info['mutate_info'] = mutate_info

        # complex_mut_info['agent_mask'] = agent_mask
        complex_mut_info['PDB_id'] = PDB_id
        complex_mut_info['mutate_info'] = mutate_info

        # batch = transform({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})
        batch = {'wt': complex_wt_info, 'mut': complex_mut_info,}
        batch['ddG'] = ddG
        # pt_mt_G.view = "residue"
        # pt_wt_G.view = "residue"
        # batch['wtG'] = pt_wt_G
        # batch['mtG'] = pt_mt_G
        # 获取cov_tensor
        batch['roi_repr_wt_tensor'] = repr_wt_tensor
        batch['roi_repr_mut_tensor'] = repr_mut_tensor
        batch['wt_mask_roi'] = wt_mask_roi
        batch['mut_mask_roi'] = mut_mask_roi

        return batch

