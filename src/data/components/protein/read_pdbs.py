import warnings
import math

import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio import BiopythonWarning
from Bio.PDB import Selection

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.utils.data._utils.collate import default_collate

from .protein_attribute import aa_codes, RESIDUE_SIDECHAIN_POSTFIXES, \
    augmented_is_aa, augmented_three_to_index, augmented_three_to_one, ATOM_CA
# from protein_attribute import aa_codes, RESIDUE_SIDECHAIN_POSTFIXES, \
#     augmented_is_aa, augmented_three_to_index, augmented_three_to_one, ATOM_CA


def read_pdb_3D(pdb_file_path, chain_index_list=None, is_backbone=False):
    """
    read PDB file 3D locations using readline
    :param pdb_file_path:
    :param chain_index_list:
    :param is_backbone: whether only extract backbone(bb) atoms
    :return:
    """
    store_all = []
    with open(pdb_file_path) as protein:
        for lines in protein:
            if 'ATOM' in lines:
                # todo: if is_backbone, only bb atoms
                lines = lines.split()
                store_all.append(list(map(float, lines[6:9])))
    return store_all


def show_pdb(store_all):
    """
    show pdb 3D image (but can not run orz)
    :param store_all:
    :return:
    """

    '''
    x, y, z = zip(*store_all)
    fig = plt.Figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.plot(x, y, z)
    plt.show()
    '''


def read_pdb_seq(pdb_file_path):
    """
    read PDB file seq using readline
    :param pdb_file_path:
    :return: seq for the file
    todo: add chain_index
    """
    seq = ''
    seq_index = 0
    with open(pdb_file_path) as protein:
        for lines in protein:
            if 'ATOM' in lines:
                lines = lines.split()
                aa_index = lines[5]
                if aa_index == seq_index:
                    continue
                seq_index = aa_index
                seq = seq + aa_codes[lines[3]]  # 乐 这tm是读入原子序列
    return seq


def read_pdb_Biopython(pdb_file_path):
    """
    test for Biopython, no use
    :param pdb_file_path:
    :return:
    """
    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    structure = parser.get_structure("1CBW", pdb_file_path)
    print(structure.child_dict)
    model = structure[0]
    print(model.child_dict)
    chain_F = model["F"]
    print(chain_F.child_dict)
    res = chain_F[(' ', 1, ' ')]
    print(res.child_dict)
    atom = res['N']


def get_atom_name_postfix(atom):
    """
    get the postfix（后缀） of an atom
    :param atom: an atom
    :return: str, postfix
    """
    name = atom.get_name()
    if name in ('N', 'CA', 'C', 'O'):
        return name
    if name[-1].isnumeric():
        return name[-2:]
    else:
        return name[-1:]


def get_residue_pos14(res):
    """
    extract the 3D locations of atoms in a res in a specific, ordered by postfix
    why pos14? the max num of atoms in a res (4 bb + 10 sidechain, no H), Inf for not used positions
    :param res: a residue
    :return: a tensor, noting the atoms location in a res in a specific, ordered by postfix
    """
    pos14 = torch.full([14, 3], float('inf'))
    suffix_to_atom = {get_atom_name_postfix(a):a for a in res.get_atoms()}  # {postfix(suffix): atom}
    atom_order = ['N', 'CA', 'C', 'O'] + RESIDUE_SIDECHAIN_POSTFIXES[augmented_three_to_one(res.get_resname())]  # get specific ordering for a res
    for i, atom_suffix in enumerate(atom_order):
        if atom_suffix not in suffix_to_atom: continue
        pos14[i,0], pos14[i,1], pos14[i,2] = suffix_to_atom[atom_suffix].get_coord().tolist()
    return pos14


def parse_complex(structure0, model_id=None):
    """
    extract info from pdb complex, chain, aa, atom....
    :param structure:
    :param model_id:
    :return:
    chain_id : firstly, list, chain id for a protein, such as ['A''A''A''A''A''A''C''C''C''D''D''D']
               then using ''.join to return str 'AAAAACCCCDDDDD'
    chain_seq: list, real aa index, such as 1111112222222, same structure as chain_id, just for filtered atoms
    pos14: list, ordered 3D coords of atoms in a res, 14 means max length, 99999 subs Inf
           then stack to length*14*3 tensor
    pos14_mask: list, is used? if not, False for Inf(99999)
                then stack to length*14*3 tensor
    resseq: list, res_id(middle in the res name), start from 1, maybe repeat
    seq: list, an index? never repeat, start from 1
    icode: list, unknown

    """
    header = structure0.header
    if model_id is not None:
        structure = structure0[model_id]
    chains = Selection.unfold_entities(structure, 'C')

    aa, resseq, icode, seq = [], [], [], []
    icode_len = []
    resseq_origin = []
    pos14, pos14_mask = [], []
    chain_id, chain_seq = [], []

    # 对蛋白质链进行排序
    chains_id = []
    for chain in chains:
        chains_id.append(ord(chain.id))

    chains_id = np.array(chains_id)
    idx_array = chains_id.argsort()

    for i in range(len(chains)):
        chain = chains[idx_array[i]]

    # for i, chain in enumerate(chains):
        seq_this = 0
        for res in chain:
            resname = res.get_resname()
            # filtering
            if not augmented_is_aa(resname): continue
            if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): continue

            # Chain
            chain_id.append(chain.get_id())
            chain_seq.append(i + 1)

            # Residue types
            restype = augmented_three_to_index(resname)
            aa.append(restype)

            # Atom coordinates
            pos14_this = get_residue_pos14(res)
            pos14_mask_this = pos14_this.isfinite()
            pos14.append(pos14_this.nan_to_num(posinf=99999))
            pos14_mask.append(pos14_mask_this)

            # Sequential number
            resseq_this = int(res.get_id()[1])
            icode_this = res.get_id()[2]

            # deal with the icode
            if icode_this == ' ':
                icode_len.append(1)
                resseq_origin.append(resseq_this)
            else:
                icode_len[-1] += 1
                resseq_origin.append(resseq_origin[-1])

            if seq_this == 0:
                seq_this = 1
            else:
                d_resseq = resseq_this - resseq[-1]
                if d_resseq == 0:
                    seq_this += 1
                else:
                    seq_this += d_resseq
            resseq.append(resseq_this)
            icode.append(icode_this)
            seq.append(seq_this)

    if len(aa) == 0:
        return None

    return {
        'name': structure.get_id(),

        # Chain
        'chain_id': ''.join(chain_id),
        'chain_seq': torch.LongTensor(chain_seq),

        # Sequence
        'aa': torch.LongTensor(aa),
        'resseq': torch.LongTensor(resseq),
        'resseq_origin':torch.LongTensor(resseq_origin),
        'icode': ''.join(icode),
        'seq': torch.LongTensor(seq),

        # Atom positions
        'pos14': torch.stack(pos14),
        'pos14_mask': torch.stack(pos14_mask),
    }


def parse_pdb(path, model_id=0):
    """
    An entrance for parse_complex, the handle for read a pdb file
    :param path:
    :param model_id:
    :return:
    """
    warnings.simplefilter('ignore', BiopythonWarning)
    parser = PDBParser()
    try:
        structure = parser.get_structure(None, path)
    except Exception as e:
        print("文件不存在！", path)
    return parse_complex(structure, model_id)


def _mask_list(l, mask):
    return [l[i] for i in range(len(l)) if mask[i]]


def _mask_string(s, mask):
    return ''.join([s[i] for i in range(len(s)) if mask[i]])


def _mask_dict_recursively(d, mask):
    """
    filter the input dict by the mask, same shape out
    :param d: dict, data of mut, wt and mask(mutation_mask)
    :param mask: dist mask
    :return: dict, data of mut, wt and mask(mutation_mask), just true in distance mask
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and v.size(0) == mask.size(0):
            out[k] = v[mask]
        elif isinstance(v, list) and len(v) == mask.size(0):
            out[k] = _mask_list(v, mask)
        elif isinstance(v, str) and len(v) == mask.size(0):
            out[k] = _mask_string(v, mask)
        elif isinstance(v, dict):
            out[k] = _mask_dict_recursively(v, mask)
        else:
            out[k] = v
    return out


class KnnResidue(object):

    def __init__(self, num_neighbors=128):
        super().__init__()
        self.num_neighbors = num_neighbors

    def __call__(self, data):
        """
        using CA distance to get knn nearest residues, and filter the input dict by the mask, same shape out
        :param data:
        :return:
        """
        pos_CA = data['wt']['pos14'][:, ATOM_CA]
        pos_CA_mut = pos_CA[data['mutation_mask']]
        diff = pos_CA_mut.view(1, -1, 3) - pos_CA.view(-1, 1, 3)  # 628 * mutate_num * 3
        dist = torch.linalg.norm(diff, dim=-1)  # 628 * mutate_num

        try:
            mask = torch.zeros([dist.size(0)], dtype=torch.bool)
            mask[ dist.min(dim=1)[0].argsort()[:self.num_neighbors] ] = True
        except IndexError as e:
            print(data)
            print('PDB_id is: ', data['wt']['PDB_id'])
            raise e

        return _mask_dict_recursively(data, mask)


class KnnAgnet(object):

    def __init__(self, num_neighbors=20):
        super().__init__()
        self.num_neighbors = num_neighbors

    def __call__(self, data):
        """
        using CA distance to get knn nearest agents, and filter the input dict by the mask, same shape out
        :param data:
        :return:
        """
        pos_CA = data['wt']['pos14'][:, ATOM_CA]
        pos_CA_mut = pos_CA[data['mutation_mask']]
        diff = pos_CA_mut.view(1, -1, 3) - pos_CA.view(-1, 1, 3)  # 628 * mutate_num * 3
        dist = torch.linalg.norm(diff, dim=-1)  # 628 * mutate_num

        try:
            mask = torch.zeros([dist.size(0)], dtype=torch.bool)
            mask[ dist.min(dim=1)[0].argsort()[:self.num_neighbors] ] = True
        except IndexError as e:
            print(data)
            print('PDB_id is: ', data['wt']['PDB_id'])
            print('info is: ', data['wt']['mutate_info'])
            raise e
        return mask


class PaddingCollate(object):

    def __init__(self, length_ref_key='mutation_mask', pad_values={'aa': 20, 'pos14': float('999'), 'icode': ' ', 'chain_id': '-'}, donot_pad={'foldx'}, eight=False):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.donot_pad = donot_pad
        self.eight = eight

    def _pad_last(self, x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        elif isinstance(x, str):
            if value == 0:  # Won't pad strings if not specified
                return x
            pad = value * (n - len(x))
            return x + pad
        elif isinstance(x, dict):
            padded = {}
            for k, v in x.items():
                if k in self.donot_pad:
                    padded[k] = v
                else:
                    padded[k] = self._pad_last(v, n, value=self._get_pad_value(k))
            return padded
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):

        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k))
                for k, v in data.items() if k in ('wt', 'mut', 'ddG', 'mutation_mask', 'index_info', 'mutation', 'expand_data_info')
            }
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)


if __name__ == '__main__':
    # pdb_file_path = '../data/SKEMPIv2/PDBs_filtered/1CBW.pdb'
    # pdb_file_path = "/home/lfj/projects_dir/LargeDatabase/PP/1bql.ent.pdb"
    # pdb_file_path = "/home/lfj/projects_dir/LargeDatabase/PP/6fe4.ent.pdb"
    pdb_file_path = "/home/lfj/projects_dir/LargeDatabase/PP/5xkj.ent.pdb"
    store_all = parse_pdb(pdb_file_path)
    print(read_pdb_seq(pdb_file_path))
    # show_pdb(store_all)

    read_pdb_Biopython(pdb_file_path)
