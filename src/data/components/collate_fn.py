from collections import deque
from collections.abc import Mapping, Sequence
# from .protein.read_pdbs import PaddingCollate
import torch
import math
from torchdrug import data
# def mycollate_fn(batch):
#     return batch

# From torchdrug
def graph_collate(batch):
    """
    Convert any list of same nested container into a container of tensors.

    For instances of :class:`data.Graph <torchdrug.data.Graph>`, they are collated
    by :meth:`data.Graph.pack <torchdrug.data.Graph.pack>`.

    Parameters:
        batch (list): list of samples with the same nested container
    """
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, data.Graph):
        return elem.pack(batch)
    elif isinstance(elem, Mapping):
        return {key: graph_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('Each element in list of batch should be of equal size')
        return [graph_collate(samples) for samples in zip(*batch)]

    raise TypeError("Can't collate data with type `%s`" % type(elem))

class PaddingCollate(object):

    def __init__(self, length_ref_key='mutation_mask', pad_values={'aa': 20, 'pos14': float('999'), 'icode': ' ', 'chain_id': '-','repr_within_pt':-1}, donot_pad={'foldx'}, eight=False):
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
    def _my_pad_cov(self, x, k, value=0):
        assert x.dim() == 2, "输入必须是二维张量"
        h, w = x.shape
        assert h <= k and w <= k, f"当前张量大小为 ({h},{w})，不能 pad 成 ({k},{k})"
        
        pad_h = k - h
        pad_w = k - w

        # pad 格式为 (left, right, top, bottom)，从最内层维度开始
        padded = torch.nn.functional.pad(x, pad=(0, pad_w, 0, pad_h), mode='constant', value=value)
        return padded
    def _my_pad_repr(self, x, k, value=0):
        assert x.dim() == 3, "输入必须是三维张量"
        h, w ,d= x.shape
        assert h <= k and w <= k, f"当前张量大小为 ({h},{w})，不能 pad 成 ({k},{k})"
        
        pad_h = k - h
        pad_w = k - w
        if pad_h==0 and pad_w==0:
            return x
        padded = torch.nn.functional.pad(x, pad=(0,0,0, pad_w, 0, pad_h), mode='constant', value=value)
        return padded
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
        if 'wt_mask_roi' in data_list[0].keys():
            wt_roi_max_length = max(
                [int(data['wt_mask_roi'].sum().item()) for data in data_list])
            mt_roi_max_length = max(
                [int(data['mut_mask_roi'].sum().item()) for data in data_list])
        # print(f"id: {data_list[0]['wt']['PDB_id']} wt_roi_max_length:{wt_roi_max_length}")
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:
            #*原始pad
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k))
                for k, v in data.items() if k in ('wt', 'mut', 'ddG', 'mutation_mask', 'index_info', 'mutation', 'expand_data_info')
            }
            #* 处理cov_tensor的二维padding
            my_data_padded = {k: self._my_pad_cov(v, max_length, value=self._get_pad_value(k))
                for k, v in data.items() if k in ('cov_wt_tensor', 'cov_mut_tensor')}
            data_padded.update(my_data_padded)
            #* 处理repr_tensor的三维padding
            my_data_padded = {k: self._my_pad_repr(v, max_length, value=-1)
                for k, v in data.items() if k in ('repr_wt_tensor','repr_mut_tensor')}
            data_padded.update(my_data_padded)

            my_data_padded = {"roi_repr_wt_tensor": self._my_pad_repr(data["roi_repr_wt_tensor"], wt_roi_max_length, value=-1),
                              "roi_repr_mut_tensor": self._my_pad_repr(data["roi_repr_mut_tensor"], mt_roi_max_length, value=-1),
                              "wt_mask_roi": self._pad_last(data["wt_mask_roi"], max_length, value=-1),
                              "mut_mask_roi": self._pad_last(data["mut_mask_roi"], max_length, value=-1)}
            
            data_padded.update(my_data_padded)
            
            #处理无需pad的key&val
            my_data_padded = {k: v
                              for k, v in data.items() if k in ('wtG', 'mtG', "wtDict", "mtDict")}
            data_padded.update(my_data_padded)

            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
            
        return graph_collate(data_list_padded)
mycollate_fn = PaddingCollate()