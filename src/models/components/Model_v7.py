import torch.nn.functional as F
import torch
import torch.nn as nn
from torchdrug import models
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
import numpy as np
#somtimes import会有错误的警告 reload window即可
from openfold.model.evoformer import PairStack

#* 多层attention + residue block
class Model_v7(nn.Module):
    def __init__(self,mode):
        super(Model_v7, self).__init__()
        # self.gearnet = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512], num_relation=7,
        #                                 batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")
        # caution
        # 对边的种类的独热编码甚至是动态的？7种边需要7个维度 5种边需要5个维度
        # 21+21+7+1=
        
        self.gearnet = models.GearNet(input_dim=21, hidden_dims=[512],
                       num_relation=5, edge_input_dim=57, num_angle_bin=8,
                       batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")   
        # 512*1*2=1024
        self.predicter = nn.Sequential(
            # nn.Linear(3072, 2048),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # self.predicter = nn.Sequential(
        #     nn.Linear(1536, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1),
        # )
        self.mode=mode
        # self.diasble_qk=diasble_qk
        # 512*3=1536
        # self.attnModel = StackedAttention(4,512, 8)
        
        self.stacked_pairstack = StackedPairStack(4,512, 4)

    def forward(self, dual,wtG, mtG, mask, cov_wt_tensor, cov_mut_tensor,wt_mask_roi,mut_mask_roi,debug_name="fuck"):
        d1 = self.gearnet(wtG, wtG.node_feature.float())
        d2 = self.gearnet(mtG, mtG.node_feature.float())
        node_wt = d1["node_feature"]
        node_mut = d2["node_feature"]

        wt_embedding = self.stacked_pairstack(node_wt, mask, cov_wt_tensor,wt_mask_roi,debug_name)
        mt_embedding = self.stacked_pairstack(node_mut, mask,cov_mut_tensor,mut_mask_roi,debug_name)
        if dual is False:
            res1 = self.predicter(torch.cat([wt_embedding, mt_embedding], dim=1))
            return res1
        else:
            res1 = self.predicter(torch.cat([wt_embedding, mt_embedding], dim=1))
            res2 = self.predicter(torch.cat([mt_embedding, wt_embedding], dim=1))
            return res1, res2
        # if dual is False:
        #     res1 = self.predicter(mt_embedding-wt_embedding)
        #     return res1
        # else:
        #     res1 = self.predicter(mt_embedding-wt_embedding)
        #     res2 = self.predicter(wt_embedding-mt_embedding)
        #     return res1, res2

class StackedPairStack(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads):
        super().__init__()
        
        self.repr_outdim=32
        self.repr_linear = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.repr_outdim),
        )
        
        num_layers=1
        c_z=26+self.repr_outdim*2
        c_hidden_mul=32
        c_hidden_pair_att=16
        no_heads_pair=4
        transition_n=4
        pair_dropout=0.25
        fuse_projection_weights=False
        eps=1e-10
        inf=1e9
        self.layers = nn.ModuleList([
            PairStack(c_z, 
                      c_hidden_mul, 
                      c_hidden_pair_att, 
                      no_heads_pair, 
                      transition_n, 
                      pair_dropout, 
                      fuse_projection_weights, 
                      eps, inf) for _ in range(num_layers)
        ])
        
        self.att_linear=nn.Linear(c_z,1)
        self.val_linear=nn.Linear(c_z,64)
    def process_node_feature(self, ts,msk):
        _, h = ts.shape
        B, L = msk.shape
        output = torch.zeros(B, L, h)
        start = 0
        for b in range(B):
            len = msk[b].sum()
            output[b, :len] = ts[start:start+len]
            start += len
        return output
    
    def forward(self, x, mask=None, repr_roi=None,mask_roi=None,debug_name="fuck"):
        input=self.process_node_feature(x,mask).to(x.device)

        B=input.size(0)
        roi_max_len= repr_roi.shape[1]
        selb_list=[]
        for b in range(B):
            inb=input[b]
            selb=inb[mask_roi[b]==1]
            selb=torch.nn.functional.pad(selb,pad=(0,0,0,roi_max_len-selb.shape[0]), mode='constant',value=-1)
            selb_list.append(selb)
        selb_ten=torch.stack(selb_list)
        
        selb_ten=self.repr_linear(selb_ten)
        
        selb_ten_i=selb_ten.unsqueeze(2).expand(B, roi_max_len, roi_max_len,self.repr_outdim)
        selb_ten_j=selb_ten.unsqueeze(1).expand(B, roi_max_len, roi_max_len,self.repr_outdim)
        selb_ij=torch.cat([selb_ten_i,selb_ten_j],dim=-1)
        
        
        pair=torch.cat([repr_roi,selb_ij],dim=-1)
        # print(f"debugname:{debug_name} pair.shape:{pair.shape} device:{pair.device}")
        #bool to 01
                
        mymask = ~(repr_roi == -1).all(dim=-1) 
        mymask=mymask.float()
        for layer in self.layers:
            # print(pair[0,0,0])
            pair = layer(pair, mymask)
            # print(pair[0,0,0])
        
        att=self.att_linear(pair)
        mask1d=mymask[:,0]
        #*由于之前写的是多头的mask 所以要处理一下
        att=self._mask_by_1dmask(att.squeeze(-1).unsqueeze(1),mask1d)
        att=att.squeeze(1)
        val=self.val_linear(pair.sum(dim=-2))
        
        weights = F.softmax(att, dim=-1)
        # attention output
        output = torch.matmul(weights, val)
        # input = self.final_norm(input)

        # 只取第一个 token 的表示（例如用于分类任务的 CLS token）
        # return input[:, 0, :]
        #* 按mask取sum?
        
        output3 = (output*mask1d.unsqueeze(-1)).mean(dim=1)
        return output3
    def _mask_by_1dmask(self, x, mask):
        """
        :param x: 输入张量 (B, L, h)
        :param mask: 掩蔽张量 (B, L)
        :return: 输出张量 (B, L, h)
        """
        #如果mask不是bool 则转换为bool
        if mask.dtype != torch.bool:
            mask = mask.bool()
        mask_2d = mask.unsqueeze(2) & mask.unsqueeze(1)
        mask_2d = mask_2d.bool()
        masked_x = x.masked_fill(
            ~mask_2d.unsqueeze(1), float('-inf'))
        all_inf_mask = masked_x == float('-inf')  # 记录 `-inf` 位置
        all_inf_rows = all_inf_mask.all(
            dim=-1, keepdim=True)  # 记录哪些行全是 `-inf`
        masked_x[all_inf_rows.expand_as(x)] = 0
        return masked_x


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads

        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)
        
        my_repr_indim=53
        my_repr_outdim=self.num_heads
        #53-32-8
        self.repr_linear = nn.Sequential(
            nn.Linear(my_repr_indim, 32),
            nn.ReLU(),
            nn.Linear(32, my_repr_outdim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None, refCov=None):
        B, L, h = x.size()
        residual = x

        # Q, K, V projection
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)

        query = query.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        # classic attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention_scores = self._mask_by_1dmask(attention_scores, mask)
        attn_weights = F.softmax(attention_scores, dim=-1)

        # refCov attention scores (if provided)
        if refCov is not None:
            # refCov = self._non_smooth_abs(refCov).unsqueeze(1)  # (B, 1, L, L)
            # refCov = self._mask_by_1dmask(refCov, mask)
            ref_weights=self.repr_linear(refCov)
            #B,L,L,8 to B,8,L,L
            ref_weights = ref_weights.permute(0, 3, 1, 2)
            ref_weights=self._mask_by_1dmask(ref_weights,mask)
            ref_weights = F.softmax(ref_weights, dim=-1)
            
            #注册钩子查看梯度
            # ref_weights.register_hook(lambda grad: print("ref_weights Gradient:", grad))
            # attn_weights.register_hook(lambda grad: print("attn_weights Gradient:", grad))
            # element-wise sum
            combined_weights = attn_weights + ref_weights
            # normalize again
            combined_weights = combined_weights / combined_weights.sum(dim=-1, keepdim=True)
        else:
            combined_weights = attn_weights

        # attention output
        output = torch.matmul(combined_weights, value)
        output = output.transpose(1, 2).contiguous().view(B, L, h)
        output = self.output_linear(output)

        return self.norm(residual + output)
    def _mask_by_1dmask(self, x, mask):
        """
        :param x: 输入张量 (B, L, h)
        :param mask: 掩蔽张量 (B, L)
        :return: 输出张量 (B, L, h)
        """
        mask_2d = mask.unsqueeze(2) & mask.unsqueeze(1)
        mask_2d = mask_2d.bool()
        masked_x = x.masked_fill(
            ~mask_2d.unsqueeze(1), float('-inf'))
        all_inf_mask = masked_x == float('-inf')  # 记录 `-inf` 位置
        all_inf_rows = all_inf_mask.all(
            dim=-1, keepdim=True)  # 记录哪些行全是 `-inf`
        masked_x[all_inf_rows.expand_as(x)] = 0
        return masked_x
    
class AttentionBlock_disable_qk(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads

        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)
        
        my_repr_indim=53
        my_repr_outdim=self.num_heads
        #53-32-8
        self.repr_linear = nn.Sequential(
            nn.Linear(my_repr_indim, 32),
            nn.ReLU(),
            nn.Linear(32, my_repr_outdim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None, refCov=None):
        B, L, h = x.size()
        residual = x

        # Q, K, V projection

        value = self.value_linear(x)


        value = value.view(B, L, self.num_heads, self.d_k).transpose(1, 2)


        ref_weights=self.repr_linear(refCov)
        #B,L,L,8 to B,8,L,L
        ref_weights = ref_weights.permute(0, 3, 1, 2)
        ref_weights=self._mask_by_1dmask(ref_weights,mask)
        ref_weights = F.softmax(ref_weights, dim=-1)
        # attention output
        output = torch.matmul(ref_weights, value)
        output = output.transpose(1, 2).contiguous().view(B, L, h)
        output = self.output_linear(output)

        return self.norm(residual + output)
    def _mask_by_1dmask(self, x, mask):
        """
        :param x: 输入张量 (B, L, h)
        :param mask: 掩蔽张量 (B, L)
        :return: 输出张量 (B, L, h)
        """
        mask_2d = mask.unsqueeze(2) & mask.unsqueeze(1)
        mask_2d = mask_2d.bool()
        masked_x = x.masked_fill(
            ~mask_2d.unsqueeze(1), float('-inf'))
        all_inf_mask = masked_x == float('-inf')  # 记录 `-inf` 位置
        all_inf_rows = all_inf_mask.all(
            dim=-1, keepdim=True)  # 记录哪些行全是 `-inf`
        masked_x[all_inf_rows.expand_as(x)] = 0
        return masked_x

class StackedAttention(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads):
        super().__init__()
        self.diable_qk=False
        self.disable_ref=False
        if not self.diable_qk and not self.disable_ref:
            self.layers = nn.ModuleList([
                AttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)
            ])
        elif self.diable_qk and not self.disable_ref:
            self.layers = nn.ModuleList([
                AttentionBlock_disable_qk(hidden_dim, num_heads) for _ in range(num_layers)
            ])
        elif not self.diable_qk and self.disable_ref:
            #* 不用重写模块 控制输入即可
            #* 好吧需要把repr_linear注释掉 否则lightning会提醒
            
            self.layers = nn.ModuleList([
                AttentionBlock(hidden_dim, num_heads) for _ in range(num_layers)
            ])
        else:
            raise ValueError("Invalid configuration")
        self.final_norm = nn.LayerNorm(hidden_dim)
    def process_node_feature(self, ts,msk):
        _, h = ts.shape
        B, L = msk.shape
        output = torch.zeros(B, L, h)
        start = 0
        for b in range(B):
            len = msk[b].sum()
            output[b, :len] = ts[start:start+len]
            start += len
        return output
    
    def forward(self, x, mask=None, refCov=None):
        input=self.process_node_feature(x,mask).to(x.device)
        for layer in self.layers:
            if self.disable_ref:
                input = layer(input, mask,None)
            else:
                input = layer(input, mask, refCov)
        input = self.final_norm(input)

        # 只取第一个 token 的表示（例如用于分类任务的 CLS token）
        # return input[:, 0, :]
        #* 按mask取sum?
        output3 = (input*mask.unsqueeze(-1)).sum(dim=1)
        return output3



