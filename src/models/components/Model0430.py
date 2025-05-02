import torch.nn.functional as F
import torch
import torch.nn as nn
from torchdrug import models
from .conan_fgw.conan_fgw.src.model.fgw.barycenter import fgw_barycenters, normalize_tensor
from .conan_fgw.conan_fgw.src.model.graph_embeddings.schnet_no_sum import get_list_node_features_batch, get_adj_dense_batch
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
import numpy as np

class Model_w_attention_v2(nn.Module):
    def __init__(self,mode):
        super(Model_w_attention_v2, self).__init__()
        # self.gearnet = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512], num_relation=7,
        #                                 batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")
        # caution
        # 对边的种类的独热编码甚至是动态的？7种边需要7个维度 5种边需要5个维度
        # 21+21+7+1=
        
        self.gearnet = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512],
                       num_relation=5, edge_input_dim=57, num_angle_bin=8,
                       batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")
        # 512*3*2=3072
        # self.predicter = nn.Sequential(
        #     nn.Linear(3072, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1),
        # )
        self.predicter = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.mode=mode
        # 512*3=1536
        self.attnModel = ScaledDotAttention(1536, 8,self.mode)

    def forward(self, dual,wtG, mtG, mask, cov_wt_tensor, cov_mut_tensor):
        d1 = self.gearnet(wtG, wtG.node_feature.float())
        d2 = self.gearnet(mtG, mtG.node_feature.float())
        node_wt = d1["node_feature"]
        node_mut = d2["node_feature"]

        wt_embedding = self.attnModel(node_wt, mask, cov_wt_tensor)
        mt_embedding = self.attnModel(node_mut, mask,cov_mut_tensor)
        # if dual is False:
        #     res1 = self.predicter(torch.cat([wt_embedding, mt_embedding], dim=1))
        #     return res1
        # else:
        #     res1 = self.predicter(torch.cat([wt_embedding, mt_embedding], dim=1))
        #     res2 = self.predicter(torch.cat([mt_embedding, wt_embedding], dim=1))
        #     return res1, res2
        if dual is False:
            res1 = self.predicter(mt_embedding-wt_embedding)
            return res1
        else:
            res1 = self.predicter(mt_embedding-wt_embedding)
            res2 = self.predicter(wt_embedding-mt_embedding)
            return res1, res2


class ScaledDotAttention(torch.nn.Module):
    def __init__(self, h, num_heads,mode):
        super(ScaledDotAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = h // num_heads  # 每个头的维度

        # 定义Q, K, V的线性变换
        self.query_linear = torch.nn.Linear(h, h)
        # self.query_linear.weight.register_hook(lambda grad: print("query_linear weight Gradient:", grad))
        # self.query_linear.bias.register_hook(lambda grad: print("query_linear bias Gradient:", grad))

        self.key_linear = torch.nn.Linear(h, h)
        self.value_linear = torch.nn.Linear(h, h)
        self.output_linear = torch.nn.Linear(h, h)
        
        self.mode = mode
        if self.mode==2:
            # self.balance_factor = nn.Parameter(torch.tensor(0.8))
            self.balance_factor = torch.tensor(0.8)
        
        # self.balance_factor.register_hook(
        #     lambda grad: print("balance factor Gradient:", grad))
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

    def _smooth_abs(self,x, epsilon=1e-6):
        return torch.sqrt(x**2 + epsilon)
    def _non_smooth_abs(self,x):
        return torch.abs(x)   
    def forward(self, x, mask=None,refCov=None):
        """
        :param x: 输入张量 (B, L, h)
        :param mask: 掩蔽张量 (B, L, L) (可选)
        :return: output, attention_weights
        """
        # todo:似乎是否基于cpu/gpu来做切片并不影响结果？
        input=self.process_node_feature(x,mask).to(x.device)
        B, L, h = input.size()
        attention_scores, value = self.classic_attn(
            input, mask)  # (B, num_heads, L, L)

        # if self.mode==2:
        #     refCov=self._smooth_abs(refCov)
        #     # cov_norm = self._norm_by_1dmask(refCov.unsqueeze(1), mask)
            
        #     # att_scores = att_norm+self.balance_factor*cov_norm
        #     # att_scores = attention_scores + self.balance_factor * refCov.unsqueeze(1)
        #     att_scores = (1-self.balance_factor)*attention_scores + \
        #         self.balance_factor * refCov.unsqueeze(1)
        # else:
        #     att_scores = attention_scores
        #     # pass
        refCov=self._non_smooth_abs(refCov)
        refCov=refCov.unsqueeze(1)
        scores_list = [attention_scores,refCov]
        output_list = []
        for att_scores in scores_list:
            att_scores_masked=self._mask_by_1dmask(att_scores,mask)
            attention_weights = F.softmax(
                att_scores_masked, dim=-1)  # (B, num_heads, L, L)

            # Step 6: 计算加权和输出
            # (B, num_heads, L, d_k)
            output = torch.matmul(attention_weights, value)
            output2 = output.transpose(1, 2).contiguous().view(B, L, h)  # (B, L, h)
            output2 = self.output_linear(output2)  # (B, L, h)
            
            output3=output2[:,0,:]
            output_list.append(output3)
        #相加返回
        return torch.sum(torch.stack(output_list),dim=0)
    def classic_attn(self,x,mask):
        B, L, h = x.size()

        # Step 1: 通过线性层计算Q, K, V
        query = self.query_linear(x)  # (B, L, h)
        key = self.key_linear(x)      # (B, L, h)
        value = self.value_linear(x)  # (B, L, h)

        # Step 2: 重塑为多头注意力的形状
        query = query.view(B, L, self.num_heads, self.d_k).transpose(
            1, 2)  # (B, num_heads, L, d_k)
        key = key.view(B, L, self.num_heads, self.d_k).transpose(
            1, 2)      # (B, num_heads, L, d_k)
        value = value.view(B, L, self.num_heads, self.d_k).transpose(
            1, 2)  # (B, num_heads, L, d_k)

        # Step 3: 计算缩放点积注意力
        attention_scores = torch.matmul(
            query, key.transpose(-2, -1))  # (B, num_heads, L, L)
        attention_scores = attention_scores / (self.d_k ** 0.5)  # 缩放

        return attention_scores, value
    
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

    def _norm_by_1dmask(self,tensors, masks):
        mask2d = masks.unsqueeze(2) & masks.unsqueeze(1)  # (B, H, W)

        # 扩展 mask2d，使其适用于所有通道
        mask2d_expanded = mask2d.unsqueeze(1)  # (B, 1, H, W)

        # 计算 mask 选中的元素个数 (B, 1, 1, 1)，确保能广播
        num_elements = mask2d_expanded.sum(dim=(-1, -2), keepdim=True).clamp(min=1)

        # 计算均值
        sum_values = (tensors * mask2d_expanded).sum(dim=(-1, -2), keepdim=True)
        mean_val = sum_values / num_elements  # (B, C, 1, 1)

        # 计算标准差
        sum_squared_diff = ((tensors - mean_val) *
                            mask2d_expanded).pow(2).sum(dim=(-1, -2), keepdim=True)
        std_val = (sum_squared_diff / num_elements).sqrt().clamp(min=1e-6)  # 避免除零

        # 归一化
        tensors = torch.where(
            mask2d_expanded, (tensors - mean_val) / std_val, tensors)
        return tensors





