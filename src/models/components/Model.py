import torch.nn.functional as F
import torch
import torch.nn as nn
from torchdrug import models
from .conan_fgw.conan_fgw.src.model.fgw.barycenter import fgw_barycenters, normalize_tensor
from .conan_fgw.conan_fgw.src.model.graph_embeddings.schnet_no_sum import get_list_node_features_batch, get_adj_dense_batch
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
import numpy as np
#*deprecated
class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.gearnet = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512], num_relation=7,
                                      batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")
        # 512*3*2=3072
        self.predicter = nn.Sequential(
            nn.Linear(3072, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, wtG, mtG):

        d1 = self.gearnet(wtG, wtG.node_feature.float())
        # node_wt = d1["node_feature"]
        graph_wt = d1["graph_feature"]
        d2 = self.gearnet(mtG, mtG.node_feature.float())
        # node_mut = d2["node_feature"]
        graph_mut = d2["graph_feature"]
        res = self.predicter(torch.cat([graph_wt, graph_mut], dim=1))
        return res

    def forward_for_train(self, wtG, mtG):
        d1 = self.gearnet(wtG, wtG.node_feature.float())
        # node_wt = d1["node_feature"]
        graph_wt = d1["graph_feature"]
        d2 = self.gearnet(mtG, mtG.node_feature.float())
        # node_mut = d2["node_feature"]
        graph_mt = d2["graph_feature"]
        res1 = self.predicter(torch.cat([graph_wt, graph_mt], dim=1))
        res2 = self.predicter(torch.cat([graph_mt, graph_wt], dim=1))
        return res1, res2

class Model_w_attention(nn.Module):
    def __init__(self,mode):
        super(Model_w_attention, self).__init__()
        # self.gearnet = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512], num_relation=7,
        #                                 batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")
        # caution
        # 对边的种类的独热编码甚至是动态的？7种边需要7个维度 5种边需要5个维度
        # 21+21+7+1=
        
        self.gearnet = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512],
                       num_relation=5, edge_input_dim=57, num_angle_bin=8,
                       batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")
        # 512*3*2=3072
        self.predicter = nn.Sequential(
            nn.Linear(3072, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.mode=mode
        # 512*3=1536
        self.attnModel = ScaledDotAttention(1536, 8,self.mode)

    def forward(self, dual,wtG, mtG, mask, wtBigG, mtBigG):
        d1 = self.gearnet(wtG, wtG.node_feature.float())
        d2 = self.gearnet(mtG, mtG.node_feature.float())
        node_wt = d1["node_feature"]
        node_mut = d2["node_feature"]
        # graph_wt = d1["graph_feature"]
        # graph_mut = d2["graph_feature"]
        if self.mode==2:
            wt_cov_mat = self.calCov(wtBigG, len(wtG))
            mt_cov_mat = self.calCov(mtBigG, len(mtG))
        else:
            wt_cov_mat = None
            mt_cov_mat = None
        wt_embedding = self.attnModel(node_wt, mask, wt_cov_mat)
        mt_embedding = self.attnModel(node_mut, mask,mt_cov_mat)
        if dual is False:
            res1 = self.predicter(torch.cat([wt_embedding, mt_embedding], dim=1))
            return res1
        else:
            res1 = self.predicter(torch.cat([wt_embedding, mt_embedding], dim=1))
            res2 = self.predicter(torch.cat([mt_embedding, wt_embedding], dim=1))
            return res1, res2
    
    def calCov(self,BigGraph,batchsize):
        d1 = self.gearnet(BigGraph, BigGraph.node_feature.float())
        # grapF = d1["graph_feature"]
        nodeF = d1["node_feature"]
        edge_list = BigGraph.edge_list
        num_residues = BigGraph.num_residues
        
        # 由于gearnet的edge_list有边的种类之分 可能重复 因此进行一定的处理
        ndarr = np.array(edge_list[:, :2].cpu())
        unique_array = np.unique(ndarr, axis=0)
        # 只考虑边连接信息 并去重
        sorted_indices = np.lexsort(
        (unique_array[:, 0], unique_array[:, 1]))
        sorted_array = unique_array[sorted_indices]
        # 按一定优先级排序（第二列优先
        edge_index = torch.tensor(sorted_array).t()

        mol_id = -1
        conformer_indicater = []
        for i in range(len(num_residues)):
            mol_id += 1
            mol_batch_index = torch.Tensor(
            [mol_id for _ in range(num_residues[i])])
            conformer_indicater.append(mol_batch_index)
        conformer_indicater = torch.cat(conformer_indicater).long()
        Cov = self._compute_Cov_by_nodemean(
            node_feature=nodeF,
            edge_index=edge_index.to(nodeF.device),
            batch=conformer_indicater.to(nodeF.device),
            batch_size=batchsize,
            num_conformers=5,
        )
        return Cov

    def _compute_Cov_by_nodemean(
            self,
            # Node feature matrix [h, pos] if else outter
            node_feature: torch.Tensor,
            edge_index: torch.Tensor,
            batch: torch.Tensor,
            batch_size: int,
            num_conformers: int,
    ):

        # Myreadout = aggr_resolver('sum')
        # (N1+..NB)*F --> B*N_max*F
        out_ft_dense, mask_ft_dense = to_dense_batch(
            x=node_feature, batch=batch  # [Nc*B*nodes] -> sum(nodes)
        )
        # len(out_total) is a batch size, len(out_total[i]) is the number of conformer graphs of each molecular;
    # out_total[i][j] is the node feature matrices of the j-th conformer.
        out_ft_batch = get_list_node_features_batch(
            batch_size, num_conformers, out_ft_dense, mask_ft_dense
        )
# 返回每个conformer的邻接矩阵 长度为B*num_conformers 矩阵的大小也是N_max*N_max
        adj_dense = to_dense_adj(
            edge_index=edge_index,
            batch=batch,
        )
        # 有填充
        # np.where(np.all(np.array(adj_dense[0].cpu()) == 0, axis=1))[0]
        # array([478,  479,  480, ..., 1697, 1698, 1699])
        # np.where(np.all(np.array(adj_dense[0].cpu()) == 0, axis=0))[0]
        # array([478,  479,  480, ..., 1697, 1698, 1699])
        # 长度为batchsize 每个元素包含num_conformers个邻接矩阵
        adj_dense_batch = get_adj_dense_batch(
            batch_size, num_conformers, adj_dense)
        # F_bary_batch = torch.zeros(
        #     (batch_size * num_conformers, node_feature.shape[1]), device=self.device
        # )
        # my_out = torch.zeros(
        #     (batch_size, node_feature.shape[1]), device=self.device
        # )
        myOut = []

        for index in range(batch_size):

            out_ft_sample = out_ft_batch[index]
            mean_bary = torch.mean(torch.stack(out_ft_sample), dim=0)


            cov = self._comput_cov(out_ft_sample, mean_bary)
            myOut.append(cov)

            # h_out_bary = Myreadout(F_bary, dim=0)
            # my_out[index, :] = h_out_bary
            # F_bary形如（N,512）N为构想的
            # F_bary_batch[index_bary: index_bary + num_conformers, :] = h_out_bary.repeat(
            #     num_conformers, 1
            # )
            # index_bary = index_bary + num_conformers

        # node_feature = Myreadout(node_feature, batch, dim=0)
        # return node_feature, F_bary_batch, my_out
        return torch.stack(myOut, dim=0)
    def _compute_barycenter(
            self,
            # Node feature matrix [h, pos] if else outter
            node_feature: torch.Tensor,
            edge_index: torch.Tensor,
            batch: torch.Tensor,
            batch_size: int,
            num_conformers: int,
    ):

        # Myreadout = aggr_resolver('sum')
        # (N1+..NB)*F --> B*N_max*F
        out_ft_dense, mask_ft_dense = to_dense_batch(
            x=node_feature, batch=batch  # [Nc*B*nodes] -> sum(nodes)
        )
        # len(out_total) is a batch size, len(out_total[i]) is the number of conformer graphs of each molecular;
    # out_total[i][j] is the node feature matrices of the j-th conformer.
        out_ft_batch = get_list_node_features_batch(
            batch_size, num_conformers, out_ft_dense, mask_ft_dense
        )
# 返回每个conformer的邻接矩阵 长度为B*num_conformers 矩阵的大小也是N_max*N_max
        adj_dense = to_dense_adj(
            edge_index=edge_index,
            batch=batch,
        )
        # 有填充
        # np.where(np.all(np.array(adj_dense[0].cpu()) == 0, axis=1))[0]
        # array([478,  479,  480, ..., 1697, 1698, 1699])
        # np.where(np.all(np.array(adj_dense[0].cpu()) == 0, axis=0))[0]
        # array([478,  479,  480, ..., 1697, 1698, 1699])
        #长度为batchsize 每个元素包含num_conformers个邻接矩阵
        adj_dense_batch = get_adj_dense_batch(
            batch_size, num_conformers, adj_dense)
        # F_bary_batch = torch.zeros(
        #     (batch_size * num_conformers, node_feature.shape[1]), device=self.device
        # )
        # my_out = torch.zeros(
        #     (batch_size, node_feature.shape[1]), device=self.device
        # )
        myOut=[]
        

        for index in range(batch_size):

            out_ft_sample = out_ft_batch[index]
            adj_dense_sample = adj_dense_batch[index]
            list_adjs = [item for item in adj_dense_sample]
            if adj_dense_sample[0].get_device() >= 0:
                w_tmp = [
                    torch.ones(t.shape[0], dtype=torch.float32).to(
                        adj_dense_sample[0].get_device()
                    )
                    / t.shape[0]
                    for t in list_adjs
                ]
                lambdas = torch.ones(len(list_adjs), dtype=torch.float32).to(
                    adj_dense_sample[0].get_device()
                ) / len(list_adjs)
            else:
                w_tmp = [
                    torch.ones(t.shape[0], dtype=torch.float32) / t.shape[0] for t in list_adjs
                ]
                lambdas = torch.ones(
                    len(list_adjs), dtype=torch.float32) / len(list_adjs)

            F_bary, C_bary, log = fgw_barycenters(
                N=adj_dense_sample.shape[1],
                Ys=out_ft_sample,
                Cs=list_adjs,
                ps=w_tmp,
                lambdas=lambdas,
                warmstartT=True,
                symmetric=True,
                method="sinkhorn_log",
                alpha=0.1,
                solver="PGD",
                fixed_structure=False,
                fixed_features=False,
                # epsilon=0.1,
                epsilon=0.6,
                p=None,
                loss_fun="square_loss",
                max_iter=5,
                tol=1e-2,
                # numItermax=5,
                numItermax=15,
                stopThr=1e-2,
                verbose=False,
                log=True,
                init_C=list_adjs[0],
                init_X=None,
                random_state=None,
            )
            
            cov= self._comput_cov(out_ft_sample,F_bary)
            myOut.append(cov)
            
            
            # h_out_bary = Myreadout(F_bary, dim=0)
            # my_out[index, :] = h_out_bary
            # F_bary形如（N,512）N为构想的
            # F_bary_batch[index_bary: index_bary + num_conformers, :] = h_out_bary.repeat(
            #     num_conformers, 1
            # )
            # index_bary = index_bary + num_conformers

        # node_feature = Myreadout(node_feature, batch, dim=0)
        # return node_feature, F_bary_batch, my_out
        return torch.stack(myOut,dim=0)
    def _comput_cov(self,X_list,C):
        distances = []
        for X in X_list:
            dist = torch.norm(X - C, dim=1, keepdim=True)  # 计算欧氏距离
            distances.append(dist)
        D = torch.cat(distances, dim=1)
        D_centered = D - D.mean(dim=1, keepdim=True)
        cov_matrix = torch.mm(D_centered, D_centered.t()) / (D.size(1) - 1)
        return cov_matrix

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

        # att_norm=self._norm_by_1dmask(attention_scores,mask)
        # att_norm=attention_scores
        if self.mode==2:
            refCov=self._smooth_abs(refCov)
            # cov_norm = self._norm_by_1dmask(refCov.unsqueeze(1), mask)
            
            # att_scores = att_norm+self.balance_factor*cov_norm
            # att_scores = attention_scores + self.balance_factor * refCov.unsqueeze(1)
            att_scores = (1-self.balance_factor)*attention_scores + \
                self.balance_factor * refCov.unsqueeze(1)
        else:
            att_scores = attention_scores
            # pass
        att_scores_masked=self._mask_by_1dmask(att_scores,mask)
        attention_weights = F.softmax(
            att_scores_masked, dim=-1)  # (B, num_heads, L, L)

        # Step 6: 计算加权和输出
        # (B, num_heads, L, d_k)
        output = torch.matmul(attention_weights, value)
        # output.register_hook(lambda grad: print("Output Gradient:", grad))
        output2 = output.transpose(1, 2).contiguous().view(B, L, h)  # (B, L, h)
        # output2.register_hook(lambda grad: print("Output2 Gradient:", grad))
        # Step 7: 通过输出线性变换
        output2 = self.output_linear(output2)  # (B, L, h)
        
        output3=output2[:,0,:]
        # 这样的话不是只算第一行就行了嘛 后面白算了啊
        # 基于有效token取mean?
        # output3 = (output2*mask.unsqueeze(-1)).mean(dim=1)
        # output3.register_hook(lambda grad: print("Output3 Gradient:", grad))
        return output3
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





