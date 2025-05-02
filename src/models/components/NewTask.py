from .Model import myModel
from .Model import Model_w_attention
from .Model0430 import Model_w_attention_v2
from .Model0502 import Model_w_attention_v3
import torch.nn as nn
from torchdrug import layers
from torchdrug.layers import geometry
from scipy.stats import pearsonr
from collections import defaultdict
from torchdrug import data


class NewTask(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode == 0:
            self.model = myModel()
        elif mode == 1:
            self.model = Model_w_attention(mode)
            self.use_cov = False
        elif mode == 2:
            self.model = Model_w_attention(mode)
            self.use_cov = True
        elif mode == 3:
            self.model = Model_w_attention_v2(mode)
        elif mode == 4:
            self.model = Model_w_attention_v3(mode)
        else:
            raise ValueError("mode must be legal")
        # self.graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
        #                                                          edge_layers=[geometry.SequentialEdge(max_distance=2),
        #                                                                       geometry.SpatialEdge(
        #                                                              radius=10.0, min_distance=5),
        #     geometry.KNNEdge(k=10, min_distance=5)])
        
        # self.graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
        #                                                          edge_layers=[geometry.SequentialEdge(max_distance=2),
        #                                                                       geometry.SpatialEdge(
        #                                                              radius=10.0, min_distance=5),
        #     geometry.KNNEdge(k=10, min_distance=5)],edge_feature="gearnet")
        self.graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                                 edge_layers=[geometry.SequentialEdge(max_distance=1),
                                                                              geometry.SpatialEdge(
                                                                     radius=7.0, min_distance=6),
            geometry.KNNEdge(k=6, min_distance=6)], edge_feature="gearnet")
        self.loss_fn = nn.MSELoss()
        self.mode=mode

    def organizeBigG(self, batchsize, ConfDict):
        tmp_dict = defaultdict(list)
        for key in ConfDict:
            for idx in range(batchsize):
                # 根据idx填充对应列表
                tmp_dict[idx].append(ConfDict[key][idx])
        conformers_list = []
        for idx in sorted(tmp_dict.keys()):  # 按键的顺序拼接
            conformers_list.extend(tmp_dict[idx])
        PackConformers = data.Protein.pack(conformers_list)
        PackConformers.view = "residue"
        BigGraph = self.graph_construction_model(PackConformers)
        return BigGraph

    # def forward(self, batch):
    #     pre1, pre2 = self.predict(batch, dual=True)
    #     # 当bs=1时，pre1和pre2的维度为[1,1]，只需要去掉最后一个维度
    #     pre1 = pre1.squeeze(-1)
    #     pre2 = pre2.squeeze(-1)
    #     lab1, lab2 = self.target(batch)
    #     loss = self.loss_fn(pre1, lab1) + self.loss_fn(pre2, lab2)
    #     return loss,pre1,lab1
    def forward(self, batch,dual:bool):
        if dual:
            pre1, pre2 = self.predict(batch, dual=dual)
            pre1 = pre1.squeeze(-1)
            pre2 = pre2.squeeze(-1)
            lab1, lab2 = self.target(batch)
            loss = self.loss_fn(pre1, lab1) + self.loss_fn(pre2, lab2)
        else:
            pre1 = self.predict(batch, dual=dual)
            pre1 = pre1.squeeze(-1)
            lab1 = batch['ddG']
            loss = self.loss_fn(pre1, lab1)
        return loss,pre1,lab1

    def predict(self, batch, dual):
        #* 似乎graph_construction只能batch之后再做
        wtG = self.graph_construction_model(batch["wtG"])
        mtG = self.graph_construction_model(batch["mtG"])
        if self.mode == 3 or self.mode == 4:
            return self.model(dual, wtG, mtG, batch['mask'], batch['cov_wt_tensor'], batch['cov_mut_tensor'])
        else:
            if self.use_cov:
                bs = len(wtG)
                wtBigG = self.organizeBigG(bs, batch["wtDict"])
                mtBigG = self.organizeBigG(bs, batch["mtDict"])
            else:
                wtBigG = None
                mtBigG = None
            return self.model(dual, wtG, mtG, batch['mask'], wtBigG, mtBigG)

    def target(self, batch):
        return batch['ddG'], -batch['ddG']

    # def _eval(self, batch):
    #     # 在eval阶段不再使用对偶损失 仅仅观察原始样本对
    #     pre1 = self.predict(batch, dual=False)
    #     pre1 = pre1.squeeze(-1)
    #     lab1 = batch['ddG']
    #     loss = self.loss_fn(pre1, lab1)
    #     # 计算相关系数
    #     # pre1 = pre1.detach().cpu().numpy()
    #     # lab1 = lab1.detach().cpu().numpy()
    #     # correlation, _ = pearsonr(pre1, lab1)
    #     # 似乎应该把所有的样本放在一起去求相关系数？
    #     return loss, pre1, lab1
