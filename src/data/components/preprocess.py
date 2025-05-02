from torch.utils.data import DataLoader
import sys
sys.path.append('/home/lmh/projects_dir/MulConf')
from dataset.skempi2lazy import SKEMPIV2DatasetMulConf, SKEMPIV2DatasetMulConfLazy
from dataset.collate_fn import mycollate_fn
import pandas as pd

from joblib import Parallel, delayed
# 添加当前文件目录的父目录


train_path = "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/skempi_v2_train_filtered.csv"
val_path = '/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/skempi_v2_val_filtered.csv'

train_df = pd.read_csv(train_path, dtype={"PDB_id": "string"})
# train_df = train_df.iloc[0:4]
val_df = pd.read_csv(val_path, dtype={"PDB_id": "string"})
# 筛选df 如果PDB_id是3VR6就删除该行
train_df = train_df[train_df["PDB_id"] != "3VR6"]
val_df = val_df[val_df["PDB_id"] != "3VR6"]
# train_dataset = SKEMPIV2DatasetMulConf(train_df, is_train=True, knn_num=20,
#                                    knn_agents_num=20)
train_dataset = SKEMPIV2DatasetMulConfLazy(train_df, is_train=True, knn_num=20,
                                           knn_agents_num=20, name="train", path=train_path)
val_dataset = SKEMPIV2DatasetMulConfLazy(val_df, is_train=True, knn_num=20,
                                         knn_agents_num=20, name="val", path=val_path)


def process_single(i):
    return val_dataset[i]
def process_all():
    """使用 joblib 并行处理"""
    results = Parallel(n_jobs=-1)(delayed(process_single)(i) for i in range(len(val_dataset)))


process_all()
