import pandas as pd
from components.skempi2 import SKEMPIV2Dataset0429
from components.skempi2_v2 import SKEMPIV2Dataset0503TST
from torch.utils.data import DataLoader
from components.collate_fn import mycollate_fn
from joblib import Parallel, delayed
if __name__ == "__main__":
    data_df = pd.read_csv("/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/skempi_v2_val_filtered.csv")
    # data_df = data_df[data_df["PDB_id"] != "3VR6"]
    data_df = data_df[~data_df["PDB_id"].isin(["3VR6", "1KBH"])]
    dataset = SKEMPIV2Dataset0503TST(data_df, is_train=True, knn_num=6, knn_agents_num=6,rand=False)
    def process_single(i):
        # print(f"check {i}")
        return dataset[i]
    def process_all():
        """使用 joblib 并行处理"""
        results = Parallel(n_jobs=32)(delayed(process_single)(i) for i in range(len(dataset)))
    process_all()
    print("check dataset done!")
    # print(len(dataset))
    
    # print(dataset[0])
    # dataloader=DataLoader(
    #         dataset=dataset, 
    #         shuffle=False,
    #         batch_size=2, 
    #         pin_memory=False,
    #         drop_last=False, 
    #         num_workers=0, 
    #         collate_fn=mycollate_fn,
    #     )
    # for batch in dataloader:
    #     print(batch)
    #     break