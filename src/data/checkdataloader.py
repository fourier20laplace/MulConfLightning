import pandas as pd
from components.skempi2 import SKEMPIV2Dataset0429
from torch.utils.data import DataLoader
from components.collate_fn import mycollate_fn
from joblib import Parallel, delayed
if __name__ == "__main__":
    data_df = pd.read_csv("/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/skempi_v2_train_filtered.csv")
    data_df = data_df[data_df["PDB_id"] != "3VR6"]
    dataset = SKEMPIV2Dataset0429(data_df, is_train=True, knn_num=6, knn_agents_num=6)

    dataloader=DataLoader(
            dataset=dataset, 
            shuffle=False,
            batch_size=8, 
            pin_memory=True,
            drop_last=False, 
            num_workers=4, 
            collate_fn=mycollate_fn,
        )
    for batch in dataloader:
        print("check batch")
        # print(batch)
    #     break