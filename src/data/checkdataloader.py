import pandas as pd
from components.skempi2 import SKEMPIV2Dataset0429
from components.skempi2_v3 import SKEMPIV2Dataset0504
from torch.utils.data import DataLoader
from components.collate_fn import mycollate_fn
from joblib import Parallel, delayed
if __name__ == "__main__":
    data_df = pd.read_csv("/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/skempi_v2_train_filtered.csv")
    data_df = data_df[~data_df["PDB_id"].isin(["3VR6", "1KBH"])][:4]
    dataset = SKEMPIV2Dataset0504(data_df, is_train=True, knn_num=6, knn_agents_num=6,rand=False)

    for i in range(len(dataset)):
        sample = dataset[i]
        if sample["roi_repr_wt_tensor"].shape[0] >=500 or sample["roi_repr_mut_tensor"].shape[0] >=500:
            print(sample["wt"]["PDB_id"])

    # dataloader=DataLoader(
    #         dataset=dataset, 
    #         shuffle=False,
    #         batch_size=2, 
    #         pin_memory=True,
    #         drop_last=False, 
    #         num_workers=0, 
    #         collate_fn=mycollate_fn,
    #     )
    # for batch in dataloader:
    #     print("check batch")
    #     break
        # print(batch)
    #     break