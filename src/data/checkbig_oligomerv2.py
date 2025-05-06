from tqdm import tqdm
import pandas as pd
from components.skempi2 import SKEMPIV2Dataset0429
from components.skempi2_v3 import SKEMPIV2Dataset0504
from torch.utils.data import DataLoader
from components.collate_fn import mycollate_fn
from joblib import Parallel, delayed

import concurrent.futures
data_df = pd.read_csv(
    "/home/lmh/projects_dir/Antibody_Mutation/data/SKEMPIv2/skempi_v2_train_filtered.csv")
data_df = data_df[~data_df["PDB_id"].isin(["3VR6", "1KBH"])]
dataset = SKEMPIV2Dataset0504(
    data_df, is_train=True, knn_num=6, knn_agents_num=6, rand=False)


def is_problematic(sample):
    """判断样本是否超过阈值"""
    return (sample["roi_repr_wt_tensor"].shape[0] >= 500 or
            sample["roi_repr_mut_tensor"].shape[0] >= 500)


def process_single(i):
    try:
        sample = dataset[i]
        if is_problematic(sample):
            return sample["wt"]["PDB_id"]
        else:
            return None
    except Exception as e:
        return f"Error at index {i}: {str(e)}"


def process_all():
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_single, i)                   : i for i in range(len(dataset))}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result is not None:
                results.append(result)

    print("check dataset done!")
    if results:
        print("Problematic samples:")
        for pdb_id in results:
            print(pdb_id)
    else:
        print("All samples are OK.")


process_all()



