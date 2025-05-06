from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
# from torchvision.datasets import MNIST
# from torchvision.transforms import transforms

from .components.skempi2 import SKEMPIV2Dataset,SKEMPIV2Dataset0429,SKEMPIV2Dataset0429_mode1
from .components.skempi2_v2 import SKEMPIV2Dataset0503
from .components.skempi2_v3 import SKEMPIV2Dataset0504
from .components.skempi2lazy import SKEMPIV2DatasetMulConfLazy
from .components.collate_fn import mycollate_fn
import pandas as pd
import os

class SKEMPI2DataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_root_path:str,
        knn_neighbors_num: int ,
        knn_agents_num: int,
        batch_size: int ,
        num_workers: int ,
        pin_memory: bool ,
        debug: bool ,
        model_mode: int ,
        rand: bool,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # self.save_hyperparameters(logger=False)
        self.save_hyperparameters(logger=True)
        self.batch_size_per_device = batch_size



    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # 似乎这个位置就是用来下载数据的？
        # MNIST(self.hparams.data_dir, train=True, download=True)
        # MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        # if not self.data_train and not self.data_val and not self.data_test:
        #     trainset = MNIST(self.hparams.data_dir, train=True, transform=self.transforms)
        #     testset = MNIST(self.hparams.data_dir, train=False, transform=self.transforms)
        #     dataset = ConcatDataset(datasets=[trainset, testset])
        #     self.data_train, self.data_val, self.data_test = random_split(
        #         dataset=dataset,
        #         lengths=self.hparams.train_val_test_split,
        #         generator=torch.Generator().manual_seed(42),
        #     )
        train_path = os.path.join(self.hparams.data_root_path,"Antibody_Mutation/data/SKEMPIv2/skempi_v2_train_filtered.csv")
        val_path = os.path.join(self.hparams.data_root_path,'Antibody_Mutation/data/SKEMPIv2/skempi_v2_val_filtered.csv')
        train_df = pd.read_csv(train_path, dtype={"PDB_id": "string"})
        val_df = pd.read_csv(val_path, dtype={"PDB_id": "string"})
        # 这个蛋白太大了 去掉之
        # 4NM8 太大了 去掉 罪过罪过
        myset={"3VR6", "1KBH",'4NM8', '4PWX', '4GXU', '4LRX'}
        train_df = train_df[~train_df["PDB_id"].isin(myset)]
        val_df = val_df[~val_df["PDB_id"].isin(myset)]
        # 测试的时候取一个小的数据集
        if self.hparams.debug:
            train_df = train_df.iloc[0:16]
            val_df = val_df.iloc[0:16]
        if self.hparams.model_mode == 2:
            self.data_train = SKEMPIV2DatasetMulConfLazy(train_df, is_train=True, knn_num=self.hparams.knn_neighbors_num,
                                                knn_agents_num=self.hparams.knn_agents_num)
            self.data_val = SKEMPIV2DatasetMulConfLazy(val_df, is_train=False, knn_num=self.hparams.knn_neighbors_num,
                                                knn_agents_num=self.hparams.knn_agents_num)
        elif self.hparams.model_mode == 1:
            self.data_train = SKEMPIV2Dataset0429_mode1(train_df, is_train=True, knn_num=self.hparams.knn_neighbors_num,
                                            knn_agents_num=self.hparams.knn_agents_num)
            self.data_val = SKEMPIV2Dataset0429_mode1(val_df, is_train=False, knn_num=self.hparams.knn_neighbors_num,
                                        knn_agents_num=self.hparams.knn_agents_num)
        elif self.hparams.model_mode in [6]:
            self.data_train = SKEMPIV2Dataset0503(train_df, is_train=True, knn_num=self.hparams.knn_neighbors_num,
                                            knn_agents_num=self.hparams.knn_agents_num,rand=self.hparams.rand)
            self.data_val = SKEMPIV2Dataset0503(val_df, is_train=False, knn_num=self.hparams.knn_neighbors_num,
                                        knn_agents_num=self.hparams.knn_agents_num,rand=self.hparams.rand)
        elif self.hparams.model_mode in [7]:
            self.data_train = SKEMPIV2Dataset0504(self.hparams.data_root_path,train_df, is_train=True, knn_num=self.hparams.knn_neighbors_num,
                                            knn_agents_num=self.hparams.knn_agents_num,rand=self.hparams.rand)
            self.data_val = SKEMPIV2Dataset0504(self.hparams.data_root_path,val_df, is_train=False, knn_num=self.hparams.knn_neighbors_num,
                                        knn_agents_num=self.hparams.knn_agents_num,rand=self.hparams.rand)
        
        elif self.hparams.model_mode in [3,4,5]:
            self.data_train = SKEMPIV2Dataset0429(train_df, is_train=True, knn_num=self.hparams.knn_neighbors_num,
                                            knn_agents_num=self.hparams.knn_agents_num,rand=self.hparams.rand)
            self.data_val = SKEMPIV2Dataset0429(val_df, is_train=False, knn_num=self.hparams.knn_neighbors_num,
                                        knn_agents_num=self.hparams.knn_agents_num,rand=self.hparams.rand)
        else:
            raise ValueError(f"model_mode {self.hparams.model_mode} is not supported")

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        # return DataLoader(
        #     dataset=self.data_train,
        #     batch_size=self.batch_size_per_device,
        #     num_workers=self.hparams.num_workers,
        #     pin_memory=self.hparams.pin_memory,
        #     shuffle=True,
        # )
        return DataLoader(
            dataset=self.data_train, 
            shuffle=True,
            batch_size=self.batch_size_per_device, 
            pin_memory=self.hparams.pin_memory,
            drop_last=True, 
            num_workers=self.hparams.num_workers, 
            collate_fn=mycollate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        # return DataLoader(
        #     dataset=self.data_val,
        #     batch_size=self.batch_size_per_device,
        #     num_workers=self.hparams.num_workers,
        #     pin_memory=self.hparams.pin_memory,
        #     shuffle=False,
        # )
        return DataLoader(
            dataset=self.data_val, 
            shuffle=False,
            batch_size=self.batch_size_per_device, 
            pin_memory=self.hparams.pin_memory,
            drop_last=False, 
            num_workers=self.hparams.num_workers, 
            collate_fn=mycollate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        # return DataLoader(
        #     dataset=self.data_test,
        #     batch_size=self.batch_size_per_device,
        #     num_workers=self.hparams.num_workers,
        #     pin_memory=self.hparams.pin_memory,
        #     shuffle=False,
        # )
        pass

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = SKEMPI2DataModule()
