from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import SpearmanCorrCoef,PearsonCorrCoef

class SKEMPI2LitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        Task: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `SKEMPI2LitModule`.

        :param Task: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True,ignore=['optimizer','Task'])
        #*版本更新 Attribute 'optimizer' removed from hparams because it cannot be pickled. 
        # *You can suppress this warning by setting `self.save_hyperparameters(ignore=['optimizer'])`.
        self.optimizer = optimizer

        # self.net = net
        self.Task = Task
        # loss function
        # self.criterion = torch.nn.CrossEntropyLoss() #?内嵌在Task中

        # # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="multiclass", num_classes=10)
        # self.val_acc = Accuracy(task="multiclass", num_classes=10)
        # self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # # for averaging loss across batches
        # self.train_loss = MeanMetric()
        # self.val_loss = MeanMetric()
        # self.test_loss = MeanMetric()

        # # for tracking best so far validation accuracy
        # self.val_acc_best = MaxMetric()
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        
        self.train_spearman = SpearmanCorrCoef()
        self.val_spearman = SpearmanCorrCoef()
        self.train_pearson = PearsonCorrCoef()
        self.val_pearson = PearsonCorrCoef()
        
        # 用于保存最好的spearman和pearson值
        self.val_spearman_best = MaxMetric()
        self.val_pearson_best = MaxMetric()

    def forward(self, x,dual:bool):
        """Perform a forward pass through the model `self.Task`.
        """
        return self.Task(x,dual)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        # self.val_acc.reset()
        # self.val_acc_best.reset()
        self.val_spearman_best.reset()
        self.val_pearson_best.reset()

    def model_step(
        self, batch,dual:bool
    ):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        
        # x, y = batch
        # logits = self.forward(x)
        # loss = self.criterion(logits, y)
        # preds = torch.argmax(logits, dim=1)
        # return loss, preds, y
        return self.forward(batch,dual)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch,dual=True)

        # update and log metrics
        self.train_loss.update(loss)
        # self.train_acc(preds, targets)
        self.train_spearman.update(preds, targets)
        self.train_pearson.update(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/spearman", self.train_spearman, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/pearson", self.train_pearson, on_step=False, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        return loss
    
    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch,dual=False)

        # update and log metrics
        # self.val_loss(loss) 
        # # self.val_acc(preds, targets)
        # self.val_spearman(preds, targets)
        # self.val_pearson(preds, targets)
        self.val_loss.update(loss)
        self.val_spearman.update(preds, targets)
        self.val_pearson.update(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/spearman", self.val_spearman, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/pearson", self.val_pearson, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        
        spearman = self.val_spearman.compute()
        pearson = self.val_pearson.compute()

        # 更新最佳指标
        self.val_spearman_best(spearman)
        self.val_pearson_best(pearson)

        # log最佳指标
        self.log("val/spearman_best", self.val_spearman_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/pearson_best", self.val_pearson_best.compute(), sync_dist=True, prog_bar=True)
        

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass
        # loss, preds, targets = self.model_step(batch)

        # # update and log metrics
        # self.test_loss(loss)
        # self.test_acc(preds, targets)
        # self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.Task = torch.compile(self.Task)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        # optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        optimizer = self.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = SKEMPI2LitModule(None, None, None, None)
