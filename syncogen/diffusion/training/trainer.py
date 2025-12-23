from typing import List

import gin
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger


@gin.configurable
class Trainer:
    """Wrapper around lightning.Trainer."""

    def __init__(
        self,
        accelerator: str = "auto",
        devices: str = "auto",
        num_nodes: int = 1,
        precision: str = "bf16-mixed",
        max_epochs: int = None,
        max_steps: int = 1_000_000,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float = 1.0,
        val_check_interval: float = None,
        check_val_every_n_epoch: int = 1,
        num_sanity_val_steps: int = 2,
        log_every_n_steps: int = 10,
        limit_train_batches: float = 1.0,
        limit_val_batches: float = 1.0,
        fast_dev_run: bool = False,
        strategy: str = "auto",
        enable_checkpointing: bool = True,
        default_root_dir: str = None,
        callbacks: List[Callback] = None,
        logger: Logger = None,
    ):
        self.trainer = L.Trainer(
            accelerator=accelerator,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            max_epochs=max_epochs,
            max_steps=max_steps,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            num_sanity_val_steps=num_sanity_val_steps,
            log_every_n_steps=log_every_n_steps,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            fast_dev_run=fast_dev_run,
            strategy=strategy,
            enable_checkpointing=enable_checkpointing,
            default_root_dir=default_root_dir,
            callbacks=callbacks or [],
            logger=logger,
        )

    def fit(self, model, train_dataloaders=None, val_dataloaders=None, **kwargs):
        return self.trainer.fit(model, train_dataloaders, val_dataloaders, **kwargs)

    def validate(self, model=None, dataloaders=None, **kwargs):
        return self.trainer.validate(model, dataloaders, **kwargs)

    def test(self, model=None, dataloaders=None, **kwargs):
        return self.trainer.test(model, dataloaders, **kwargs)

    def predict(self, model=None, dataloaders=None, **kwargs):
        return self.trainer.predict(model, dataloaders, **kwargs)
