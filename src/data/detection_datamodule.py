from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.yolo_detection_dataset import yolo_collate_fn


class DetectionDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        attributes: Optional[object] = None,
        data_name: Optional[str] = None,
        teacher_image_size: Optional[int] = None,
        label_offset: Optional[int] = None,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train = train_dataset
        self.data_val = val_dataset
        self.data_test = test_dataset if test_dataset is not None else val_dataset
        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        if hasattr(self.data_train, "class_num"):
            return self.data_train.class_num
        return self.data_train.attributes.class_num

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by "
                    f"the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=yolo_collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=yolo_collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=yolo_collate_fn,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        return self.test_dataloader()

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass
