import torch
from torch.utils.data import DataLoader, Dataset

from yacs.config import CfgNode
from typing import Tuple


def create_dataloader(train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, 
                      config: CfgNode) -> Tuple[DataLoader, DataLoader, DataLoader]:
    r"""创建数据集加载器"""
    torch.manual_seed(config.data.seed)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.data.batch_size, 
                                  shuffle=config.data.train_shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=config.data.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=config.data.batch_size)
    
    return train_dataloader, val_dataloader, test_dataloader