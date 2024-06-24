import torch
from torch.utils.data import Dataset

from yacs.config import CfgNode
from typing import Tuple, Optional


class DataBook(Dataset):
    r"""书籍交互记录数据集, 加载预处理划分好的数据张量"""
    def __init__(self, X: torch.LongTensor, y: Optional[torch.LongTensor]=None):
        self.X = X
        self.y = y
    
    def __len__(self):
        r"""返回数据集长度"""
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        r"""返回切分后对应索引的X和y"""
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


def create_dataset(config: CfgNode) -> Tuple[Dataset, Dataset, Dataset]:
    r"""加载数据集(训练集/验证集/测试集)""" 
    # 从文件中加载数据
    X_train = torch.load(config.data.path + "X_train.pt")
    y_train = torch.load(config.data.path + "y_train.pt")
    X_valid = torch.load(config.data.path + "X_valid.pt")
    y_valid = torch.load(config.data.path + "y_valid.pt")
    X_test = torch.load(config.data.path + "X_test.pt")
    
    # 是否转置
    if config.data.T == True:
        y_train = y_train.unsqueeze(-1)
        y_valid = y_valid.unsqueeze(-1)
    
    # 构建数据集
    train_dataset = DataBook(X_train, y_train)
    val_dataset = DataBook(X_valid, y_valid)
    test_dataset = DataBook(X_test)
    
    return train_dataset, val_dataset, test_dataset
