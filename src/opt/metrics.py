import torch
from torchmetrics import Metric
from torchmetrics import F1Score, Accuracy

class F1Score(F1Score):
    def __str__(self):
        return "F1Score"
    
class Accuracy(Accuracy):
    def __str__(self):
        return "Accuracy"


class Hit(Metric):
    def __init__(self, k: int, label: str=''):
        super().__init__()
        self.k = k
        self.label = label
        self.add_state("hits", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_num", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        n = target.shape[0]
        sorted_preds, sorted_indices = torch.sort(preds, descending=True, dim=1)
        temp = sorted_indices[:, :self.k]
        self.hits += (temp.T == target.squeeze()).sum()
        self.total_num += n

    def compute(self) -> torch.Tensor:
        return self.hits.float() / self.total_num
    
    def __str__(self):
        return self.label + "Hit@" + str(self.k)


class NDCG(Metric):
    def __init__(self, k: int, label: str=''):
        super().__init__()
        self.k = k
        self.label = label
        self.add_state("ndcg", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_num", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        n = target.shape[0]
        # 排序
        sorted_preds, sorted_indices = torch.sort(preds, descending=True, dim=1)
        # 计算每个批量的rank
        rank = (sorted_indices.T == target.squeeze()).T.nonzero()[:, 1] + 1
        # 计算 DCG
        dcg = 1 / (torch.log(rank+1) / torch.log(torch.tensor(2.0)))
        # 去掉排名大于 k 的
        dcg[dcg < 1 / (torch.log(torch.tensor(self.k+1)) / torch.log(torch.tensor(2.0)))] = 0
        # 计算 Ideal DCG
        ideal_dcg = 1
        # 计算 NDCG
        ndcg = (dcg / ideal_dcg).sum()
        self.ndcg += ndcg
        self.total_num += n

    def compute(self) -> torch.Tensor:
        return self.ndcg.float() / self.total_num
    
    def __str__(self):
        return self.label + "NDCG@" + str(self.k)