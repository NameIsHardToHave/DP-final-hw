import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.functional import F

from yacs.config import CfgNode


class MyCrossEntropyLoss(nn.Module):
    r"""加入了一个极小量"""
    def __init__(self, delta: float=1e-10):
        super().__init__()
        self.delta = delta

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        r"""计算交叉熵，需要考虑到批处理维度"""
        # 替换nan值为某个极小值
        nan_mask = torch.isnan(input)
        input[nan_mask] = self.delta
        # 计算交叉熵
        return F.cross_entropy(input, target)
        # n = target.shape[0]
        # normal_k = torch.exp(input).sum(dim=1)      # 这一步有可能造成数值溢出
        # select_k = input[torch.arange(n), target.squeeze()]
        # return -torch.log(torch.exp(select_k) / normal_k).mean()
        
        
class PairwiseRankLoss(nn.Module):
    r"""成对打分损失"""
    def __init__(self, config: CfgNode):
        super().__init__()
        self.device = config.device
        self.neg_num = config.rl.neg_num
        self.input_size = config.model.input_size
        
    def negative_sample(self, target: torch.Tensor) -> torch.Tensor:
        r"""对动作进行负采样"""
        # 这里采用随机采样，因为动作是足够稀疏的 （返回 批处理大小 x 负样本大小）
        return torch.randint(0, self.input_size, (target.shape[0], self.neg_num)).to(self.device) 
        
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        r"""计算成对排名损失, 需要考虑到批处理维度"""
        n, _ = input.shape
        index = torch.arange(n).to(self.device)
        
        # 提取出负样本位置 （批处理大小 x 负样本大小）
        if self.neg_num > 0:
            neg_action = self.negative_sample(target)
            in_neg = input[index.repeat_interleave(self.neg_num).reshape(n, self.neg_num), neg_action]
        else:
            raise RuntimeError(f"neg_num should be set more than 0, but got {self.neg_num}")
        
        # 提取出正样本位置 （批处理大小 x 负样本大小）
        if target.dim() == 1:
            in_pos = input[index, target]
        elif target.dim() == 2:
            target = target.flatten()
            in_pos = input[index, target]
        else:
            raise RuntimeError(f"target should be 1 dimension or 2 dimension, but got {target.dim()}")
        in_pos = in_pos.repeat_interleave(self.neg_num).reshape(n, self.neg_num)
            
        # 计算成对损失
        Loss = - (in_pos - in_neg).sigmoid().log().mean()

        # 计算交叉熵
        return Loss