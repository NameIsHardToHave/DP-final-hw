import torch
from torch import nn

# 1.自监督学习头部
class SupervisedHead(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(SupervisedHead, self).__init__()
        # 输入特征表示，输出每个项目的得分
        self.fc = nn.Linear(input_size, output_size)
        
    def forward(self, x: torch.Tensor):
        return self.fc(x)