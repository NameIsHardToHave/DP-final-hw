import torch
from torch import nn
from torch.backends import cudnn

from yacs.config import CfgNode
from copy import deepcopy
from typing import Tuple

from .base import build_base
from .head import SupervisedHead


# 1 自监督算法模型
class SN(nn.Module):
    def __init__(self, config: CfgNode):
        super(SN, self).__init__()
        self.base = build_base(config)
        self.supervisedhead = SupervisedHead(config.model.state_size, config.model.input_size)
        
    def forward(self, x: torch.Tensor):
        x = self.base(x)
        x = self.supervisedhead(x)
        return x
    

# 2 采用原论文的训练方式
class SNPair(nn.Module):
    def __init__(self, config: CfgNode):
        super(SNPair, self).__init__()
        # 设置模型基本框架
        self.base = build_base(config)
        # 设置模型损失函数
        self.pos_weight = torch.tensor(config.model.pair.pos_weight)
        self.input_size = config.model.input_size
        self.bce_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.cro_criterion = nn.CrossEntropyLoss(ignore_index=self.input_size)
        # 模型的项目嵌入层
        self.l2_emb = config.model.pair.l2_emb
        self.item_emb = self.base.item_emb
        # 其它可能用到的参数
        self.loss = config.model.pair.loss
        self.device = config.device
        self.item_all = torch.arange(self.input_size, requires_grad=False).to(self.device)
        self.padding_idx = self.input_size      # 设置填充位置
    
    def negative_sample(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""对动作进行负采样"""
        # 这里采用随机采样，因为动作是足够稀疏的 （batch_size x max_len）
        return torch.randint(0, self.input_size, x.shape).to(self.device)
    
    def positive_sample(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""对动作进行正采样（即下一个序列）"""
        return torch.cat([x[:, 1:], target.unsqueeze(-1)], dim=1).to(self.device)     # batch_size x max_len
    
    def forward_and_loss(self, x: torch.Tensor, target: torch.Tensor):
        # 得到全部时间步下的隐藏表示
        res = self.base(x, all=True)        # batch_size x max_len x state_size
        # 采集正负样本
        neg_seq = self.negative_sample(x, target)   # batch_size x max_len
        pos_seq = self.positive_sample(x, target)   # batch_size x max_len
        # 不同的损失计算方式
        if self.loss == 'bce':
            # 这里的项目嵌入需要流经梯度吗???
            pos_embs = self.item_emb(pos_seq)   # batch_size x max_len x state_size
            neg_embs = self.item_emb(neg_seq)   # batch_size x max_len x state_size
            # 得到预测输出
            pos_logits = (res * pos_embs).sum(dim=-1)   # batch_size x max_len
            neg_logits = (res * neg_embs).sum(dim=-1)   # batch_size x max_len
            # 生成预测标签
            pos_labels = torch.ones(pos_logits.shape, device=self.device)   # batch_size x max_len
            neg_labels = torch.zeros(neg_logits.shape, device=self.device)  # batch_size x max_len
            # 计算损失（和某个项目嵌入的内积越大，表示该项目嵌入越有可能是最终结果）
            indices = torch.where(pos_seq != self.padding_idx)
            loss = self.bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += self.bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in self.item_emb.parameters():
                loss += self.l2_emb * torch.norm(param)
        elif self.loss == 'cro':
            # 计算和所有项目嵌入的内积(计算量较大), 同样需要流经梯度吗???
            all_seq = self.item_emb(self.item_all)      # item_num x state_size
            all_logits = torch.matmul(res, all_seq.T)   # batch_size x max_len x item_num
            preds = all_logits.reshape(-1, self.input_size)
            loss = self.cro_criterion(preds, pos_seq.flatten())
            for param in self.item_emb.parameters():
                loss += self.l2_emb * torch.norm(param)
        else:
            raise NameError(f"loss must be in ['bce', 'cro'], but got {self.loss}")
            
        return loss
    
    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        r"""部署时的推理"""
        res = self.base(x, all=False)               # batch_size x state_size
        item_embs = self.item_emb(self.item_all)    # item_num x state_size
        batch_size, state_size = res.shape
        
        # 转化为 batch_size x item_num x state_size
        res = res.repeat_interleave(self.input_size, dim=0).reshape(batch_size, self.input_size, state_size)
        item_embs = item_embs.flatten().repeat(batch_size).reshape(batch_size, self.input_size, state_size)

        logits = (res * item_embs).sum(dim=-1)  # batch_size x item_num
        # logits = logits.sigmoid()             # 担心sigmoid之后变成相似的值
        return logits
    

def build_model(config: CfgNode) -> nn.Module:
    r"""构建完整的模型"""
    # 检查正误
    model_dict = {
        'SN': SN,
        'SNPair': SNPair,
    }
    assert config.model.merge_name in list(model_dict.keys()), \
        f"config.model.merge_name should be in {list(model_dict.keys())}, but got {config.model.merge_name}."

    # 固定随机数种子
    torch.manual_seed(config.model.seed)
    cudnn.deterministic = config.cudnn.deterministic
    cudnn.benchmark = config.cudnn.benchmark

    # 返回构建的模型
    return model_dict[config.model.merge_name](config).to(config.device)