'''生成模型底座（只输出隐藏表示，不经过最后一层全连接层/内积层）'''

import torch
from torch import nn
from torch.functional import F
import numpy as np

from yacs.config import CfgNode


# 1 基于GRU的深度推荐模型
class GRU4Rec(nn.Module):
    def __init__(self, input_size: int, state_size: int, max_len: int, 
                 embedding_dim: int, num_layers: int, dropout: float):
        super(GRU4Rec, self).__init__()
        self.embedding = nn.Embedding(input_size+1, embedding_dim)
        self.gru = nn.GRU(embedding_dim, state_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x: torch.Tensor):
        embedded = self.embedding(x)
        out, h = self.gru(embedded, None)
        return out[:, -1, :]
    

# 2 基于自注意力机制的深度推荐模型
class SAS4Rec(nn.Module):
    def __init__(self, input_size: int, state_size: int, max_len: int,
                 num_layers: int, dropout: float, num_heads: int, dim_feedforward: int, activation: str,
                 shrink: bool=False):
        super(SAS4Rec, self).__init__()
        self.padding_idx = input_size
        self.max_len = max_len
        self.shrink = shrink
        self.item_emb = nn.Embedding(input_size+1, state_size, padding_idx=self.padding_idx)
        self.pos_emb = nn.Embedding(max_len, state_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=state_size, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=dim_feedforward,
                                                   activation=activation,
                                                   dropout=dropout, 
                                                   batch_first=True)
        norm_layer = nn.LayerNorm(state_size, eps=1e-5)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, 
                                            num_layers=num_layers,
                                            norm=norm_layer)

    def forward(self, x: torch.Tensor, all: bool=False):
        device = x.device
        # 创建项目嵌入并缩放
        seqs = self.item_emb(x)
        if self.shrink == True:
            seqs *= self.item_emb.embedding_dim ** 0.5
        # 生成可学习的位置编码信息并叠加
        positions = torch.tile(torch.arange(x.shape[1]), [x.shape[0], 1]).long().to(device)
        seqs += self.pos_emb(positions)
        # 创建未来信息掩码矩阵和填充掩码矩阵
        mask = ~torch.tril(torch.ones(self.max_len, self.max_len)).bool().to(device)
        # src_key_padding_mask = (x == self.padding_idx)
        # 馈入Transformer编码器层
        out = self.encoder(seqs, mask=mask, src_key_padding_mask=None)
        # print(out[:,-1,:].isnan().sum())
        
        # 是否仅输出最后一个时间步的表示
        if all == False:
            return out[:, -1, :]    # batch_size x state_size
        else:
            return out              # batch_size x max_len x state_size
        

# 2.5 SASRec在github上的原始实现
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, input_size: int, state_size: int, max_len: int,
                 num_layers: int, dropout: float, num_heads: int):
        super(SASRec, self).__init__()
        self.item_num = input_size
        self.padding_idx = input_size   # set padding location

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(input_size+1, state_size, padding_idx=self.padding_idx)
        self.pos_emb = torch.nn.Embedding(max_len, state_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=dropout)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(state_size, eps=1e-8)

        for _ in range(num_layers):
            new_attn_layernorm = torch.nn.LayerNorm(state_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(state_size,
                                                          num_heads,
                                                          dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(state_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(state_size, dropout)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def forward(self, log_seqs: torch.Tensor, all: bool=True) -> torch.Tensor:
        self.dev = log_seqs.device
        
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        # timeline_mask = torch.BoolTensor(log_seqs == self.padding_idx).to(self.dev)
        timeline_mask = torch.tensor(log_seqs == self.padding_idx, dtype=torch.bool, device=self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        if all == True:
            return log_feats
        else:
            return log_feats[:, -1, :]
    
    
# 3 基于CNN的改进的深度推荐模型
class NextItNet(nn.Module):
    def __init__(self, input_size: int, state_size: int, max_len: int,
                 dilations: list, kernel_size: int):
        super(NextItNet, self).__init__()
        self.dilations = dilations
        self.state_size = state_size
        self.kernel_size = kernel_size
        self.embedding = nn.Embedding(input_size+1, state_size)
        residual_block = [nn.ModuleList([nn.Conv2d(state_size, state_size,
                                                  kernel_size=(1, kernel_size), padding=0, dilation=dilation),
                                         nn.LayerNorm(state_size),
                                         nn.Conv2d(state_size, state_size,
                                                   kernel_size=(1, kernel_size), padding=0, dilation=2*dilation),
                                         nn.LayerNorm(state_size),
                                ]) for dilation in self.dilations]
        self.residual_blocks = nn.ModuleList(residual_block)

    def forward(self, x: torch.Tensor): # inputs: [batch_size, seq_len]
        # 项目嵌入
        inputs = self.embedding(x) # [batch_size, seq_len, embed_size]
        # 遍历所有残差块
        for i, block in enumerate(self.residual_blocks):
            ori = inputs
            # 填充 -> 卷积 -> 标准化 -> ReLU -> 填充 -> 卷积（更大核） -> 标准化 -> ReLU
            inputs_pad = self.conv_pad(inputs, self.dilations[i])
            dilated_conv = block[0](inputs_pad).squeeze(2) # [batch_size, embed_size, seq_len]
            dilated_conv = dilated_conv.permute(0, 2, 1)
            relu1 = F.relu(block[1](dilated_conv)) # [batch_size, seq_len, embed_size]
            inputs_pad = self.conv_pad(relu1, self.dilations[i]*2)
            dilated_conv = block[2](inputs_pad).squeeze(2)  # [batch_size, embed_size, seq_len]
            dilated_conv = dilated_conv.permute(0, 2, 1)
            relu1 = F.relu(block[3](dilated_conv))  # [batch_size, seq_len, embed_size]
            # 残差结构，相加
            inputs = ori + relu1
        # 输出最后一个步长的表示
        hidden = inputs[:, -1, :].view(-1, self.state_size) # [batch_size, embed_size]
        return hidden

    def conv_pad(self, inputs: torch.Tensor, dila_: int) -> torch.Tensor:
        r"""在左端填充0, 确保输入输出参数相等"""
        inputs_pad = inputs.permute(0, 2, 1)  # [batch_size, embed_size, seq_len]
        inputs_pad = inputs_pad.unsqueeze(2)  # [batch_size, embed_size, 1, seq_len]
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dila_, 0, 0, 0))
        inputs_pad = pad(inputs_pad)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*self.dilations[i]]
        return inputs_pad
    

def build_base(config: CfgNode) -> nn.Module:
    r"""创建模型基座"""
    # 检查基座名字正误
    base_lists = ['GRU4Rec', 'SAS4Rec', 'SASRec', 'NextItNet']
    assert config.model.base_name in base_lists, \
        f"config.model.base should be in {base_lists}, but got {config.model.base_name}."
    
    # 统一的参数
    fund_info = (config.model.input_size, config.model.state_size, config.model.max_len)
        
    # 创建基座模型
    if config.model.base_name == 'GRU4Rec':
        args = config.model.base.GRU4Rec
        return GRU4Rec(*fund_info, 
                       embedding_dim = args.embedding_dim,
                       num_layers = args.num_layers,
                       dropout = args.dropout)
    elif config.model.base_name == 'SAS4Rec':
        args = config.model.base.SAS4Rec
        return SAS4Rec(*fund_info, 
                       num_layers = args.num_layers,
                       dropout = args.dropout,
                       num_heads = args.num_heads,
                       dim_feedforward = args.dim_feedforward,
                       activation = args.activation,
                       shrink = args.shrink)
    elif config.model.base_name == 'SASRec':
        args = config.model.base.SAS4Rec
        return SASRec(*fund_info,
                       num_layers = args.num_layers,
                       dropout = args.dropout,
                       num_heads = args.num_heads)
    elif config.model.base_name == 'NextItNet':
        args = config.model.base.NextItNet
        return NextItNet(*fund_info, 
                       dilations = args.dilations, 
                       kernel_size = args.kernel_size)