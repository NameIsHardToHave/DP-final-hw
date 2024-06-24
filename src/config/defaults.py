from .config_node import ConfigNode

# 全局配置（设备）
config = ConfigNode()
config.device = 'cuda'

# 日志记录器
config.logger = ConfigNode()
config.logger.path = 'logging/'
config.logger.name = 'default.log'
config.logger.out_file = True
config.logger.out_console = True

# CUDNN（控制计算稳定性）
config.cudnn = ConfigNode()
config.cudnn.benchmark = True
config.cudnn.deterministic = False

# 数据集配置
config.data = ConfigNode()
config.data.path = 'data/'
config.data.seed = 41
config.data.batch_size = 128
config.data.train_shuffle = True
config.data.T = False

# 模型基础配置
config.model = ConfigNode()
config.model.seed = 41
config.model.path = 'model/'
config.model.name = 'default.pth'
config.model.input_size = 10000
config.model.max_len = 50
config.model.state_size = 128
config.model.base_name = 'GRU4Rec'
config.model.merge_name = 'SN'

# 部分模型配置
config.model.pair = ConfigNode()
config.model.pair.pos_weight = 1.0
config.model.pair.l2_emb = 0.0
config.model.pair.loss = 'bce'

# 模型底座配置
config.model.base = ConfigNode()
config.model.base.GRU4Rec = ConfigNode()
config.model.base.GRU4Rec.embedding_dim = 128
config.model.base.GRU4Rec.num_layers = 2
config.model.base.GRU4Rec.dropout = 0.0

config.model.base.SAS4Rec = ConfigNode()
config.model.base.SAS4Rec.num_layers = 2
config.model.base.SAS4Rec.dropout = 0.0
config.model.base.SAS4Rec.num_heads = 8
config.model.base.SAS4Rec.dim_feedforward = 256
config.model.base.SAS4Rec.activation = 'relu'
config.model.base.SAS4Rec.shrink = False

config.model.base.NextItNet = ConfigNode()
config.model.base.NextItNet.dilations = [1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4]
config.model.base.NextItNet.kernel_size = 3

# 训练配置
config.train = ConfigNode()
config.train.seed = 41
config.train.epochs = 500
config.train.patience = 10

# 训练组件配置
config.opt = ConfigNode()

# 绘图配置
config.draw = ConfigNode()
config.draw.path = 'imgs/'
config.draw.name = 'default.png'


def get_default_config():
    r"""加载默认配置"""
    return config.clone()