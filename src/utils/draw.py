import numpy as np
import matplotlib.pyplot as plt

from yacs.config import CfgNode

def draw_loss_and_metrics(train_losses: list, val_losses: list, train_metrses: list, val_metrses: list,
                          metrics_name: list, config: CfgNode):
    r"""绘图单次训练过程的函数"""
    # 统计迭代次数
    epochs = range(1, len(train_losses) + 1)
    # 创建图形和坐标轴
    fig, ax1 = plt.subplots(figsize=(10, 8))
    # 绘制训练集损失和验证集损失
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'b--', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    # 创建右侧 y 轴
    ax2 = ax1.twinx()
    # 绘制训练集准确率和验证集准确率
    color_sets = ['g', 'y', 'r', 'c', 'm', 'k']
    for i, train_metrs in enumerate(np.array(train_metrses).T):
        ax2.plot(epochs, train_metrs, color_sets[i]+'-', label=f'Train {metrics_name[i]}')
    for i, val_metrs in enumerate(np.array(val_metrses).T):
        ax2.plot(epochs, val_metrs, color_sets[i]+'--', label=f'Validation {metrics_name[i]}')
    ax2.set_ylabel('Metrics', color='g')
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    # 添加标题
    plt.title('Loss and Metrics Result')
    # 保存图片
    plt.savefig(config.draw.path+config.draw.name)
    # 显示图形
    plt.show()