## 目录介绍
- configs: 配置文件
- data: 预处理后的数据和结果文件
- imgs: 绘制的训练图像
- logging: 日志文件
- model: 模型权重文件
- notebook: 演示文件
- src: 代码部分

## 运行方式
确认具有环境依赖后，运行 notebook 下的 SN_demo.ipynb 或 SN_pair_demo.ipynb 即可。
前者和后者对应不同的训练方式，前者是自监督学习头部直接输出类概率，后者则输出高维表示并和项目嵌入做内积来产生预测概率。

## 超参数调整
在 configs 中新建文件并键入自己的超参数设置，然后在 jupyter 文件中的第3个代码单元格修改配置文件的导入路径即可。所有的默认超参数见 src/config/defaults.py。

## 主要思路介绍
数据预处理将数据划分为若干长度50的窗口，并采用左填充补全。同时，为了保证样本的充足性，对于每个用户，我们至少生成10组训练样本。我们使用每个用户最后一次交互的图书以及之前的50个交互样本作为验证集，这样验证集和测试集具有最相似的结构。

模型层面，我们使用了三种模型：GRU4Rec 、 SAS4Rec 和 NextItItem。其中初步的试验表明 SAS4Rec 相较于另两者具有微弱的优势，受限于算力，后续的调参等都围绕 SAS4Rec 展开。我们尝试了两种训练方式：一是最后一层直接输出每个 item 的类概率，二是最后一层得到隐藏表示，再和项目嵌入做内积，将内积的数值大小作为类概率。训练采用交叉熵损失，并且使用全部的正负样本，因为这样收敛的速度要快得多。

## 我们的结果
截至2024年6月20日。我们在CCF图书推荐系统比赛中取得了排行榜第3名的好成绩。使用的超参数配置为 DROP_1.yaml，结果文件为 drop_submission_1.csv。
![排行榜展示](result.png)

## 备注
- 数据由 DataFountain 比赛官方直接提供，最早的来源可追溯到公开数据集Goodbooks-10k。
- 预处理后的训练集数据 data/train.pt 文件由于过大而仅上传了压缩文件，请自行解压 data/train.zip 得到 train.pt。
- 最新的更改中删除了大量提交结果文件，仅保留 data/SAS_submission_2.csv，这不是最优的结果。
- 2024年春季学期参与同场比赛的队伍请慎重照搬我们的代码，可以使用如下引用：

```bibtex
@Misc{CCFbookrec,
  title =        {CCFbookrec},
  author =       {a team from USTC},
  howpublished = {\url{https://github.com/NameIsHardToHave/DP-final-hw}},
  year =         {2024}
}
```
