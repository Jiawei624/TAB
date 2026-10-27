# ASEI 使用说明

ASEI 实现依据工作区的 `In1_1_3.md`。入口为 `self_impl.ASEI`，直接接入 TAB 的数据加载、模型训练、异常检测、指标计算和 CSV 汇总流程。

## 服务器运行

在 TAB 仓库根目录运行；数据沿用 `dataset/anomaly_detect/` 及 `DETECT_META.csv`，由 `train_lens` 保留原始训练/测试边界。先安装服务器适配的 CUDA 版本 PyTorch，再安装最小依赖：

```bash
python -m pip install -r requirements-asei.txt

# 默认采用示例超参数及 GPU1 主卡、GPU0 副卡；两项任务按顺序运行。
bash scripts/run_asei.sh score --data-name-list YOUR_SERIES.csv
bash scripts/run_asei.sh label --data-name-list YOUR_SERIES.csv
```

这两个命令分别执行 TAB 的连续分数评测和二值报警评测，每条序列各自训练一次。`scripts/multivariate/{score,label}/ASEI.sh` 提供同样的入口。每种模式的结果写入 `result/{score,label}/ASEI/seed0/`，完整模型默认保存到 `result/asei_checkpoints/`。相同数据、种子和参数再次运行会覆盖对应模型文件；检查点文件名包含种子、数据 SHA256 前 16 位和参数哈希，独立实验也可指定不同目录。

可直接使用 TAB 命令：

```bash
python -B scripts/run_benchmark.py \
  --config-path unfixed_detect_label_multi_config.json \
  --model-name self_impl.ASEI \
  --model-hyper-params '{"batch_size":128,"lr":0.0001,"num_epochs":10,"win_size":100,"multi_gpu":true}' \
  --gpus 1 0 --eval-backend sequential --num-workers 1 \
  --seed 0 --save-path label/ASEI/seed0
```

`--gpus 1 0` 按顺序暴露设备，因此模型的逻辑 `cuda:0` 对应物理 GPU1。预测器、补偿网络和评分候选使用 `torch.nn.DataParallel` 训练。线性 AR 用 float64 全批量投影梯度下降，将样本分到两卡计算 Huber 损失及梯度之和，再在主卡按全体样本归一化后更新参数。评分训练中的一对正常/扰动轨迹始终位于同一副本，单条轨迹的时间轴保持连续。所有 batch size 都表示两卡合计批量；线性 AR 覆盖全部拟合样本，不受 `correction_batch_size` 控制。

使用 sequential 后端时，TAB 依次评测实体，每个实体内部两卡训练。Ray 的 `num-workers` 控制独立评测任务，不能替代模型数据并行；这里采用 sequential，确保两张卡共同分配给同一个模型。冻结模型后的连续推断在主卡执行。

调整参数或训练种子：

```bash
export ASEI_HYPER_PARAMS='{"batch_size":64,"lr":0.0001,"num_epochs":10,"win_size":100,"multi_gpu":true,"checkpoint_dir":"result/asei_checkpoints"}'
ASEI_SEED=2 bash scripts/run_asei.sh label --data-name-list YOUR_SERIES.csv

# 文档中的五个训练种子，各自产出 TAB 报表。
for seed in 0 1 2 3 4; do
  ASEI_SEED="$seed" bash scripts/run_asei.sh score --data-name-list YOUR_SERIES.csv
done
```

单卡或 CPU 运行需显式传入 `multi_gpu=false`。`device` 默认 `auto`；指定 `device="cpu"` 可强制 CPU。`device_ids` 是可见设备内的逻辑编号，默认 `[0,1]`。显存不足时先减小 `batch_size`、`correction_batch_size`、`auxiliary_batch_size` 或 `inference_chunk_size`，多卡训练的三个批量参数均至少为 2。

## 可指定超参数

`--model-hyper-params` 接受与 AnomalyTransformer 相同形式的 JSON：

```json
{"batch_size":128,"lr":0.0001,"num_epochs":10,"win_size":100,"multi_gpu":true}
```

| 参数 | ASEI 中的含义 | 未指定时 |
|---|---|---|
| `batch_size` | 预测器的全局训练批量 | 128 |
| `predictor_block_size` | 每个训练窗口连续预测的点数；可指定 16 减少重叠窗口的重复计算 | 1；启动脚本采用 16 |
| `log_interval_seconds` | 各阶段状态日志及心跳间隔，单位秒 | 30 |
| `lr` | `learning_rate` 的别名，控制预测器学习率 | 0.001 |
| `num_epochs` | `predictor_epochs` 的别名，控制预测器最大训练轮数，仍适用早停 | 50 |
| `win_size` | 自历史编码器的实际感受野 W | 原始 TCN 的 63 |
| `multi_gpu` | 启用可见设备内的数据并行 | false |

`lr`、`num_epochs` 分别与详细参数 `learning_rate`、`predictor_epochs` 对应。同一 JSON 同时指定别名及详细参数时，数值必须一致。补偿 MLP 继续使用 `correction_learning_rate=0.001`、`correction_epochs=20`，评分模块继续使用 `score_learning_rate=0.001`、`score_epochs=30` 和第 5/10/15/20/25/30 轮检查点；这些高级参数也可在 JSON 中单独指定。线性 AR 使用独立的投影优化设置。

指定 `win_size` 时，编码器按该 W 构造精确的因果感受野：保留能完整放入窗口的基础卷积层，再用宽度 2 的因果卷积补足所需历史。`win_size=100` 对应 5 个原有卷积层和 1 个膨胀率 37、宽度 2 的卷积层。关系分支仍使用正滞后历史；默认最大滞后 16、P=4、R=8 时，预测历史为 115 点，完整原始依赖为 126 点。数据划分和独立参考片段用途保持原配置。

`win_size` 改变后，未显式指定的 `recovery_suffix` 和 `validation_gap` 会增加原始依赖长度的变化量，以保留 679 次恢复更新和 1319 次充分分离更新，再应用短序列缩放。显式指定的高级时长参数按传入值执行。实际 W、卷积布局、历史长度和预测器训练参数写入模型元数据。

分块模式将所有有效预测时刻划为不重叠的连续目标块，块顺序随机，每轮每个真实目标仍恰好使用一次；`batch_size=128,predictor_block_size=16` 表示每批 8 个块、最多 128 个预测点。块内仍严格用过去观测预测当前点。尾块只对张量空位置作带掩码的对齐，补齐位置不参与损失，也不补充任何真实训练、参考或辅助轨迹样本。相同目标集合上的预测、Huber 损失及参数梯度与逐点窗口计算一致；分块会改变随机批内目标组合，因此训练轨迹不保证逐次相同。设为 1 可保留原来的逐点窗口采样。`batch_size` 须能被块长整除，双卡训练每批至少有两个块。该选项只改变预测器的计算与批次组织，评分模块的时间递推保持原实现。

## 评测口径

直接使用 TAB 原有的 `unfixed_detect_label_multi_config.json` 和 `unfixed_detect_score_multi_config.json`。两份配置均采用 `metrics="all"`，`report_metrics` 也列出对应策略注册的全部指标，逐序列记录及汇总 CSV 均包含完整指标集合，计算函数保持 TAB 实现：

| 模式 | TAB 指标 |
|---|---|
| score | auc_roc、auc_pr、R_AUC_ROC、R_AUC_PR、VUS_ROC、VUS_PR |
| label | accuracy、f_score、precision、recall、adjust_accuracy、adjust_f_score、adjust_precision、adjust_recall、rrecall、rprecision、precision_at_k、rf、affiliation_f、affiliation_precision、affiliation_recall、add |

`auc_pr` 使用 TAB 原有 Average Precision 实现。ADD 使用原始二值报警，点调整指标沿用 TAB 的单独计算。分数保留式（13）的符号，当前 TAB 指标可以直接接收，无需用测试集进行归一化。配置仍采用 TAB 原有的跨序列均值和缺失值处理。

每个实体采用 `alpha=0.01`，独立估计阈值。模型选定后，在未扰动正常验证序列上从零状态顺序运行完整评分流程；原始依赖就绪后的有效分数数为 `N`，取 `K=floor(0.01*N)`，阈值为升序分数数组的第 `N-K-1` 项（零基索引）。报警严格使用 `score > threshold`，允许负阈值。同分可能使实际误报数小于 `K`；`N<100` 时 `K=0`，阈值为正常验证分数最大值。标定不读取测试数据。

检查点元数据的 `calibration` 保存 `N`、`K`、`threshold`、`false_alarm_count` 及实际误报比例。二值输出字典及 TAB 的 `typical_anomaly_ratio` 列仍用百分数键 `1.0`，表示目标正常点误报比例，不是测试异常占比。

测试标签只由 TAB 用于评测，ASEI 不使用传入的 `train_label` 拟合参数。输入训练部分需要满足方法的正常数据假设，代码不根据标签删除时刻或拼接不连续历史。

## 训练与初始化

未提供独立正常验证集时，TAB 提供的训练部分按时间取前 80% 训练、后 20% 验证。前一部分内部再按 80/10/10 划分预测器拟合、校正器拟合、独立参考三个片段。标准化统计量和滞后图只来自预测器拟合段；完整校正器冻结后才建立参考分布。各段均从自身历史开始构造有效样本，不跨片段借用观测。直接调用时支持 `detect_fit(train_data, valid_data=normal_validation)`。

未指定 `win_size` 时，默认编码器感受野为 63，原始依赖历史为 89 个观测。每条独立测试序列从零状态启动，前 `D_raw` 个观测不报警，连续分数填充理论下界 `-B-max(kappa_values)`，完整时间轴保留，因此这些时刻不会被从 ADD 中删除。正常验证阈值只统计依赖完整后的有效分数，不使用该填充值。辅助前缀更新开始前须另外建立完整原始历史缓存，长度不随短序列缩放。

训练顺序为正常预测器、受约束线性校正器、冻结线性参数后的共享补偿 MLP、独立参考估计和辅助评分学习。参数初值、网络结构、经验上尾的相等值处理、分支先融合后中心化、门控投影和投影前读出均按方法文档实现。梯度使用 ReLU 零点的零次梯度和 `torch.amax` 的并列均分。

评分训练默认每轮抽取 128 个正常/扰动配对片段，三个扰动族等概率，训练事件长度使用文档的去重集合。一半均匀抽样，一半从 16 个候选基础轨迹的正常状态证据均值最高四分位抽样。所有候选共享同一批已冻结证据，临时特征只保留当前批量。事件判别和排序目标分别按事件权重、正常点权重和每事件 32 个点对计算，前缀只形成状态，梯度沿完整轨迹传播。

验证使用种子 1001、1002，共 72 个固定实例：3 个扰动族 × 4 个情景 × 3 个幅度 × 2 个种子。短、长、密集情景各放置 3 个事件；成对情景放置两对事件，第二事件沿用第一事件通道组。按扰动族和情景等权汇总。对 `kappa_values × rank_weights` 的六组配置，在第 5、10、15、20、25、30 轮用 `L_det+L_rank` 选择；并列取较早轮次，再取较早配置。

各实体分别拟合模型、选择配置/检查点并估计阈值。补偿 MLP 在同一实体内跨通道和预测分支共享，不在实体之间共享。固定配置顺序为 `(kappa,gamma_rank)`：`(0.01,0.5)`、`(0.01,1)`、`(0.02,0.5)`、`(0.02,1)`、`(0.04,0.5)`、`(0.04,1)`。训练目标为 `L_det+gamma_rank*L_rank`；验证目标始终为 `L_det+L_rank`。每个实体保留全部 36 个评分检查点，其验证分组损失及所选结果均记录在该实体的检查点中。

## 短序列协议

在预测器训练前，按 `s=[1,0.5,0.25,0.125]` 从大到小检查正常预测器拟合片段和验证片段的连续长度，取能完整容纳最长训练轨迹及全部固定验证情景的最大值。选择只使用片段长度和由正常校正片段确定的 `T0`，不依赖随机种子或验证性能。同一实体、同一模型超参数下，所有候选配置和训练种子使用同一档位。缩放期间原始数据、划分、配置的 W、P、R、最大滞后及独立正常参考片段均保留。

| 有效设置 | 公式 |
|---|---|
| 原始缓存历史 | `D_raw=W+最大滞后-1+P+R-1`；默认 89，`win_size=100` 时为 126 |
| 正常前缀更新数 | `ceil(640*s)` |
| 恢复后缀点数 | `D_raw+ceil(679*s)` |
| 充分分离及再发对间间隔 | `D_raw+ceil(1319*s)` |
| λ 下界 / 上界 / 初值 | `0.8**(1/s)` / `0.98**(1/s)` / `0.9**(1/s)` |
| 事件长度、密集间隔、再发对内间隔 | `max(1,floor(原长度*s))` |

缩放后的长度候选集合去重。短、长、密集情景仍各有 3 个事件，按对应去重集合循环取长度；再发情景保留两对事件，确保全部多事件情景都存在。扰动幅度与错位延迟保持原配置。所有情景均包含完整事件和恢复段；候选不足不会通过填充或跨实体拼接补足。最小档仍无法容纳，或预测器、校正器、参考片段没有有效样本时，在训练前明确报错，TAB 将错误写入 `log_info`。

`metadata_["auxiliary_protocol"]` 保存实际档位、时长、长度集合和 λ 参数，`protocol_feasibility` 保存从大到小的长度检查。程序不修改基准配置对象，因此重复拟合及其他实体不继承前一个实体的缩放值。显式架构或小规模测试配置使用其实际 `D_raw`；默认配置严格对应上表。

## 线性 AR 优化

线性主干以全体有效正常校正样本的 Huber₁ 均值为目标，使用 float64 全批量投影梯度下降，权重衰减为 0，最多 1000 次迭代。`beta` 初始化为 0，截距初始化为该片段正常预测误差的均值。预测器默认学习率为 `1e-3`，可用 `lr` 指定；补偿 MLP 和评分模块各有独立学习率参数。线性 AR 不使用 Adam/AdamW。

首轮试探步长为 0.01，后续从上一轮成功步长的两倍开始。对每个候选点，将每个通道/预测分支的 `beta` 投影到半径 0.95 的 L1 球，截距不投影；使用排序与软阈值求真正的欧氏投影。检查投影后候选点 `y` 是否满足 `f(y) <= f(x)+<grad f(x),y-x>+||y-x||²/(2*t)`，不满足则将步长减半，float64 比较仅保留机器舍入级容差。

返回参数处的投影梯度映射无穷范数不超过 `1e-5` 时标记收敛；迭代达到上限且未满足条件则标记 `converged=false`。随后冻结线性参数，再训练共享补偿 MLP。线性参数在训练、参考估计、推断和保存/加载时均保留 float64，MLP 权重使用 float32。

`metadata_["corrector"]["ar_optimization"]` 记录实际优化设置、迭代数、步长、回溯次数、最终映射范数、收敛标志及二次上界检查轨迹。`maximum_beta_l1` 记录实际最大系数 L1 范数。

## 流式推断与检查点

```python
from ts_benchmark.baselines.self_impl import ASEI

model = ASEI(multi_gpu=True)
model.detect_fit(normal_train)
labels, scores = model.detect_label(test)  # 独立序列，从零启动
model.save("result/asei.pt")

loaded = ASEI.load("result/asei.pt", device="auto")
loaded.reset_state()
labels_1, scores_1 = loaded.detect_stream(chunk_1)
labels_2, scores_2 = loaded.detect_stream(chunk_2)  # 延续历史缓存和时序状态
```

`detect_score` 返回 `(scores, scores)`，`detect_label` 和 `detect_stream` 返回 `(labels_dict, scores)`。前两者每次开始独立序列，流式接口保留状态。`last_ready_mask_` 标识当前批次中依赖完整的时刻。检查点格式为 2，保存预处理统计量、三组模型参数、图、校正输入统计量、固定补偿幅度、参考数组、中心化均值、实体缩放协议、全部评分检查点、阈值、正常验证分数及运行元数据，权重键不带 DataParallel 前缀，可在 CPU、单卡和双卡间加载。检查点加载从新序列开始。

元数据记录数据 SHA256、划分边界、参考样本数、有效正常验证点数、经验误报比例、时间尺度、每个固定验证任务及依赖版本。采样周期和真实数据版本名称需要由实验记录另行给出。

## 验证

```bash
# 小规模真实训练、公式和 TAB 链路检查，不运行完整基准。
python -B -m unittest discover -s tests -p 'test_asei.py' -v

# 服务器双卡检查；包含不等长批尾的输出/梯度对照及真实四阶段双卡拟合。
CUDA_VISIBLE_DEVICES=1,0 python -B -m unittest discover -s tests -p 'test_asei.py' -v

# 保留默认网络结构的两个合成实体，生成实际配置及 AR 收敛状态报告。
python -B scripts/validate_asei.py --output result/asei_validation/summary.json
```

测试包含阈值误报预算、L1 欧氏投影、回溯二次上界、收敛/未收敛记录、四档长度可行性、逐实体选参隔离，以及 TAB 全指标与 CSV 链路。小规模真实训练显式使用更小架构并减少训练轮次，仅检查执行，不作为方法效果证据。CPU 环境自动跳过双卡测试。

配置与实测记录见 [ASEI_validation.json](ASEI_validation.json)。报告脚本保留默认网络结构、数据划分、独立参考机制和 AR 的 1000 次上限，预测器、补偿 MLP 和六组评分候选各只训练 1 轮，并只保存第 1 轮评分检查点。完整 36 个评分检查点的枚举和并列顺序另由受控损失测试检查。报告如实保存达到上限但未收敛的标志，不能将小规模检查用作基准效果结果。

DataParallel 的批维拆分和主卡参数要求参见 [PyTorch 官方文档](https://docs.pytorch.org/docs/2.12/generated/torch.nn.DataParallel.html)。

## 长时间没有日志时的定位

TAB 的 `scheduling ASEI: 0/1` 统计完整实体任务，模型内部训练不会推进这一外层计数。大实体可长时间保持 0/1。启动日志中的 `W`、训练轮数、批次数和 `predictor_block_size` 用于核对实际参数；未指定 `num_epochs` 时预测器最多训练 50 轮。

模型启动时打印实现标识 `asei-progress-blocks-1` 和实际加载路径。各阶段在开始、首批次、结束以及默认每 30 秒输出进度，含当前阶段、已完成批次、耗时、吞吐和剩余时间估计。即使尚未完成一个批次，后台心跳仍报告当前 Python 阶段，例如取数、传输、前向或反向，`since_progress_s` 表示距离上次完成进度的时间；心跳不增加完成计数，也不代表 CUDA 一定正常。阶段耗时写入 `metadata_["stage_seconds"]`。

按默认 W=63、batch_size=128，316800 点预测器拟合片段有 316722 个有效训练目标，需 2475 个批次；99000 点验证片段需 773 个批次。完整训练和验证均完成后才产生 epoch 汇总。实际速度取决于硬件，GPU 活跃与进程存活不足以单独确认或排除死锁。训练阶段由 label 和 score 两种模式共用，label 指标尚未参与该阶段。

独立检查少量与 SWaT 同通道数的合成批次：

```bash
uv run python -u scripts/check_asei_runtime.py \
  --gpus 1 0 --channels 51 --win-size 63 --batch-size 128 \
  --batches 2 --block-sizes 1 16
```

该命令不读取数据集，只检查两种窗口组织的前向和反向运行，并输出耗时及当前 PyTorch/设备信息。它用于定位环境与计算阶段，不是完整实验或正式性能比较。CPU 检查可增加 `--device cpu`。

更新服务器时同步整个 `ts_benchmark/baselines/self_impl/ASEI/` 目录及需要使用的脚本。已经启动的 Python 进程继续执行其已加载代码，更新文件后须重新启动任务才能采用新实现。两个双卡基准任务按顺序运行。
