# DynamicNeighborhoodStat

本方法实现 `In1_5.md` 的完整主模型：动态多尺度邻域重构、单变量因果统计预测和末点预测，经正常验证集完成支持验证、误差校准与阈值估计。模型注册名为 `self_impl.DynamicNeighborhoodStat`。

评价使用 TAB 现有的 `unfixed_detect_label` 和 `unfixed_detect_score` 策略。标签评价采用固定的验证阈值；分数评价中的 `auc_pr` 即 average precision（AP）。两个入口复用同一检查点，不使用测试标签选配置。

## 环境与数据

在 `TAB/` 仓库目录运行。主方法及现有评价入口所需依赖汇总在：

```bash
python -m pip install -r ts_benchmark/baselines/self_impl/dynamic_neighborhood_stat/requirements.txt
```

使用服务器已有的 CUDA 版 PyTorch 可进行 GPU 训练。主方法不需要下载预训练权重。独立依赖文件覆盖本方法和现有评价入口；其他基线仍按 TAB 的完整 requirements.txt 配置。

数据沿用 TAB 的长表 CSV 和 `dataset/anomaly_detect/DETECT_META.csv`。`train_lens` 指定官方正常训练段，模型内部再按原始顺序分为前 80% 训练、后 20% 验证，各集合独立构窗。

令内部划分位置为 s，窗口长度为 L。最后一个训练窗口为 `[s-L,s)`，第一个验证窗口为 `[s,s+L)`；这里使用零基、左闭右开的位置区间。第一个验证评分端点为 `s+L-1`。验证段前 L-1 点仍参与构窗，不单独形成完整窗口的评分端点。边界、构窗数量和评分端点记录在训练日志及 `calibration.json` 的 `split_metadata` 中。

- 默认使用本地 `SWAT.csv`。其元数据为正常训练段 495,000 点、测试段 449,919 点。
- 每个文件视为一个连续独立实体，分别训练和校准。拼接了多个实体的数据必须先按真实实体边界拆分并登记元数据。
- 使用全部数值特征，保留离散及常量通道，排除标签和时间列。列顺序在检查点中固定。
- 不自动填补、重采样或裁切。非有限值和训练段异常标签会报错，不能删除异常行后拼接两侧时间段。
- 各变量的均值与总体标准差仅在前 80% 部分估计；标准差不大于 `1e-8` 时缩放因子取 1。
- `date` 为序号时不能据此推断物理采样间隔。ADD 默认单位是采样点。

## 服务器运行

默认跑五个种子 `42 43 44 45 46`，50 个 epoch，batch size=32，L=128。推荐同时输出标签和分数指标：

```bash
python scripts/run_dynamic_neighborhood_stat.py --data-name-list SWAT.csv --mode both --gpus 0
```

先运行一个种子：

```bash
python scripts/run_dynamic_neighborhood_stat.py --data-name-list SWAT.csv --seeds 42 --mode both --gpus 0
```

多个已确认适用的数据文件：

```bash
python scripts/run_dynamic_neighborhood_stat.py --data-name-list SWAT.csv PSM.csv --mode both --gpus 0
```

每次只加载一个实体，完成其全部种子和评价入口。`--gpus` 控制可见物理卡，`--device` 使用可见卡内的逻辑编号；单卡通常保留 `--device auto`。CPU 可用 `--device cpu`。显存不足时可以用 `--batch-size 16`；`target_chunk_size` 控制目标分块，不减少邻居。

也可以直接调用 TAB 原有入口：

```bash
python scripts/run_benchmark.py --config-path dynamic_neighborhood_stat_label.json --model-name self_impl.DynamicNeighborhoodStat --data-name-list SWAT.csv --seed 42 --gpus 0 --num-workers 1 --save-path DynamicNeighborhoodStat/label
python scripts/run_benchmark.py --config-path dynamic_neighborhood_stat_score.json --model-name self_impl.DynamicNeighborhoodStat --data-name-list SWAT.csv --seed 42 --gpus 0 --num-workers 1 --save-path DynamicNeighborhoodStat/score
```

在 Windows 的指定环境中，将命令开头的 `python` 替换为：

```powershell
& 'E:\Miniconda3\Scripts\conda.exe' run -n codex python
```

默认可复用已有校准检查点。训练数据内容、列顺序、种子或模型配置变化后会使用不同目录。多种子脚本的 `--retrain` 会重新训练每个种子的第一个入口，第二个入口继续加载该次训练的检查点。

## 运行进度与故障定位

`run_benchmark.py` 默认使用 sequential 后端。指定 `--seed 42` 只运行该种子；`--num-workers 1` 是评价 worker 设置，与模型的 DataLoader `num_workers=0` 不同。sequential 后端不实施任务超时。

TAB 的 `fix_all_random_seed()` 将全局 `torch.backends.cudnn.enabled` 设为 false。DynamicNeighborhoodStat 在训练、验证、校准和推理期间使用 `enabled=True, deterministic=True, benchmark=False` 的局部 cuDNN 上下文，包含反向传播，退出后恢复调用方设置。CUDA 张量仍需可用的 CUDA 版 PyTorch 和 cuDNN；关闭 cuDNN 本身不会把模型移到 CPU。

默认每 30 秒在完成批次后输出阶段进度，同时记录首批和末批。日志包含批次数、耗时、批次速度及当前阶段的预计剩余时间。训练历史每轮落盘。参考统计、校准残差、校准分位数及测试推理都有各自的进度。若单个批次尚未完成，进度也不会更新；此时可依据上一条阶段记录检查服务器资源。

建议在服务器上明确要求 GPU，并关闭 Python 标准输出缓冲：

```bash
mkdir -p log
nohup uv run python -u ./scripts/run_benchmark.py \
  --config-path dynamic_neighborhood_stat_label.json \
  --model-name self_impl.DynamicNeighborhoodStat \
  --data-name-list SWAT.csv \
  --model-hyper-params '{"batch_size":32,"learning_rate":0.001,"win_size":128,"num_epochs":50,"patience":7,"device":"cuda:0","log_interval_seconds":30}' \
  --seed 42 --gpus 0 --num-workers 1 --timeout -1 \
  --save-path label/DynamicNeighborhoodStat_SWaT_seed42 \
  > ./log/dynamic_neighborhood_stat_swat_label_seed42.log 2>&1 &
```

启动日志会打印实际设备、GPU 名称、cuDNN 设置及 artifact 目录。`device=cuda:0` 在 CUDA 不可用时明确报错；`auto` 可以回退到 CPU。模型目录中的 `runtime_environment.json` 保存运行环境，`runtime_progress.json` 保存最近完成的工作及更新时间。`log_interval_seconds=0` 可输出每一步进度，该值不改变检查点身份。评价异常同时打印完整 traceback 并保留在 TAB 的 `log_info` 中。

SWAT 在默认配置下每轮包含 12,372 个训练批次和 3,090 个验证批次。一条 `epoch=1` 汇总表示第一轮训练和验证均已完成；下一条整轮汇总需要等待第二轮的全部批次。阶段日志中的 ETA 根据当前实测速率计算。

仅用几批 51 变量的合成窗口检查服务器速度和设备：

```bash
uv run python -u validation/profile_dynamic_neighborhood_stat.py --channels 51 --steps 3 --device cuda:0
```

脚本保留默认 L=128 和 batch size=32，分别测量启用和关闭梯度重计算时的前向、反向耗时。添加 `--disable-cudnn --output result/DynamicNeighborhoodStat/diagnosis/profile_no_cudnn.json` 可测量 cuDNN 关闭时的速度。这里的整轮耗时外推仅用于诊断，不代表完整训练结果。

真实 SWAT 的少量连续点可用于两轮流程测试：

```bash
uv run python -u validation/smoke_dynamic_neighborhood_stat.py --device cuda:0
```

该测试默认读取 3 个变量，取官方正常训练段内 800 点和官方测试段前 192 点，覆盖独立构窗、两轮训练、校准、推理及检查点重载；结果写入 `result/DynamicNeighborhoodStat/diagnosis/swat_smoke/`。小样本不用于评价检测效果。完整检查记录见 [运行检查报告](dynamic_neighborhood_stat_runtime_audit.md)。

## 主配置

| 项目 | 默认值 |
|---|---|
| 窗口 / 邻域尺度 / 统计跨度 | `128` / `{4,8,16,32,64,128}` / `{16,32}` |
| 邻域、自身 TCN 宽度 | 均为 32；每块两次核长 3 的左因果卷积；GELU、逐位置 LayerNorm、dropout=0.1 |
| 感受野 | 邻域 61；自身按 L 自动覆盖 `L-1`，L=128 时为 253 |
| 嵌入 / 上下文 / 消息 / MLP 隐层宽度 | `8 / 32 / 32 / 64` |
| 尺度温度 / 邻域温度 | `1 / 1`，固定 |
| `lambda_n / lambda_s / lambda_a` | `1 / 1 / 0.1` |
| 反集中约束 | `r=min(2,d-2)`，`rho_a=(1+r/(d-1))/2`；d=2 时关闭正则 |
| `chi / alpha / beta` | `0.25 / 0.5 / 1` |
| 支持参数 `eta_g / epsilon_g` | `0.05 / 1e-8` |
| `epsilon_v / epsilon_q` | `1e-6 / 1e-6` |
| 能量下限 | `max(1e-8, 1e-4 * median(V_train))` |
| 能量预测范围 / 误差尺度 / 阈值 | 训练 A 的 1%—99%；验证平方残差的 95%；验证整体分数的 99% |
| 优化器 | AdamW，lr=`1e-3`，weight decay=`1e-4`，梯度范数裁剪 1 |
| 早停 | 最多 50 轮，patience=7，绝对 min_delta=`1e-4`；监控验证归一化重构损失，不计正则；恢复最佳参数 |
| 窗口与数据精度 | 所有阶段 stride=1，不丢最后一批；目标/参考/校准为 float64，网络为 float32，关闭混合精度 |

完整配置定义在 `ts_benchmark/baselines/self_impl/dynamic_neighborhood_stat/config.py`。多种子脚本可用 `--model-hyper-params` 传 JSON；例如：

```bash
python scripts/run_dynamic_neighborhood_stat.py --data-name-list SWAT.csv --seeds 42 --gpus 0 --model-hyper-params '{"target_chunk_size":8,"learning_rate":0.001}'
```

目标分块仍对全部非自环来源、全部六个尺度归一化。训练默认使用梯度重计算控制路由中间张量显存。参考统计和分位数使用完整数据；校准误差暂存磁盘，不随机抽样。SWAT 默认验证误差缓存约需 0.77 GB，校准结束后删除。

`calibration.json` 的 `freeze_energy` 按变量和统计跨度记录 `margin = a_low - log(energy_floor)`、`has_positive_bound` 及 `energy_squared_error_lower_bound = max(margin,0)^2`。这是完全冻结/线性化片段的原始 A 坐标平方误差下界，不是整体分数或必然报警的保证。margin=0 时没有正下界，正常训练包含合法冻结状态时可能出现这种情况。观测目标保留原值；近常量坐标的预测使用固定训练参考值。

已有校准检查点可以直接复用；通过 `detect_fit` 复用时，诊断报告依据实际输入和已有参考统计写出，无须重新训练或改变阈值。

## 时间对齐与指标

完整窗口的分数只归属当前末点。自身统计头在零基索引 `L-h-1` 读出，末点头在 `L-2` 读出。邻域仅汇总其他变量，支持标志只控制评分融合，不移除任何邻域来源。

输出长度与测试段一致。前 `L-1` 点分数为 NaN、有效掩码为 false；现有评价策略对声明此掩码的模型仅在有效点计算指标，不将无效位置视为正常预测。其他未声明掩码的模型沿用原有对齐行为。

- 标签指标：原始逐点 Precision、Recall、F1、FPR、ADD。报警条件为 `score > threshold`，不执行 point adjustment。
- 分数指标：AP（TAB 名称 `auc_pr`）和 AUC-ROC。
- ADD 使用原始测试标签的完整事件。与无效前缀相交或触及测试边界的事件不计入 ADD，并记录排除数；这些事件的有效点仍参与逐点指标。
- 最后一批和异常结束后的正常尾部均保留。
- 不同窗口长度或方法间比较时，应另行统一有效评价时段。

评价记录使用测试段内的零基位置，`valid_start` 为首个有效点，`valid_end_exclusive` 为末个有效点之后的位置。训练/验证的 `split_metadata` 则以官方正常训练段为位置基准，两者分别标记 `index_basis`。

## 结果与检查点

默认输出到 `result/DynamicNeighborhoodStat/`：

| 文件 | 内容 |
|---|---|
| `records/` | TAB 原始压缩评价记录，含模型参数、策略、运行时间、指标及错误信息 |
| `all_records.csv` | 本次所有实体、种子、评价入口的原始记录 |
| `per_entity_metrics.csv` | 各实体、种子、指标的实际结果及评价覆盖信息 |
| `evaluation_coverage.csv` | 每个实体、种子、评价入口的总点数、有效点数、排除点数、有效区间、完整事件数及排除事件数 |
| `per_seed_metrics.csv` | 每个种子内对实体指标等权汇总，并合计该种子各实体的点数及事件数；有实体指标缺失或失败时该组指标为 NaN |
| `summary.csv` | 种子间指标均值、样本标准差、有效与预期种子数，以及点数/事件数的种子间最小值和最大值；不填补缺失值 |
| `run_manifest.json` | 本次命令参数及结果文件路径 |

默认每个文件独立作为一个数据集汇总。多个实体属于同一数据集时，用 `--entity-groups '{"entity1.csv":"SMD","entity2.csv":"SMD"}'` 指定分组。跨实体先在每个种子内等权平均，再计算种子间统计；不把重叠窗口当作独立样本。

覆盖字段包括 `test_length`、`valid_points`、`excluded_points`、`total_events`、`complete_events`、`excluded_boundary_events`。标签和分数入口采用相同的有效区间和真实事件边界判定。`summary.csv` 中对应的 `*_min`、`*_max` 在种子间取范围，不把不同种子的重复评价相加。`coverage_seeds` 表示覆盖计数已知的种子数；有实体计数未知时，该种子的合计数保留 NaN，不按零处理。

`artifacts/<文件名>/seed_<种子>/<配置与训练内容哈希>/` 保存：

- `checkpoint.pt`：完整模型、训练统计、列顺序、支持标志、校准尺度和阈值，可直接加载。
- `training_history.csv`、`calibration.json`：训练/验证损失、集中度、最佳轮数、参数量、独立构窗边界、冻结能量误差下界及完整校准数据。
- `runtime_environment.json`、`runtime_progress.json`：设备与计算库设置，以及当前阶段的已完成数量、速度和 ETA。训练历史额外包含各轮训练与验证耗时。
- `test_predictions.csv`：位置、输入索引、有效掩码、整体分数及报警。
- `test_variable_scores.npy`：逐变量分数，默认磁盘存储，可用 `np.load(path, mmap_mode="r")` 读取。
- `test_events.csv`、`evaluation_label.json`、`evaluation_score.json`：评价侧生成的事件和指标记录，测试标签不传入模型。
- `test_diagnostics.json`：阈值、支持比例、测试集中度等摘要。

完整逐窗口路由张量默认不保存。批量评价沿原始时间顺序计算因果窗口，模型不在测试时更新。

```python
from ts_benchmark.baselines.self_impl.dynamic_neighborhood_stat import DynamicNeighborhoodStat

detector = DynamicNeighborhoodStat.load("/path/to/checkpoint.pt", device="cuda:0")
scores, _ = detector.detect_score(test_dataframe)  # ordered numeric feature columns, no label
alarms, _ = detector.detect_label(test_dataframe)
valid = detector.evaluation_valid_mask_
```

## 实现验证

```bash
python validation/validate_dynamic_neighborhood_stat.py
```

验证覆盖目标公式、参考统计、输入排除的梯度和扰动检查、零方差坐标的固定输出和梯度、固定坐标的评分一致性、条件性冻结误差下界，以及训练/验证构窗边界。微型生命周期验证覆盖训练、顺序校准、旧检查点复用、TAB 入口和评价覆盖汇总。合成运行只验证代码，不作为检测效果证据；完整数据训练在服务器进行。
