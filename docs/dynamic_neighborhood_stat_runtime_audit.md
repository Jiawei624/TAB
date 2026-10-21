# DynamicNeighborhoodStat 运行检查

检查日期：2026-10-21。方法定义来自工作区根目录的 `In1_5.md`。本地环境为 Python 3.10.20、PyTorch 2.12.0+cpu；测试没有覆盖 CUDA 内核的实际执行和加速比例。

## 服务器现象与执行位置

启动入口为 `scripts/run_benchmark.py`，配置为 `dynamic_neighborhood_stat_label.json`，SWAT、seed=42、L=128、batch size=32，最多 50 轮，patience=7。该入口默认使用 sequential 后端，只执行指定的一个种子；`--timeout -1` 不会产生超时报错。

日志已出现第一轮 epoch 汇总，说明完整 CSV 读取、正常集拆分、参考统计、第一轮训练和第一轮验证均已完成。紧接着执行第二轮训练，校准和测试尚未开始。整轮日志之间的静默不能证明程序死锁。

存在两个可从源码确认的原因：

1. `AnomalyDetect.execute()` 首先调用 `fix_all_random_seed()`；`ts_benchmark/utils/random_utils.py` 将 `torch.backends.cudnn.enabled=False`。这是进入模型前的全局设置，影响模型中的因果卷积。CUDA 上的卷积不会因此自动移到 CPU，但无法使用 cuDNN 实现。
2. SWAT 每轮需要完成 12,372 个训练批次和 3,090 个验证批次。仅依赖整轮汇总会形成很长的日志间隔。邻域路由保留所有来源，计算量随变量数近似按平方增长；默认梯度重计算也会增加计算工作。

设备为 `auto` 时还可能选择 CPU，需要依据启动环境日志判断。现有信息不足以确定服务器是否回退到 CPU，也不足以排除显存、主存或其他作业竞争导致的额外等待。

## 完整调用链与方法对应

| 阶段 | 实现位置 | 检查结果 |
|---|---|---|
| 命令解析与 GPU 可见性 | `scripts/run_benchmark.py` → `ParallelBackend` → `SequentialBackend.init()` | `--gpus 0` 在 sequential 后端也设置 `CUDA_VISIBLE_DEVICES`；`--num-workers` 不改变模型内部 DataLoader 配置 |
| 数据加载 | `pipeline()` → `LocalAnomalyDetectDataSource` → `read_data()` → `process_data_df()` | 长表转宽表，按元数据 `train_lens` 保留官方训练/测试边界；SWAT CSV 为 976,006,181 字节，读取阶段会完整物化长表 |
| 模型注册与评价 | `get_models()` → `self_impl.DynamicNeighborhoodStat` → `UnFixedDetectLabel.execute()` | 测试标签留在评价侧；正常训练标签会检查是否含异常 |
| 拆分与标准化 | `DynamicNeighborhoodStat.detect_fit()` | 495,000 个官方训练点分成 396,000/99,000 点；标准化参数仅来自前段；训练和验证分别构窗 |
| 参考统计 | `features.training_references()` | 使用完整训练窗口端点，计算粗指标、去趋势 A/C/J、能量下限、目标方差和固定输出坐标；不抽样校准 |
| 邻域路径 | `DualPathNetwork.forward()`、`_route()` | 六尺度共享单变量 TCN；目标上下文及注意力排除自身；尺度门控按来源的六尺度归一化，关系权重按全部非自环来源归一化；目标分块不裁剪邻域 |
| 自身路径 | `CausalEncoder`、统计头和末点头 | 左因果卷积与逐位置 LayerNorm；统计跨度 h 从零基位置 `L-h-1` 读取，末点从 `L-2` 读取；目标片段不进入其预测输入 |
| 损失与早停 | `DualPathNetwork.losses()` → `_epoch()` → `_train()` | 训练目标方差归一化；保留反集中正则；验证重构损失不含正则；恢复最佳轮权重 |
| 正常校准 | `_calibrate()` | 固定网络后先确定邻域支持标志，再求各坐标 95% 误差分位数，最后求整体分数 99% 阈值；误差缓存使用磁盘映射并清理 |
| 测试与汇总 | `_predict()` → `detect_label()` → 评价策略 → CSV 报告 | 稳定 log-sum-exp 汇总，固定阈值，前 127 点无效且分数为 NaN；其余点按窗口末点对齐，不做 point adjustment |
| 保存与复用 | `save()`、`_load()`、`load()` | 检查配置和训练内容指纹；标签与分数入口复用校准检查点；同一实例对相同测试数据可以复用分数缓存 |

关键公式和信息依赖有数值、梯度及扰动验证覆盖。检查中未发现邻域目标自身泄漏、统计目标片段泄漏、训练/验证跨界构窗或测试标签用于校准的情况。

## 本地短测试

所有 Python 命令均使用 `E:\Miniconda3\Scripts\conda.exe run -n codex python` 执行，没有安装额外依赖。

### 机制与 TAB 集成

`validation/validate_dynamic_neighborhood_stat.py` 的 7 项检查覆盖：

- 粗指标与统计目标公式、训练参考统计，以及自身输入排除的梯度和扰动检查。
- 近常量坐标、d=2 邻域、反向传播和梯度重计算。
- 条件性冻结能量下界、独立构窗边界和有效评分区间。
- 合成数据训练、顺序校准、检查点复用、标签/分数评价及覆盖计数。
- TAB 种子函数关闭 cuDNN 后，模型前向期间启用 cuDNN，结束后恢复调用方设置。
- 原始 `scripts/run_benchmark.py` 入口的 CSV 加载、模型加载、逐点评价和最终 CSV 报告。

### 真实 SWAT 小样本

`validation/smoke_dynamic_neighborhood_stat.py` 直接读取真实长表中的少量连续行，保留原始列名，不修改原数据或元数据。

| 项目 | 结果 |
|---|---|
| 变量 | 前 3 个通道：FIT101、LIT101、原名含前导空格的 MV101 |
| 官方正常训练段内范围 | 零基 `[50000,50800)`，共 800 点 |
| 内部训练/验证 | 640/160 点，各 513/33 个完整窗口 |
| 官方测试段内范围 | 零基 `[0,192)`，共 192 点 |
| 窗口与批量 | L=128、batch size=32 |
| 执行轮数 | 2 轮，完成第二轮训练与验证 |
| 训练和校准总耗时 | 约 7.42 秒，本地 CPU，含初始化 |
| 首次测试推理 | 约 0.09 秒 |
| 有效测试点 | 65，分数全部有限；前 127 点为 NaN |
| 检查点重载 | 分数在 `rtol=1e-12, atol=1e-12` 下相同 |

报告位于 `result/DynamicNeighborhoodStat/diagnosis/swat_smoke/smoke_report.json`。训练片段和变量数量均很小，阈值与报警数量不作为检测质量证据。

### 默认 51 变量的计算量

`validation/profile_dynamic_neighborhood_stat.py` 使用默认 L=128、batch size=32、网络宽度和目标分块。每组预热一次，再测量三个前向、损失及反向步骤，CPU 线程数为 3。

| 变量数 | 梯度重计算 | 平均每批耗时 | 窗口/秒 |
|---|---|---:|---:|
| 3 | 开 | 0.186 秒 | 172.4 |
| 3 | 关 | 0.174 秒 | 184.1 |
| 51 | 开 | 3.826 秒 | 8.36 |
| 51 | 关 | 3.819 秒 | 8.38 |

原始测量位于 `result/DynamicNeighborhoodStat/diagnosis/profile_before.json`。按本地 CPU 的 51 变量吞吐量乘以 12,372 批次，仅一轮训练约 13.15 小时，另需验证时间。该数字是诊断性外推，不能用于预测服务器 GPU 的运行时间，也不能证明某个服务器进程没有卡住。

## 当前运行控制与服务器检查

模型的 `_epoch()`、`_calibrate()` 和 `_predict()` 使用局部 cuDNN 上下文，训练的反向传播处于同一上下文内。默认配置保持确定性卷积，关闭算法基准搜索。

阶段开始、首批完成、默认每隔 30 秒完成批次，以及阶段结束时均有日志。训练/验证日志中的单位是批次，ETA 只针对当前阶段；在单个批次执行期间不会额外输出心跳。`training_history.csv` 每轮保存，并包含训练与验证耗时。模型目录另有：

- `runtime_environment.json`：实际设备、PyTorch、GPU 名称、继承的 cuDNN 设置和模型使用的设置。
- `runtime_progress.json`：最近完成的阶段工作、数量、速度、ETA、更新时间及失败状态。
- `calibration.json`、`checkpoint.pt`：完成训练和校准后形成的模型与阈值。

`--save-path` 控制 TAB 评价记录和报告目录；模型 artifact 目录由模型的 `artifact_dir` 控制，启动日志打印其绝对路径。校准需要的临时磁盘空间会在分配前记录和检查。评价异常会立即打印完整 traceback，并继续保留在结果的 `log_info` 字段。

服务器命令见 [运行说明](dynamic_neighborhood_stat.md#运行进度与故障定位)。显式设置 `"device":"cuda:0"` 可以避免 CPU 静默回退。源码中的 cuDNN 设置问题有调用链和回归检查支持，服务器的实际加速比例仍需在 CUDA 环境中测量。
