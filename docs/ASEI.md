# ASEI 使用说明

实现依据工作区的 `In1_1_3.md`，入口为 `self_impl.ASEI`。数据加载、测试边界、评价函数和结果汇总沿用 TAB。ASEI 在一张 CUDA GPU 上运行；没有 CUDA 时，`device="auto"` 回退至 CPU。

## 安装与运行

在 TAB 仓库根目录运行。服务器先安装适配的 CUDA 版本 PyTorch，再安装依赖：

```bash
uv pip install -r requirements-asei.txt
mkdir -p log

nohup uv run python -u scripts/run_benchmark.py \
  --config-path unfixed_detect_label_multi_config.json \
  --model-name self_impl.ASEI --data-name-list SWAT.csv \
  --model-hyper-params '{"batch_size":128,"lr":0.001,"num_epochs":50,"win_size":63,"predictor_block_size":16,"alpha":0.01,"log_interval_seconds":30,"checkpoint_dir":"result/asei_checkpoints/SWAT_label_seed42"}' \
  --seed 42 --eval-backend sequential --gpus 1 --num-workers 1 \
  --save-path label/ASEI_SWaT_seed42 \
  > log/asei_swat_label_seed42.log 2>&1 &
```

上述任务结束后运行分数评测：

```bash
nohup uv run python -u scripts/run_benchmark.py \
  --config-path unfixed_detect_score_multi_config.json \
  --model-name self_impl.ASEI --data-name-list SWAT.csv \
  --model-hyper-params '{"batch_size":128,"lr":0.001,"num_epochs":50,"win_size":63,"predictor_block_size":16,"alpha":0.01,"log_interval_seconds":30,"checkpoint_dir":"result/asei_checkpoints/SWAT_score_seed42"}' \
  --seed 42 --eval-backend sequential --gpus 1 --num-workers 1 \
  --save-path score/ASEI_SWaT_seed42 \
  > log/asei_swat_score_seed42.log 2>&1 &
```

`--gpus 1` 暴露物理 GPU1，模型在其逻辑 `cuda:0` 上执行。CPU 运行可在 JSON 中加入 `"device":"cpu"`。模型不接受 `multi_gpu` 和 `device_ids`，不创建训练副本。`--num-workers` 是 TAB 的实体任务设置；模型 DataLoader 工作进程数由 JSON 的 `num_workers` 控制，默认 0。

也可使用入口脚本：

```bash
ASEI_GPU=1 ASEI_SEED=42 bash scripts/run_asei.sh label --data-name-list SWAT.csv
ASEI_GPU=1 ASEI_SEED=42 bash scripts/run_asei.sh score --data-name-list SWAT.csv
```

脚本默认采用文档起始配置：预测器学习率 0.001、最多 50 轮、W=63，评分模块 30 轮，补偿 MLP 20 轮。`ASEI_HYPER_PARAMS` 可替换默认 JSON。两种评测分别训练一次；模型文件包含全部评分检查点和运行统计量。

## 数据划分

TAB 根据 `dataset/anomaly_detect/DETECT_META.csv` 的 `train_lens` 保留官方训练/测试边界。ASEI 不读取测试数据或标签以拟合模型、选择配置或估计阈值。

| 数据部分 | 划分与用途 |
|---|---|
| 官方测试集 | 原始 train_lens 边界之后的数据，模型冻结后评价 |
| D_tr | 没有独立正常验证集时，取官方正常训练部分的前 80% |
| D_val | 取官方正常训练部分的后 20%；用于正常预测早停、辅助配置选择和最终正常阈值 |
| 预测器拟合段 | D_tr 的前 80%；拟合标准化、滞后图和预测器，并抽取辅助基础轨迹 |
| 校正器拟合段 | D_tr 的随后 10%；拟合线性 AR、补偿统计量和共享 MLP |
| 独立正常参考段 | D_tr 的最后 10%；完整校正器冻结后建立 d/r 分布和中心化统计量 |

官方正常训练部分有 495000 点时，D_tr 为 396000 点，D_val 为 99000 点；D_tr 内部分别为 316800、39600、39600 点。先按原始时间轴划分，再构造窗口；各段缓存从自身开始，不跨段借用观测或拼接实体。

基准另有正常验证序列时，支持 `detect_fit(normal_train, valid_data=normal_validation)`：验证序列保持独立，不再从 normal_train 中扣出 20%；训练内部仍按 80/10/10 划分。train_label 不用于拟合，输入正常训练数据需满足方法假设。

## 历史缓存与参考样本

令 `H=W+ell_max-1`，以下偏移均为片段内零基索引：

| 输出或统计量 | 首个有效索引 | 默认 W=63、ell_max=16、P=4、R=8 |
|---|---:|---:|
| 预测误差 d | H | 78，即第 79 个观测 |
| 配对参考 d/r | H+P | 82，即第 83 个观测 |
| 完整在线分数 | D_raw=H+P+R-1 | 89，即第 90 个观测 |

独立参考段使用自身前 H 个原始观测、随后 P 个误差建立预测和 AR 缓存，收集 `N_ref-H-P` 个对齐的 d/r 样本。参考数组不需要滚动证据窗口 R；R 只参与完整评分缓存。中心化均值在同一批有效参考点上计算。元数据记录 reference_warmup、reference_first_observation、reference_points 和 score_warmup。

正常验证阈值使用依赖完整后的有效分数。独立测试从零状态启动，前 D_raw 点不报警，分数填充理论下界以保留完整时间轴，这些时间仍计入 ADD。流式输入延续历史及状态。

## 辅助扰动与固定验证实例

正常和扰动轨迹共享前缀及事件后的原始观测。事件标签保持预定区间，首末点必须确有观测变化。局部常值导致错位无效时，程序从同一正常片段重新抽取有效基础轨迹，保留扰动族、子类型、事件布局和幅度。

错位任务先用每个事件首末点的滞后变化掩码筛除必然无效的位置，再按任务自身种子确定候选顺序。困难正常任务的重采样仍优先取有效候选中正常状态证据较高的轨迹。最多尝试 perturbation_base_attempts=128 个新基础轨迹；没有有效位置或尝试耗尽时明确报错，不把未改变的轨迹当作异常。

验证种子固定为 1001、1002。默认 72 个实例覆盖 3 个扰动族、4 个情景、3 个幅度和 2 个种子。位置、最终扰动种子、事件布局及重采样次数在预测器训练前确定并记录，不依赖训练种子、候选、检查点或测试数据。相同任务描述可以重建相同扰动数组。

## 短序列与逐实体选择

按 `s=1,0.5,0.25,0.125` 检查正常预测器拟合段和验证段长度，选取能完整容纳训练轨迹及全部验证情景的最大档位。s 不参与性能选优，同一实体和模型参数下各训练种子、候选使用相同档位。

| 有效设置 | 默认时间预算的缩放公式 |
|---|---|
| 前缀完整更新数 | ceil(640*s)，其前另有完整原始缓存 |
| 恢复后缀 | D_raw+ceil(679*s) |
| 充分分离间隔 | D_raw+ceil(1319*s) |
| lambda_min / max / init | 0.8**(1/s) / 0.98**(1/s) / 0.9**(1/s) |
| 事件长度、密集和再发对内间隔 | max(1,floor(原长度*s))，长度集合去重 |

缩放不改变原始数据、W/P/R、最大滞后、三类划分或独立参考段。短/长/密集情景各保留 3 个事件，再发保留两对。最小档仍无法容纳，或拟合、参考段无有效样本时明确报错。张量批次的掩码对齐不增加真实样本，不用于补足辅助轨迹。

每个实体独立训练六组 `(kappa,gamma_rank)`：`(0.01,0.5),(0.01,1),(0.02,0.5),(0.02,1),(0.04,0.5),(0.04,1)`。保存第 5、10、15、20、25、30 轮权重。训练目标为 `L_det+gamma_rank*L_rank`，验证始终用 `L_det+L_rank`，按扰动族和情景等权汇总；并列先取较早检查点，再取较早配置。MLP 仅在同一实体内跨通道及预测分支共享。

选定模型后，正常验证序列顺序输出 N 个有效分数。固定 alpha=0.01，K=floor(alpha*N)，阈值为升序分数第 N-K-1 项，严格使用 S>h。允许负阈值，同分时实际误报可小于 K。保存 N、K、阈值和实际计数。

## 优化与参数

| 参数 | 含义与默认值 |
|---|---|
| lr / learning_rate | 预测器学习率 0.001；同时指定别名时须一致 |
| num_epochs / predictor_epochs | 预测器最多 50 轮，正常验证早停 patience=5 |
| win_size | 自历史感受野 W；默认五层 TCN 的 63，可指定正整数 |
| batch_size | 每批最多 128 个真实预测目标 |
| predictor_block_size | 默认 1，脚本采用 16 以复用连续目标的卷积计算 |
| correction_epochs / score_epochs | 20 / 30 |
| correction_learning_rate / score_learning_rate | 均为 0.001 |
| validation_cache_mb | 固定验证证据的设备缓存上限，默认 256 MiB，超出部分临时写盘 |
| recurrence_backend | scripted，reference 保留原循环作为对照 |
| device | auto，自动选择一张 CUDA GPU 或 CPU，可显式指定 cpu、cuda:0 |
| log_interval_seconds | 默认 30 秒 |

win_size 构造精确因果感受野；W=100 时原始依赖为 126 点，恢复和分离时长随之调整。分块模式保证每轮每个有效目标恰好一次，尾块对齐位置不计入损失。同一目标集合上的输出、Huber 损失和梯度与逐点窗口一致，但随机批内组合改变；设为 1 可保留原采样。

冻结预测器后的误差、校正、上尾查询和滚动证据留在同一设备，避免这些步骤间来回传输完整数组。MLP 训练复用驻留历史和目标，只传小批索引。固定验证证据每实体只构建一次，供所有候选及检查点复用；缓存不足时临时写盘，实体结束后清理，不跨实体共享。

评分采用脚本化顺序递推及显式一阶伴随，保持式（13）的顺序、ReLU 零点的零次梯度和最大通道并列均分梯度，避免逐时刻构建 Python 自动微分图。reference 后端用于数值、梯度及性能对照。

线性 AR 使用 float64 全批量投影梯度下降，Huber 阈值 1，权重衰减 0，最多 1000 次，首次步长 0.01、回溯因子 0.5，后续从上轮成功步长两倍开始。beta 投影到半径 0.95 的 L1 球，截距不投影，检查投影后的标准二次上界下降条件。映射无穷范数不超过 1e-5 时收敛，否则到上限后记录未收敛。冻结线性参数后才训练 MLP。

## 评价与检查点

使用 TAB 原有的 unfixed_detect_label_multi_config.json 和 unfixed_detect_score_multi_config.json，报表包含全部 16 个二值指标及 6 个分数指标。auc_pr 保持 Average Precision，ADD 使用原始在线报警。同一次评价中的 VUS_ROC 和 VUS_PR 共享一次 generate_curve 调用，公式及结果不变，缓存不跨序列或评价调用。

```python
from ts_benchmark.baselines.self_impl import ASEI

model = ASEI(device="auto", predictor_block_size=16)
model.detect_fit(normal_train, valid_data=normal_validation)
labels, scores = model.detect_label(test)
model.save("result/asei.pt")

loaded = ASEI.load("result/asei.pt", device="cpu")
loaded.reset_state()
labels_1, scores_1 = loaded.detect_stream(chunk_1)
labels_2, scores_2 = loaded.detect_stream(chunk_2)
```

detect_score 返回 (scores,scores)，detect_label 和 detect_stream 返回 (labels_dict,scores)。前两者每次重置序列，流式接口保留状态。检查点保存模型、预处理、参考数组、协议、全部评分检查点和阈值，可在 CPU 与单卡之间加载。元数据包含数据 SHA256、划分边界、历史就绪偏移、缓存构建数、阶段耗时、依赖版本和 AR 收敛状态。

## 验证与日志排查

```bash
uv run python -B -m unittest discover -s tests -p 'test_asei*.py' -v
uv run python -B scripts/validate_asei.py --device cpu --output result/asei_validation/summary.json

# 有一张 CUDA GPU 时补充检查。
CUDA_VISIBLE_DEVICES=1 uv run python -B -m unittest discover -s tests -p 'test_asei*.py' -v
uv run python -u scripts/check_asei_runtime.py --gpus 1 --channels 51 --win-size 63 \
  --batch-size 128 --batches 2 --block-sizes 1 16 --score-steps 256
```

测试覆盖划分、参考就绪、局部常值重采样、验证复现、单样本尾批、设备证据与 NumPy 对照、递推输出及一阶梯度（含并列和零点）、缓存复用、VUS 数值和 TAB 报表链路。本地记录见 [ASEI_validation.json](ASEI_validation.json)。合成小样本减少了训练轮次，不是完整基准效果结果。

日志标识为 asei-single-device-2，显示加载路径、参数、阶段、批次和心跳。TAB 的 scheduling 0/1 统计完整实体任务。提供的服务器日志在最后一批出现 forward() missing history，来自多卡分发中的空输入副本；本实现直接在单卡接收该尾批次。

更新服务器时同步完整 ASEI 目录、评价模块、配置和脚本，重新启动任务以采用新代码。
