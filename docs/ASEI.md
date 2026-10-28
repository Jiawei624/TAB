# ASEI 运行与配置

实现依据工作区的 [In1_1_4.md](../../In1_1_4.md) 和 [impl.md](../../impl.md)，TAB 入口为 `self_impl.ASEI`。每个文件分别拟合预测器、校正器、评分器和阈值。模型使用一张 CUDA GPU；`device="auto"` 在 CUDA 不可用时使用 CPU。完整默认值见 [Config](../ts_benchmark/baselines/self_impl/ASEI/asei.py)，实现版本为 `asei-finite-bank-scan-3.4`。默认 `training_profile="original"` 使用方案规定的容量与预算；`"extended"` 使用下文的容量选择和较长训练预算。两者分别报告，不能混记为同一训练配置。

## 服务器运行

以下命令在 TAB 仓库根目录执行，显式启用适用于 SWaT 容量与预算检验的 `extended` 配置。先安装服务器 CUDA 环境配套的 PyTorch；其余运行和验收依赖列在 [requirements-asei.txt](../requirements-asei.txt)，不要求替换已有锁定版本。

```bash
mkdir -p log

nohup uv run python -u ./scripts/run_benchmark.py \
  --config-path unfixed_detect_label_multi_config.json \
  --model-name self_impl.ASEI \
  --data-name-list SWAT.csv \
  --model-hyper-params '{"device":"auto","seed":42,"training_profile":"extended","win_size":63,"alpha":0.01,"frozen_evidence_backend":"canonical","checkpoint_dir":"result/asei_checkpoints/SWAT_label_seed42_extended_v34"}' \
  --seed 42 \
  --eval-backend sequential \
  --gpus 1 \
  --num-workers 1 \
  --report-method csv \
  --save-path label/ASEI_SWaT_seed42_extended_v34 \
  > ./log/asei_swat_label_seed42_extended_v34.log 2>&1 &
```

另一个评测入口：

```bash
nohup uv run python -u ./scripts/run_benchmark.py \
  --config-path unfixed_detect_score_multi_config.json \
  --model-name self_impl.ASEI \
  --data-name-list SWAT.csv \
  --model-hyper-params '{"device":"auto","seed":42,"training_profile":"extended","win_size":63,"alpha":0.01,"frozen_evidence_backend":"canonical","checkpoint_dir":"result/asei_checkpoints/SWAT_score_seed42_extended_v34"}' \
  --seed 42 \
  --eval-backend sequential \
  --gpus 1 \
  --num-workers 1 \
  --report-method csv \
  --save-path score/ASEI_SWaT_seed42_extended_v34 \
  > ./log/asei_swat_score_seed42_extended_v34.log 2>&1 &
```

`--gpus 1` 将物理 GPU1 映射为逻辑 `cuda:0`。CPU 可显式指定 JSON `"device":"cpu"`。TAB 的 `--seed` 与模型 JSON 的 `seed` 在上述命令中均为 42；模型独立使用 JSON 的种子，默认是 2026。两个评测进程各自训练一遍。

CUDA 拟合在重置峰值显存统计前，先显式初始化 PyTorch CUDA 运行时并选定逻辑设备；CPU 路径不调用这些 CUDA API。启动日志包含 `ASEI CUDA startup implementation=asei-finite-bank-scan-3.4`。这一初始化顺序适用于尚未创建 CUDA 张量的新进程，避免在分配器尚未注册设备时调用 `reset_peak_memory_stats`。CUDA 初始化错误直接报告，不跳过错误继续训练。

原始两个 TAB 配置保持 `metrics="all"`：label 评测及报告包含全部 16 个原有指标，score 包含全部 6 个原有指标。VUS_ROC/PR 在一次评测内共用曲线计算，数值和指标定义不变。

入口脚本使用同一默认方法：

```bash
ASEI_GPU=1 ASEI_SEED=42 bash scripts/run_asei.sh label --data-name-list SWAT.csv
ASEI_GPU=1 ASEI_SEED=42 bash scripts/run_asei.sh score --data-name-list SWAT.csv
ASEI_GPU=1 ASEI_SEED=42 ASEI_TRAINING_PROFILE=extended bash scripts/run_asei.sh label --data-name-list SWAT.csv
```

`ASEI_HYPER_PARAMS` 可以替换脚本默认 JSON；替换时应显式保留所需的模型 `seed`。TAB 的 `--num-workers` 控制实体任务，模型 JSON 的 `num_workers` 控制 DataLoader，默认 0。

## 数据、预处理与时间索引

TAB 依据 `DETECT_META.csv` 的 `train_lens` 保留官方训练/测试边界。ASEI 不读取 `train_label`。每次输入代表同一实体的一段连续观测，不拼接独立实体补足长度。

没有外部验证输入时，设官方训练输入长 N：

| 片段 | 零基左闭右开边界 | 用途 |
|---|---|---|
| F | [0, floor(0.8×floor(0.8N))) | 预处理、图、预测器、训练任务 |
| Cfit | [F末端, floor(0.9×floor(0.8N))) | 校正器及 T0 |
| Rref | [Cfit末端, floor(0.8N)) | 独立正常参考 |
| Vsel | 验证部分前 floor(len(V)/2) 点 | 正常预测早停、固定扰动评分早停 |
| Vcal | 验证部分其余点 | 模型全部选定后估计阈值 |

若调用 `detect_fit(..., valid_data=...)`，整个训练输入作为建模部分，内部仍为 80%/10%/10%；外部验证输入平分为 Vsel/Vcal。所有片段独立预热。

F 的逐列有限值中位数作为因果填充初值，全缺失列初值为零并记录。NaN/±Inf 前向填充；独立片段重置填充历史。均值、总体标准差仅从填充后的 F 估计，常量列标准差置为 1 并保留。以原始填充值是否完全相同判断常量，避免归约舍入产生伪小方差；常量均值精确取该常量，真实非零小方差保留原尺度。流式调用延续最后有限原始值，不使用后向填充。

默认 W=63，最大图滞后 16，H=W+16−1=78，AR 阶数 P=4，滚动长度 R=8。首预测索引为 H，首校正/参考索引为 H+P=82，首完整分数索引 D=H+P+R−1=89。参考段不额外扣除 R−1。

F 至少有 256 个有效预测目标；Cfit 至少有 64 个校正目标；Rref 至少有 128 个参考点；Vcal 至少有 128 个完整分数；F/Vsel 还须容纳辅助轨迹。长度不足时明确报出片段，保留原始采样率和时间位置。

## 模型与训练预算

`original` 容量由 F 长度解析；显式提供的配置覆盖自动值。

| 条件 | hidden_dim | predictor_block_size Q | batch_size Bp | auxiliary_train_pairs |
|---|---:|---:|---:|---:|
| len(F)<8192 | 16 | 128 | 4 | 96 |
| len(F)≥8192 | 32 | 256 | 8 | 384 |

`batch_size` 的单位是连续目标块数。例如 SWAT 默认每批 8 块、每块 256 个目标；不能沿用其他模型的 128 个窗口含义。每轮打乱目标块，并覆盖 [H,len(F)) 每个目标一次；尾块携带 mask。按 SWAT 的 495000 点训练输入计算，每轮预测器有 155 次更新，此数仅为算术预算。

预测器使用通道共享的残差深度时间卷积、当前时刻隐藏维 LayerNorm 和逐点线性混合。关系边先汇集隐藏表示，再作无偏置投影。图只在 F 上估计；超过 65536 个目标时采用所有滞后共用的确定性等距锚点，预测器仍使用全部原始目标。51 通道、d=32 的预测器有 10886 个参数；18 通道、d=16 时为 3306。

| 阶段 | 默认优化 |
|---|---|
| 预测器 | AdamW，lr=1e-3，weight_decay=1e-4；最多20轮，至少5轮，patience=4 |
| 线性校正 | FP32 Adam，lr=1e-2，weight_decay=0；100次更新，每次 beta 真正投影到 L1 半径0.95球 |
| 共享非线性补偿 | 冻结线性项；Adam，lr=1e-3，weight_decay=0；100次更新 |
| 评分器 | 一个模型，8个可学习标量；Adam，lr=1e-3，weight_decay=0；最多10轮，至少3轮，patience=3，梯度裁剪1 |

校正器按通道和分支估计 `scale=max(mean(d),IQR(d),1e-4)`，在 y=d/scale 上训练。beta 初始为零，截距为全部 y 的中位数；共享 MLP 为 P→8→1，输出层零初始化，eta=0.1×IQR(y)。两阶段批量至多 2048 个时刻，固定种子排列、尾批保留。最终 r=scale×(y−y_hat)，不对 y_hat 截断。日志保存更新数、处理样本数、首末损失和最大 beta L1 范数；固定预算不等同于精确收敛。

### 扩展训练配置

`training_profile="extended"` 使用以下明确预算。模型的因果结构、校正与证据公式、八参数评分器、固定验证协议、独立参考及阈值规则与 `original` 相同，容量与优化过程属于单独的训练配置。

| 项目 | extended 设置 |
|---|---|
| 预测器容量 | `hidden_dim_candidates=[32,64,128]`，各自完整训练；以正常 Vsel Huber 最小的检查点选宽度，完全同分优先较早轮次，再按候选顺序 |
| 预测器预算 | AdamW，初始 lr=3e-4，weight_decay=1e-4；最多60轮，至少10轮，早停 patience=12，梯度范数裁剪1 |
| 预测器学习率 | Vsel 连续4轮未改善后乘0.5（ReduceLROnPlateau patience=3），最低1e-5 |
| 连续目标批量 | 4块，每块256目标；SWaT 每轮310次更新，仍覆盖全部316722个有效目标 |
| 校正器 | 线性与非线性各20个完整 Cfit 数据遍历，尾批保留；lr 分别为1e-2、1e-3，其他定义相同；SWaT 每阶段400次更新 |
| 有限训练库 | 1536对；固定验证库仍48对、6组等权 |
| 评分器预算 | 最多100轮，至少10轮，patience=15；门控与动态参数 lr=3e-3，训练偏置 lr=3e-2，梯度范数裁剪1 |
| 评分学习率 | 固定 Vsel 任务连续6轮未改善后两组 lr 同乘0.5（patience=5）；最低分别为1e-5、1e-4 |

宽度32/64/128在51通道下分别有10886/37542/140006个预测器参数。深度、W=63和图滞后保持一致；参数跨通道共享。文献及本地检查支持这一候选范围，尚无真实 SWaT 扩展训练结果可以认定哪一档最优。容量只由 Vsel 正常预测误差选择；这个代理目标不能保证真实异常检测排序最优。每个实体重新选择，Vcal 和测试均不参与。

JSON 显式参数优先于 profile：例如 `"hidden_dim":64` 固定宽度并停用 profile 的容量搜索；显式 `"hidden_dim_candidates":[64,128]` 改变候选集合。`lr` 仅覆盖预测器学习率，`num_epochs` 仅覆盖预测器最大轮数。若希望使用扩展配置的60轮，不要保留旧命令里的 `"num_epochs":20`。显式 `corrector_linear_steps` / `corrector_nonlinear_steps` 停用相应阶段的 profile 整轮预算；也可直接设置 `corrector_linear_epochs` / `corrector_nonlinear_epochs`。所有实际参数及各候选历史写入 checkpoint。

`training_bias` 只进入训练分类 logit，单独学习率用于适配其无界分数尺度；它不是部署阈值，也不改变原始分数公式。学习率调度与早停仅读取 Vsel；恢复其最优检查点后才建立阈值。校正器另外记录阶段前后相同全 Cfit 目标上的 Huber，避免把不同随机批的首末损失误当作收敛证据；该统计不是独立验证性能。

正常 F 和 Vsel 的全量证据在缓存构建时各计算一次，并复用来记录逐通道中心化证据均值与饱和比例；不增加评分 epoch 的提取器调用。若正常数据持续达到证据上限，应结合参考覆盖分析，不能仅凭更宽网络或更低预测损失解释误报。详细文献与日志依据见 [训练分析](ASEI_training_analysis.md)。

独立参考保存全部 d/r 的 FP32 排序值。尾部查询使用左插入位置，把相等值计入上尾；两个预测分支先取最大证据，再中心化。参考自查询使用同一路径，均值 FP64 累积、中心保存 FP32。记录参考点数及 log(Nref+1)。

冻结前端默认使用 `frozen_evidence_backend="canonical"`：每次预测或校正调用的序列批量固定为1，时间计算形状固定。默认每次1035个预测计算位置、1031个校正计算位置，包含1024个完整分数所需的误差和滚动历史。尾部只补齐未使用的未来计算槽，输出立即截取真实目标；不足的历史不补造，真实训练目标、参考样本和阈值样本数量均保持原值。预测器训练使用所选profile的块批量，校正器训练批量2048；冻结计算形状不改变训练梯度公式。

所有冻结提取阶段，包括校正拟合所用误差、独立参考、两路配对任务、校准和部署，使用相同计算形状及FP32数值策略，避免改变GPU批量/时间维度后，微小舍入误差在经验分布的重复值处放大为整幅证据。每次拟合在建立参考后、评分训练前进行切片及批量证据一致性预检，最大差超过1e-5时明确停止。尾部仍严格左插入查询，不添加epsilon、合并近似相等值或降低参考精度。

评分器学习门控4个权重、门控偏置、lambda logit、kappa logit、训练偏置，共8个标量。lambda=0.8+0.18×sigmoid(logit)，初始0.9；kappa=0.01+0.19×sigmoid(logit)，初始0.02。损失固定 Ldet+Lrank，先在每个任务内计算，再对任务等权平均；验证按三个扰动族×两种布局的6组等权汇总。每轮验证，完全同分保留更早模型。

`lr/learning_rate` 与 `num_epochs/predictor_epochs` 是别名，冲突时报错；只控制预测器，不改变其他阶段预算。`win_size` 是精确感受野，支持任意正整数，随之重算 H、D 和任务长度。`predictor_amp="bf16"` 仅在支持 BF16 的 CUDA 上显式启用；冻结证据、参考、评分、校准和部署保持 FP32，并禁用 TF32。未知参数及旧网格、多卡、全量回溯参数明确报错。

## 有限任务、证据缓存与扫描

令 M=min(len(F),len(Vsel))，仅按正常长度和 Cfit 的 T0 确定：

```text
Npre = min(128, floor((M-D)/8))
Nrec = D+32
G = rolling_window
Lcap = min(16*T0, floor((M-D-Npre-Nrec-G)/2))
lengths = 去重并排序 min(Lcap, l)，l ∈ {1,max(1,T0/4),T0,4*T0,16*T0}
```

Npre<0 或 Lcap<1 时报错。单事件与等长双事件再发布局等权；双事件间隔 G，共享通道组，独立抽取启动延迟和扰动参数。事件前包含 D 点原始预热和 Npre 次正常状态更新，之后有 Nrec 点恢复。3604 点输入的默认划分产生 Vsel=360、Vcal=361；T0=8 时 Npre=33、Lcap=54，最长双事件轨迹359点。

original训练库默认96/384对，extended为1536对，验证库固定48对。三种扰动族与两种布局均衡分配，子类型均衡、事件长度循环分配。验证专用种子固定任务；训练一半均匀取位置，一半从16个候选位置正常强度最高四分位抽取。困难度查 F 正常证据前缀和。错位扰动从原始正常副本读取过去值，事件首末点必须改变；确定性重试最多128次，局部常值时重新取有效基础位置，无法构造则报错。

F、Vsel 正常证据各顺序计算一次。所有扰动在观测空间施加，然后经过完整冻结模型。缓存形状为 [2,Lraw−H−P,C,2]，包含首完整分数之前的 R−1 个证据，视角顺序为校正/状态。按批使用 avg_pool1d 生成滚动均值，避免长序列 FP32 累加和相减。评分训练只读取缓存，不调用预测器、校正器或 searchsorted。

两库共用256 MiB GPU缓存预算和1024 MiB主机预算；超限写入工作目录下的临时 .npy 映射文件。文件原子写入，检查形状、dtype和校验和。键包含数据和列序、边界、预处理、模型、图、参考、任务清单、配置、精度及版本。每次拟合构建独立临时库，结束释放，不依赖跨运行缓存。记录：

- `normal_evidence_passes=2`：F、Vsel各一次。
- `changed_trajectory_builds=M_A+M_V`：成功完成的扰动轨迹数。
- `score_epoch_feature_extractor_calls=0`：评分训练期间不重复提取证据。

状态更新 m=ReLU(lambda×m_prev+u−kappa)，分数取激活量 a 的通道最大值，保留负分数。正式后端为 max-affine 倍增前缀扫描，自定义一阶反向使用仿射扫描；默认每256个分数一块，块间传递实际状态和初态梯度，不 detach 正常前缀。`reference` 顺序后端只用于核验。两后端按容差一致，不承诺浮点逐位相同；校准与部署固定相同后端、块长和精度。

## 阈值、流式接口与持久化

模型参数全部选定后，仅在 Vcal 上从零状态运行完整流程。默认 alpha=0.01；K=floor(alpha×N)，取升序分数零基索引 N−K−1 为阈值，严格 S>h 报警，允许负阈值。保存 N、K、h、实际超阈点数、实际比例及1/N。并列不加噪声。TAB 的 `anomaly_ratio` 使用百分数，显式给出时覆盖 alpha；多个比例各自校准，不根据测试表现选择。

`detect_score` 返回 (scores,scores)，`detect_label` 返回 ({百分数:labels},scores)。两者每次冷启动；`detect_stream` 保留缺失填充状态、最多D个标准化观测和非负递推状态，`reset_state` 开始新序列。输出与输入等长，`last_ready_mask_` 标记完整分数；首D点输出−6.2、报警0。任意计算分块不重复更新历史。

默认输出原始带符号分数。`score_transform="bounded"` 显式启用固定理论界的线性变换；默认界为 [−6.2,299.5]，阈值同时变换，不使用测试 min/max。

`save/load` 使用 format_version=3，保存完整解析配置、预处理、列序、三个模型、参考、任务描述、种子、阈值和运行元数据。加载不重新估计统计量，模型冻结、流状态清空。没有冻结后端配置的历史checkpoint按 `legacy` 路径加载以复现原数值；不能把其旧参考直接用于新的canonical查询。应用新数值策略应重新训练并估计参考和阈值。旧格式模型需要重新拟合；临时证据库不属于 checkpoint 的运行依赖。自动保存文件名摘要包含实现版本，比较不同实现时同时使用独立的结果目录和日志名。

## 数据覆盖与冻结模型诊断

日志和checkpoint记录预测器每轮F/Vsel有效目标数、校正器实际访问次数和不同样本数、参考点数、评分器每轮任务覆盖及判别/排序损失分量。每轮预测和评分任务覆盖、最终校准分数数量有显式断言。增加训练预算时须记录实际配置，默认值不根据测试指标自动调整。

`scripts/audit_asei.py` 支持只解析已有日志及指标，也支持读取checkpoint和同一TAB数据版本，检查预处理、证据、通道主导情况和校准重现。它核验数据指纹，不训练或修改保存的阈值。真实SWAT运行的完整核对与命令见 [ASEI_SWAT_audit.md](ASEI_SWAT_audit.md)，解析结果见 [ASEI_SWAT_audit.json](ASEI_SWAT_audit.json)。

## 小规模验收

```powershell
& 'E:\Miniconda3\Scripts\conda.exe' run -n codex python -B -m pytest tests -q
& 'E:\Miniconda3\Scripts\conda.exe' run -n codex python -B scripts/validate_asei.py --device cpu --output docs/ASEI_validation.json
& 'E:\Miniconda3\Scripts\conda.exe' run -n codex python -B scripts/validate_asei.py --training-profile extended --device cpu --output docs/ASEI_extended_validation.json
```

服务器单卡正确性及小规模计时：

```bash
uv run python -B scripts/check_asei_runtime.py --gpus 1 --device auto --channels 51 --batch-size 8 --block-sizes 128 256
uv run python -B scripts/check_asei_runtime.py --gpus 1 --device auto --channels 51 --frozen-only
uv run python -B scripts/check_asei_runtime.py --gpus 1 --device auto --channels 51 --hidden-dim 64 --frozen-only
uv run python -B scripts/check_asei_runtime.py --gpus 1 --device auto --channels 51 --hidden-dim 128 --frozen-only
CUDA_VISIBLE_DEVICES=1 uv run python -B -m pytest tests/test_asei_core.py tests/test_asei.py tests/test_asei_audit.py tests/test_asei_frozen_numerics.py tests/test_asei_training.py -q
```

测试包含顺序/扫描前向和全部一阶梯度、ReLU零点和通道并列、跨块梯度、长10000点FP32误差、预测因果性、尾块覆盖、参考并列、L1投影、任务重建、缓存损坏、Vcal隔离、流式缺失填充、保存重载以及TAB全部指标和CSV报告。小规模测量见 [ASEI_validation.json](ASEI_validation.json)，不代表完整数据集精度或服务器加速结果。
