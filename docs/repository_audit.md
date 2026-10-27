# TAB 检查与验证记录

检查日期：2026-10-27。范围为命令行入口、数据读取、共享执行后端、六种异常检测
策略、指标注册与计算、结果保存和报告，以及 AnomalyTransformer 的完整训练推理
路径。第三方模型源码做语法解析；未逐个训练全部模型，也未执行完整 SWAT 实验。

## 主流程约束

| 环节 | 当前行为 | 验证方式 |
| --- | --- | --- |
| 数据读取 | 按 `cols` 还原变量；保留实际 `label` 列；支持交错长表；检查列长与时间戳对齐 | 合成 CSV、元数据测试 |
| 数据划分 | fixed 要求有效比例；unfixed 使用有效 `train_lens`；all 使用整条无标签特征序列拟合 | 六种策略集成测试 |
| 标签隔离 | 特征中不含 `label`；训练接口只能得到训练标签，无法通过策略读取测试标签 | 合成检测器检查 |
| AnomalyTransformer | 训练段内部按 80% / 20% 划分；标准化只拟合前 80%；阈值只使用后 20% | 实际 CPU 训练、校准独立性测试 |
| 模型配置 | `e_layers`、`d_model`、`n_heads`、`d_ff` 生效；非法参数与窗口不足有明确错误 | 微型模型训练、边界测试 |
| ADD | 使用点调整前的二值预测；每个真实异常段等权；漏检延迟取段长；无真实异常返回 NaN | 与本地 MODEM 实现核对、ADD 回归测试 |
| 标签指标 | 单类别标签可计算正类指标；完全漏检的 affiliation F 为 0；无真实事件时 affiliation 指标无定义 | 单类别、全漏检测试 |
| 区间指标 | 正确处理空区间及序列边界；默认平坦权重按区间内命中数计算 | 六种策略及边界测试 |
| Precision@K | K 为真实正例数；恰好取 K 个预测；边界并列分数按期望命中数计算；K=0 返回 NaN | 完美预测与并列分数测试 |
| 预测对齐 | AnomalyTransformer 为所有真实采样点输出预测；其他模型短预测补 0 并记录数量；超长或非向量输出报错 | 尾部窗口与长度测试 |
| 异常回退 | 构造、拟合和任务结果异常均保留完整结果行、序列名与 traceback；参数化指标失败不丢失其他指标 | 构造失败、超时、指标异常测试 |
| 报告 | 不同比例分别输出；失败序列计入缺失比例；Inf 按缺失处理；支持 DataFrame 输入 | 压缩记录与 CSV 集成测试 |
| 可选依赖 | 按需导入模型、Ray 和 Dash；选择 AnomalyTransformer 无需安装无关模型依赖 | 未安装 reformer-pytorch / numba 的环境中实际训练 |
| 并行后端 | Sequential 保留任务异常并执行初始化器；Ray 将用户任务异常交给结果收集器，数据同步异常会向上传递 | Sequential 实测、Ray 异常路径模拟 |
| LOF | 每个时间点保留完整特征向量；输出长度等于时间点数；标签使用 LOF 决策阈值 | 多变量小样本拟合 |
| LeftSTAMPi | 流式矩阵轮廓保留历史长度，避免仅返回最后一个固定缓冲区 | 源码与 STUMPY 接口核对；未在本机运行其完整算法 |

报告对于同一模型、参数、比例和序列的重复记录先取均值，再跨序列聚合。
ADD 的该聚合方式是“序列均值”，不等于汇集所有异常区间后的总体 ADD。

## 验证结果

在指定的 codex 环境中运行：

```powershell
& 'E:\Miniconda3\Scripts\conda.exe' run -n codex python -B -m unittest discover -s tests -v
```

44 项测试中 42 项通过，2 项双 GPU 测试因无 CUDA 跳过；1314 个 Python
文件语法解析通过。文件数包括仓库内第三方源码及验证脚本。

环境为 Python 3.10.20、PyTorch 2.12.0+cpu、NumPy 2.2.6、pandas 2.3.3、
scikit-learn 1.7.2、statsmodels 0.14.6。测试包含实际微型 AnomalyTransformer
训练，以及真实文件读取到压缩记录和 CSV 报告的流程；六种检测策略均执行实际
指标计算。Ray 的进程调度与多 GPU 通信未做真实集群验证。

`tests/` 随仓库分发。服务器可用 `uv run python -B -m unittest discover -s tests -v`
复跑共享流程；双 GPU 测试应按 [多 GPU 说明](multi_gpu.md) 单独运行。

## 实验口径与验证边界

- `all_detect_*` 是整条序列拟合后再评测的传导式协议，不能解释为独立测试集评估。
- `unfixed_detect_*` 只控制外层划分，不能强制每个模型使用相同阈值校准数据。
  DCdetector、CATCH、ModernTCN、TranAD、KMeans 及若干适配器仍会结合测试分数
  选择百分位阈值；LOF 也在待检测样本上拟合局部密度。这些做法不使用测试标签，
  但会依赖测试分布。与只在验证段校准的 AnomalyTransformer 比较时，需要明确
  实验协议。已有模型的这类算法口径没有统一改写。
- ADD 不惩罚区间外误报，应与 F1 / precision 共同报告。真实标签没有异常段时，
  ADD 的 NaN 表示无定义；报告的缺失值策略不能将其解释成零延迟。
- 其他模型短预测的补 0 是兼容策略，不能证明其输出与真实时间位置对齐。使用
  滑窗起点或终点表示分数的模型，仍需按模型定义核对时间对齐。
- 全量 `requirements.txt` 包含多套基线生态，未在干净环境中整体安装验证。
  NumPy `<2` 的约束仍用于这些基线；上述 CPU 检查不代表它们全部支持 NumPy 2。
  `requirements-anomaly-transformer.txt` 提供该模型的最小依赖。Numba 的版本范围
  应兼容 NumPy >=1.23.5，参见 [Numba 官方兼容表](https://numba.readthedocs.io/en/stable/user/installing.html#version-support-information)。
- 两张 GPU 同时训练的入口与模型实现见 [多 GPU 说明](multi_gpu.md)。真实双卡
  通信、显存占用及完整数据集运行时间需在服务器验证。

LeftSTAMPi 的历史保留行为依据
[STUMPY stumpi 参数说明](https://stumpy.readthedocs.io/en/latest/api.html#stumpy.stumpi)。
