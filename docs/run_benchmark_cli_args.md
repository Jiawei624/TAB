# `run_benchmark.py` 命令行参数

- `--config-path`: 评测配置文件路径。
- `--data-name-list`: 用户输入的时间序列文件名列表。
- `--data-set-name`: 数据集名称列表；仅当未指定 `data_name_list` 时生效。
- `--adapter`: 用于将方法适配到 TAB pipeline 的适配器。
- `--model-name`: 需要评测的模型相对路径。
- `--model-hyper-params`: 模型对应的输入超参数；应与 `--model-name` 中的模型一一对应。
- `--metrics`: 需要计算的评测指标。支持 `all`、指标名列表（如 `f_score add`）或 JSON 对象列表。
- `--strategy-args`: 评测策略所需的参数。
- `--seed`: 在评测每个“模型—序列”组合前设置的随机种子；默认使用配置文件中的种子值。
- `--eval-backend`: 评测后端；可选 `sequential` 或 `ray`，使用 `ray` 进行并行评测。
- `--num-cpus`: Ray 可调度的 CPU 数量；顺序后端不按此参数分配 CPU 核。
- `--gpus`: 向执行后端暴露的 GPU 编号列表，两种后端均生效。它不自动启用模型数据并行。
- `--num-workers`: Ray 的独立评测 worker 数量。顺序后端只有一个执行进程；此参数也不是 DataLoader worker 数量。
- `--timeout`: Ray 每个评测任务的时间上限，单位为秒；顺序后端不强制超时。
- `--max-tasks-per-child`: 使用并行后端时，一个 worker 最多运行的任务数。
- `--aggregate-type`（兼容 `--aggregate_type`）: 跨序列聚合方式；未指定时保留配置文件的设置。
- `--report-method`: 算法性能比较结果的展示形式；可选 `dash` 或 `csv`。
- `--save-path`: 评测结果保存路径；相对于 `result` 文件夹。

AnomalyTransformer 双卡训练需要同时设置 `--gpus 0 1` 和模型超参数
`"multi_gpu": true`，完整命令见 [多 GPU 运行](multi_gpu.md)。
