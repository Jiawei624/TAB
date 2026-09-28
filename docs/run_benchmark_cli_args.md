# `run_benchmark.py` 命令行参数

- `--config-path`: 评测配置文件路径。
- `--data-name-list`: 用户输入的时间序列文件名列表。
- `--data-set-name`: 数据集名称列表；仅当未指定 `data_name_list` 时生效。
- `--adapter`: 用于将方法适配到 TAB pipeline 的适配器。
- `--model-name`: 需要评测的模型相对路径。
- `--model-hyper-params`: 模型对应的输入超参数；应与 `--model-name` 中的模型一一对应。
- `--metrics`: 需要计算的评测指标。
- `--strategy-args`: 评测策略所需的参数。
- `--seed`: 在评测每个“模型—序列”组合前设置的随机种子；默认使用配置文件中的种子值。
- `--eval-backend`: 评测后端；可选 `sequential` 或 `ray`，使用 `ray` 进行并行评测。
- `--num-cpus`: 使用的 CPU 数量；适用于两种评测后端。
- `--gpus`: 使用的 GPU 设备编号列表；源码注释说明其仅适用于 Ray 后端。
- `--num-workers`: 评测 worker 数量。
- `--timeout`: 每个评测任务的时间上限，单位为秒。
- `--max-tasks-per-child`: 使用并行后端时，一个 worker 最多运行的任务数。
- `--aggregate_type`: 选择用于比较基线算法的聚合方式。
- `--report-method`: 算法性能比较结果的展示形式；可选 `dash` 或 `csv`。
- `--save-path`: 评测结果保存路径；相对于 `result` 文件夹。
