# AnomalyTransformer 多 GPU 运行

AnomalyTransformer 支持通过 `torch.nn.DataParallel` 在单进程中进行数据并行。
`--gpus` 指定可见 GPU，模型超参数 `multi_gpu` 决定是否启用数据并行，默认值为
`false`。此开关目前只适用于该模型，不代表其他基线也支持多卡训练。

在 TAB 仓库根目录运行以下 Bash 命令：

```bash
mkdir -p ./log
nohup uv run python -u ./scripts/run_benchmark.py \
  --config-path unfixed_detect_label_multi_config.json \
  --model-name self_impl.AnomalyTransformer \
  --data-name-list SWAT.csv \
  --model-hyper-params '{"batch_size":32,"lr":0.0001,"num_epochs":10,"win_size":100,"multi_gpu":true}' \
  --seed 42 \
  --eval-backend sequential \
  --gpus 0 1 \
  --num-workers 1 \
  --save-path label/AnomalyTransformer_SWaT_seed42 \
  > ./log/at_swat_label_seed42.log 2>&1 &
```

`batch_size=32` 是两卡合计的 batch size，通常每卡 16 个窗口。最后一个 batch
可以较小。若希望每卡 32 个窗口，可设置为 64，但这会改变每轮的优化步数及梯度
统计，不应视为与 batch size 32 等价的实验。

模型使用可见设备的本地编号，例如 `--gpus 2 3` 对应模型内的 `cuda:0` 和
`cuda:1`。启用 `multi_gpu` 时，至少需要两个可见 CUDA 设备，否则任务会记录明确
错误。日志应出现 `data_parallel=True, device_ids=[0, 1], global_batch_size=32`。

`--num-workers` 控制 Ray 中独立的“模型 × 序列”评测任务。单个模型的数据并行
由 `DataParallel` 完成，不需要使用 `torchrun`。若选择 Ray 并创建两个 worker，
两张 GPU 通常会各分配一张给一个 worker，不能据此让同一个模型使用两张卡。

DataParallel 会按 batch 维拆分输入并汇总梯度，其参数与缓冲区必须位于第一张
设备；实现中的时间距离矩阵是随设备迁移的 buffer。检查点以不带 `module.`
前缀的 CPU 张量保存，训练结束后恢复最佳验证权重。参见
[PyTorch DataParallel 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)。

两卡不会保证两倍加速。DataParallel 的注意力输出会汇总到第一张卡，复制和通信
也有开销；PyTorch 对更大规模的训练推荐使用 DDP。不同 GPU 数量的浮点归约顺序
可能不同，固定 seed 不保证单卡和双卡结果逐位一致。

## 服务器上的小规模验证

以下测试不读取 SWAT，不执行完整基准。它检查两卡前向结果与双目标反向梯度，
并实际运行一轮微型模型训练、检查点恢复及尾部预测。

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python -B -m unittest discover \
  -s tests -p test_anomaly_transformer.py -v
```

两项 `TwoGpuTests` 应显示 `ok`。若显示 `skipped`，说明运行测试的 PyTorch
未识别到两个 CUDA 设备。本地检查环境只有 CPU，未实际执行这两项硬件测试。

数据并行使用 PyTorch 自带能力，无新增第三方依赖。仅运行该模型和 CSV 报告时，
可使用 `requirements-anomaly-transformer.txt` 中的最小依赖；服务器应安装匹配
驱动的 CUDA 版 PyTorch。
