# ADD（平均序列检测延迟）

指标名为 `add`，单位为采样点，越小越好，无额外依赖。

## 计算口径

依据 MODEM 论文 *Multi-resolution decomposable diffusion model for non-stationary
time series anomaly detection* 附录 B.3、公式 (30)（第 20 页），以及本工作区
`MODEM/modem/metrics.py` 中的 `evaluate` 实现：

对真实标签中每个连续异常区间 `[begin, end)`，若区间内首次预测异常的位置为
`first_hit`，则延迟为 `first_hit - begin`；整个区间漏检时，延迟取 `end - begin`。
ADD 是所有真实异常区间延迟的算术平均，每个区间权重相同。

- 区间起点立即检出时延迟为 0；异常区间之外的误报不改变该区间延迟。
- 漏检按区间长度计入，沿用本地 MODEM 实现；论文公式未明确漏检处理方式。
- 必须使用点调整前的预测标签，点调整会抹去已检出区间的真实延迟。
- 没有真实异常区间（或输入为空）时返回 `NaN`，对应 MODEM 的 `None`，表示无定义。
- 对声明 `evaluation_valid_mask_` 的模型，评价按原始标签时间轴识别事件；与无效位置相交或触及测试边界的事件不计入 ADD，并单独记录。未声明掩码时使用上述完整输入数组的原有区间规则。参见 [DynamicNeighborhoodStat](dynamic_neighborhood_stat.md)。
- 接受等长的 `(T,)` 或 `(T, 1)` 二值标签；非二值、非有限值、长度不匹配等输入会报错。
  TAB 的 `evaluate_with_log` 将指标错误记录到 `log_info` 并填入 `NaN`。
- 按测试序列的采样点计时，不表示模型推理耗时；若要换算为实际时间，需要乘以采样间隔。

例如，真实标签 `[0, 1, 1, 1, 0, 1, 1]`、预测标签 `[0, 0, 1, 0, 0, 0, 0]`，
两个异常区间的延迟分别为 1 和 2（漏检），因此 ADD = 1.5。

## 在 TAB 中使用

`all_detect_label`、`fixed_detect_label`、`unfixed_detect_label` 均支持 `add`。
现有四份 `*_detect_label*_config.json` 的 `metrics: "all"` 会自动计算该指标，
默认报告也已包含 `add`。继续运行 `scripts/univariate/label/` 或
`scripts/multivariate/label/` 下的脚本即可；单、多变量使用同一套区间计算规则。

自定义配置可以仅启用 ADD，或与其他标签指标并用：

```json
{
  "evaluation_config": {
    "metrics": ["f_score", "add"],
    "strategy_args": {"strategy_name": "unfixed_detect_label", "seed": 42}
  },
  "report_config": {
    "aggregate_type": "mean",
    "report_metrics": ["f_score", "add"],
    "fill_type": "mean_value",
    "null_value_threshold": 0.3
  }
}
```

以上是配置片段，保留原配置中的 `data_config` 和 `model_config`。
也可在原运行命令后追加 `--metrics '{"name":"add"}'`（Bash），仅计算 ADD。
默认标签配置的报告已包含 `add`，未计算的其他报告指标将被忽略并记录警告。

ADD 依赖二值决策，不注册到 `*_detect_score` 策略。输出异常分数的模型应通过
现有 `detect_label` 接口按其阈值策略生成标签，再使用对应的标签配置。
ADD 不会自行选择阈值，也不会根据测试标签搜索最优阈值。

## 结果与汇总

逐序列结果保留 `add` 列。若模型返回多个 `anomaly_ratio`，每个比例分别计算，
比例记录在 `typical_anomaly_ratio` 列中。CSV 和 Dash 汇总报告均可通过
`report_metrics` 选择 `add`。

报告沿用 TAB 的聚合方式：先对同一模型、同一序列的多条记录取均值，再按
`aggregate_type` 跨序列汇总；这不同于把所有序列的异常区间合并后计算 ADD。
比较指定阈值下的延迟时，应固定 `anomaly_ratio`，或先筛选对应的结果行。
`NaN` 沿用 `fill_type` 和 `null_value_threshold` 的现有处理规则；不要将其当作 0。

ADD 不惩罚异常区间外的误报，全预测异常会得到 0，应结合 F1 或精确率评估。
不同采样间隔、异常区间长度的序列，其原始 ADD 也不宜直接解释为相同的实际延迟。
预测长度不足时，TAB 仍沿用末尾补 0 的对齐逻辑，补齐部分按未检出处理。

## 验证

本地验证脚本为 `tests/test_add_metric.py`。当前工作区忽略了 `tests/`，脚本不包含在
本次 Git 提交中；若要在服务器复跑验证，需要一并复制该脚本。

在 TAB 仓库根目录和已安装项目依赖的环境中运行：

```shell
python -m unittest discover -s tests -p test_add_metric.py -v
```

验证包括延迟边界、漏检、点调整顺序，以及配置到检测策略、记录读写和 CSV 报告的流程。
