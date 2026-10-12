| 配置 | 数据 | 切分方式 | 模型输出 | 默认报告 |
|---|---|---|---|---|
| `all_detect_label` | 单变量 | 全序列拟合并检测 | 标签 | F1、ADD |
| `all_detect_score` | 单变量 | 全序列拟合并检测 | 分数 | AUC-ROC |
| `fixed_detect_label` | 单变量 | 固定前 10% 训练 | 标签 | F1、ADD |
| `fixed_detect_score` | 单变量 | 固定前 10% 训练 | 分数 | AUC-ROC |
| `unfixed_detect_label` | 单变量 | 按 `train_lens` | 标签 | Affiliation-F1、ADD |
| `unfixed_detect_label_multi` | 多变量 | 按 `train_lens` | 标签 | Affiliation-F1、ADD |
| `unfixed_detect_score` | 单变量 | 按 `train_lens` | 分数 | AUC-ROC |
| `unfixed_detect_score_multi` | 多变量 | 按 `train_lens` | 分数 | AUC-ROC |
