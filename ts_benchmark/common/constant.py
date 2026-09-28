# -*- coding: utf-8 -*-
import os

# TAB\TAB路径 第一次回退会先去掉文件名
ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", "..", ".."))

# 指定具体任务（预测/检测）对应的数据文件夹路径
FORECASTING_DATASET_PATH = os.path.join(ROOT_PATH, "dataset", "forecasting")
ST_FORECASTING_DATASET_PATH = os.path.join(ROOT_PATH, "dataset", "st_forecasting")
ANOMALY_DETECT_DATASET_PATH = os.path.join(ROOT_PATH, "dataset", "anomaly_detect")

# 配置文件路径
CONFIG_PATH = os.path.join(ROOT_PATH, "config")

# 第三方实现模型路径
THIRD_PARTY_PATH = os.path.join(ROOT_PATH, "ts_benchmark", "baselines", "third_party")
