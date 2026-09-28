# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import sys
import warnings
from typing import Dict, NoReturn

import torch

# 将TAB/加入python模块搜索路径
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from ts_benchmark.utils.get_file_name import get_unique_file_suffix
from ts_benchmark.report import report
from ts_benchmark.common.constant import CONFIG_PATH, THIRD_PARTY_PATH
from ts_benchmark.pipeline import pipeline
from ts_benchmark.utils.parallel import ParallelBackend

# 将ts_benchmark/baselines/third_party加入模块路径，供第三方模型实现使用
sys.path.insert(0, THIRD_PARTY_PATH)

warnings.filterwarnings("ignore")


# 使用--data-name-list指定具体序列文件，使用--data-set-name指定数据集类别
def build_data_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the data loader config from commandline arguments and configuration dict
    """
    data_config = config_data["data_config"]
    data_config["data_name_list"] = args.data_name_list
    if args.data_set_name is not None:
        data_config["data_set_name"] = args.data_set_name
    return data_config


# 多组模型、接口适配器、超参数
def build_model_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the model config from commandline arguments and configuration dict
    """
    model_config = config_data.get("model_config", None)

    if args.adapter is not None:
        args.adapter = [None if item == "None" else item for item in args.adapter]
        if len(args.model_name) > len(args.adapter):
            args.adapter.extend([None] * (len(args.model_name) - len(args.adapter)))
    else:
        args.adapter = [None] * len(args.model_name)

    if args.model_hyper_params is not None:
        args.model_hyper_params = [
            None if item == "None" else item for item in args.model_hyper_params
        ]
        if len(args.model_name) > len(args.model_hyper_params):
            args.model_hyper_params.extend(
                [None] * (len(args.model_name) - len(args.model_hyper_params))
            )
    else:
        args.model_hyper_params = [None] * len(args.model_name)

    for adapter, model_name, model_hyper_params in zip(
        args.adapter, args.model_name, args.model_hyper_params
    ):
        model_config["models"].append(
            {
                "adapter": adapter,
                "model_name": model_name,
                "model_hyper_params": (
                    json.loads(model_hyper_params)
                    if model_hyper_params is not None
                    else {}
                ),
            }
        )

    return model_config


# 指标列表、评测策略、随机种子
def build_evaluation_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the evaluation config from commandline arguments and configuration dict
    """
    evaluation_config = config_data["evaluation_config"]

    metric_list = []
    # 希望使用配置中的全部指标时，直接省略--metrics
    if args.metrics != "all" and args.metrics is not None:
        for metric in args.metrics:
            metric = json.loads(metric)
            metric_list.append(metric)
        evaluation_config["metrics"] = metric_list

    default_strategy_args = evaluation_config["strategy_args"]
    strategy_args_updates = (
        json.loads(args.strategy_args) if args.strategy_args else None
    )

    if strategy_args_updates is not None:
        default_strategy_args.update(strategy_args_updates)

    if args.seed is not None:
        default_strategy_args["seed"] = args.seed

    return evaluation_config


# 结果聚合方式及保存位置
def build_report_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the report config from commandline arguments and configuration dict
    """
    report_config = config_data["report_config"]
    report_config["aggregate_type"] = args.aggregate_type
    report_config["save_path"] = args.save_path

    return report_config


def init_worker(env: Dict) -> NoReturn:
    """
    An initializer function for each worker that does some global setup
    """
    sys.path.insert(0, THIRD_PARTY_PATH)
    # 当前进程中PyTorch的CPU算子内部并行线程数
    # torch.set_num_threads(1)
    torch.set_num_threads(4)


if __name__ == "__main__":
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser(
        description="run_benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 指定/TAB/config下的配置文件名
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Evaluation config file path",
    )

    # 指定一个或多个序列文件
    parser.add_argument(
        "--data-name-list",
        type=str,
        nargs="+",
        default=None,
        help="List of series names entered by the user",
    )

    # 指定数据集类别
    parser.add_argument(
        "--data-set-name",
        type=str,
        nargs="+",
        default=None,
        help="List of dataset name names entered by the user,"
        "only takes effect when data_name_list is not specified",
    )

    # 指定模型接口适配器
    parser.add_argument(
        "--adapter",
        type=str,
        nargs="+",
        default=None,
        help="Adapter used to adapt the method to our pipeline",
    )

    # 指定模型名称
    parser.add_argument(
        "--model-name",
        type=str,
        nargs="+",
        required=True,
        help="The relative path of the model that needs to be evaluated",
    )

    # 指定模型超参数
    parser.add_argument(
        "--model-hyper-params",
        type=str,
        nargs="+",
        default=None,
        help=(
            "The input parameters corresponding to the models to be evaluated "
            "should correspond one-to-one with the --model-name options."
        ),
    )

    # 指定（覆盖）评测指标
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Evaluation metrics that need to be calculated",
    )

    # 指定（覆盖）策略配置
    parser.add_argument(
        "--strategy-args",
        type=str,
        default=None,
        help="Parameters required for evaluating strategies",
    )

    # 指定（覆盖）种子
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed that is set before evaluating any model-series pair, "
        "by default, use the seed value in the config file",
    )

    # 指定任务执行后端
    parser.add_argument(
        "--eval-backend",
        type=str,
        default="sequential",
        choices=["sequential", "ray"],
        help="Evaluation backend, use ray for parallel evaluation",
    )

    # 指定Ray后端可调度的逻辑CPU数量
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=os.cpu_count(),
        help="Number of cpus to use, only available in both backends",
    )

    # 指定可用的GPU编号列表
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="List of gpu devices to use, only available in ray backends",
    )

    # 指定Ray后端创建worker的数量
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count(),
        help="Number of evaluation workers",
    )

    # TODO: should timeout be part of the configuration file?
    # 每个“模型×序列”任务允许执行的最长时间（单位：秒）
    parser.add_argument(
        "--timeout",
        type=float,
        # default=600,
        default=3600,
        help="Time limit for each evaluation task, in seconds",
    )

    # 每个Ray worker最多连续执行的任务数量
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=100,
        help="Max tasks to run on a single worker when using parallel backends",
    )

    # 指定结果聚合方式
    parser.add_argument(
        "--aggregate_type",
        default="mean",
        help="Select the baseline algorithm to compare",
    )

    # 指定结果展示方式
    parser.add_argument(
        "--report-method",
        type=str,
        default="csv",
        choices=[
            "dash",
            "csv",
        ],
        help="Presentation form of algorithm performance comparison results",
    )

    # 指定结果保存路径
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="The relative path for saving evaluation results, relative to the result folder",
    )

    # 解析参数与日志初始化
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s(%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    torch.set_num_threads(3)
    with open(os.path.join(CONFIG_PATH, args.config_path), "r") as file:
        config_data = json.load(file)

    required_configs = [
        "data_config",
        "model_config",
        "evaluation_config",
        "report_config",
    ]
    for config_name in required_configs:
        if config_data.get(config_name) is None:
            raise ValueError(f"{config_name} is none")

    data_config = build_data_config(args, config_data)
    model_config = build_model_config(args, config_data)
    evaluation_config = build_evaluation_config(args, config_data)
    report_config = build_report_config(args, config_data)

    # 初始化执行后端
    ParallelBackend().init(
        backend=args.eval_backend,
        n_workers=args.num_workers,
        n_cpus=args.num_cpus,
        gpu_devices=args.gpus,
        default_timeout=args.timeout,
        max_tasks_per_child=args.max_tasks_per_child,
        worker_initializers=[init_worker],
    )

    try:
        log_filenames = pipeline(
            data_config,
            model_config,
            evaluation_config,
            save_path=args.save_path,
        )

    finally:
        ParallelBackend().close(force=True)

    report_config["log_files_list"] = log_filenames
    if args.report_method == "csv":
        filename = get_unique_file_suffix()
        leaderboard_file_name = "test_report" + filename
        report_config["leaderboard_file_name"] = leaderboard_file_name
    report(report_config, report_method=args.report_method)
