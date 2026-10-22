# -*- coding: utf-8 -*-

import logging
import os
from typing import Union, List

import pandas as pd

from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.recording import load_record_data
from ts_benchmark.report.utils.leaderboard import get_leaderboard

logger = logging.getLogger(__name__)

# currently we do not support showing or processing artifact columns
# these columns are dropped as soon as data is loaded in order to save memory
ARTIFACT_COLUMNS = [
    FieldNames.ACTUAL_DATA,
    FieldNames.INFERENCE_DATA,
    FieldNames.LOG_INFO,
]


# TODO: update the docstring to match the common format in OTB.
def report(report_config: dict) -> None:
    """
    Generate a report based on specified configuration parameters.

    Parameters:
    - report_config (dict): A dictionary containing the following keys and their respective values:
        - log_files_list (List[str]): A list of file paths for log files.
        - leaderboard_file_name (str): The name for the saved report file.
        - aggregate_type (str): The aggregation type used when reporting the final results of evaluation metrics.
        - report_metrics (Union[str, List[str]]): The metrics for the report, can be a string or a list of strings.
        - fill_type (str): The type of fill for missing values.
        - null_value_threshold (float): The threshold value for null metrics.
        - group_by (List[str]): Optional record columns to report separately.

    Raises:
    - ValueError: If all metrics have too many null values, making performance comparison impossible.

    Returns:
    - None: The function does not return a value, but generates and saves a report to a CSV file.
    """
    log_files: Union[List[str], pd.DataFrame] = report_config.get("log_files_list")
    if log_files is None or len(log_files) == 0:
        raise ValueError("No log files to report")

    log_data = (
        log_files
        if isinstance(log_files, pd.DataFrame)
        else load_record_data(log_files, drop_columns=ARTIFACT_COLUMNS)
    )

    group_by = report_config.get("group_by", [])
    if isinstance(group_by, str):
        group_by = [group_by]
    missing = set(group_by) - set(log_data.columns)
    if missing:
        raise ValueError("Report grouping columns are missing: {}".format(sorted(missing)))
    groups = (log_data.groupby(group_by[0] if len(group_by) == 1 else group_by, sort=False, dropna=False)
              if group_by else [((), log_data)])
    leaderboards = []
    for key, group in groups:
        leaderboard = get_leaderboard(
            group,
            report_config["report_metrics"],
            report_config.get("aggregate_type", "mean"),
            report_config.get("fill_type", "mean_value"),
            report_config.get("null_value_threshold", 0.3),
        )
        keys = key if isinstance(key, tuple) else (key,)
        for column, value in reversed(list(zip(group_by, keys))):
            leaderboard.insert(0, column, value)
        leaderboard.insert(0, "strategy_args", group.iloc[0]["strategy_args"])
        leaderboards.append(leaderboard)
    leaderboard_df = pd.concat(leaderboards, ignore_index=True)

    # Create final DataFrame and save to CSV
    if report_config.get("save_path", None) is not None:
        save_path = report_config.get("save_path", None)
        report_path = os.path.join(ROOT_PATH, "result", save_path, report_config["leaderboard_file_name"])
    else:
        report_path = os.path.join(ROOT_PATH, "result", report_config["leaderboard_file_name"])
    leaderboard_df.to_csv(report_path, index=False)
    logger.info("Metric report saved to %s", report_path)
