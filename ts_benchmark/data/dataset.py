from typing import Optional, Dict, NoReturn

import pandas as pd


# 保存时间序列数据、协变量和元数据
class Dataset:
    """
    A class that stores data information and meta information

    Any integrity checks and special update rules should be encapsulated under this interface.
    """

    def __init__(self):
        self._metadata = None
        self._data_dict = {}
        self._covariate_dict = {}

    @property
    def metadata(self) -> Optional[pd.DataFrame]:
        """
        Returns the full metadata in DataFrame format

        If the metadata is not available, returns None.

        DO NOT perform inplace operations on the return value.
        """
        return self._metadata

    # 整体设置数据
    def set_data(
        self,
        data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        covariate_dict: Optional[Dict[str, Dict]] = None,
        metadata: Optional[pd.DataFrame] = None,
    ) -> NoReturn:
        """
        Sets the series data and the meta information

        :param data_dict: A dictionary of DataFrames where the keys are the names of the
            series, and the values are DataFrames following the OTB protocol. If None is
            given, the data dictionary is not set.
        :param covariate_dict: A dictionary of covariates where the keys are the names of the
            series, and the values are dictionary of the series' covariates.
        :param metadata: A DataFrame of meta information where the index contains series names
            and the columns contains meta-info fields. If None is given, the metadata is not
            set.
        """
        new_metadata = metadata if metadata is not None else self._metadata
        new_data_dict = data_dict if data_dict is not None else self._data_dict
        new_covariate_dict = (
            covariate_dict if covariate_dict is not None else self._covariate_dict
        )

        self._validate_data(new_data_dict, new_covariate_dict, new_metadata)

        self._metadata = new_metadata
        self._data_dict = new_data_dict
        self._covariate_dict = new_covariate_dict

    # 整体设置前的校验钩子
    def _validate_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        covariate_dict: Dict[str, Dict],
        metadata: Optional[pd.DataFrame],
    ) -> NoReturn:
        """
        Validates if the given data_dict, covariate_dict and metadata are compatible

        Currently, we do not enforce any checks on the data.

        :param data_dict: A dictionary of DataFrames where the keys are the names of the
            series, and the values are DataFrames following the OTB protocol.
        :param covariate_dict: A dictionary of covariates where the keys are the names of the
            series, and the values are dictionary of the series' covariates.
        :param metadata: A DataFrame of meta information where the index contains series names
            and the columns contains meta-info fields. The value might be None when the
            metadata is not available.
        :return:
        """

    # 增量加入或覆盖数据
    def update_data(
        self,
        inc_data_dict: Dict[str, pd.DataFrame],
        inc_covariate_dict: Dict[str, Dict],
    ) -> NoReturn:
        """
        Updates the data dictionary

        :param inc_data_dict: The incremental dictionary of data.
        :param inc_covariate_dict: The incremental dictionary of covariate.
        """
        self._validate_update_data(inc_data_dict, inc_covariate_dict)
        self._data_dict.update(inc_data_dict)
        self._covariate_dict.update(inc_covariate_dict)

    # 增量更新前的校验钩子
    def _validate_update_data(
        self,
        inc_data_dict: Dict[str, pd.DataFrame],
        inc_covariate_dict: Dict[str, Dict],
    ) -> NoReturn:
        """
        Validates if the incremental data is compatible with the current data

        Currently, we do not enforce any checks on the incremental data.

        :param inc_data_dict: The incremental update of the data dictionary.
        :param inc_covariate_dict: The incremental update of the covariates dictionary.
        """

    # 清空全部内容
    def clear_data(self) -> NoReturn:
        """
        Clear all data stored in this dataset
        """
        self._metadata = None
        self._data_dict = {}
        self._covariate_dict = {}

    # 按名称获取序列
    def get_series(self, name: str) -> Optional[pd.DataFrame]:
        """
        Gets a single time series by name

        :param name: The name of the series to get.
        :return: A time series in DataFrame format. If the time series is not available,
            return None.
        """
        return self._data_dict.get(name, None)

    # 按名称获取协变量
    def get_covariates(self, name: str) -> Optional[Dict]:
        """
        Gets covariates by time series' name.

        :param name: The name of the series to get.
        :return: The time series' covariates in DataFrame format. If the covariates is not available,
            return None.
        """
        return self._covariate_dict.get(name, None)

    # 获取单条序列的元信息
    def get_series_meta_info(self, name: str) -> Optional[pd.Series]:
        """
        Gets the meta information of time series by name

        We do not return the meta information of unexisting series even if
        the meta information itself is available.

        :param name: The name of the series to get.
        :return: Meta information data in Series format. If the meta information or the
            corresponding series is not available, return None.
        """
        if name not in self._data_dict:
            return None
        if self._metadata is None or name not in self._metadata.index:
            return None
        return self._metadata.loc[name]

    # 判断序列是否已加载
    def has_series(self, name: str) -> bool:
        """
        Check if a series is available in the dataset

        :param name: The name of the series.
        :return: True if the series is contained in the dataset, False otherwise.
        """
        return name in self._data_dict

    # 判断已加载序列是否有元信息
    def has_series_meta_info(self, name: str) -> bool:
        """
        Check if the meta-info of a series is available in the dataset

        :param name: The name of the series.
        :return: True if the series and its meta-info are both contained in the dataset,
            False otherwise.
        """
        return (
            self.has_series(name)
            and self._metadata is not None
            and name in self._metadata.index
        )

    # 导出当前状态
    def get_state(self) -> Dict:
        """
        Get a serializable state

        :return: A dictionary of state which is composed of simple types as well as numpy
            or pandas array types.
        """
        return {
            "metadata": self._metadata,
            "data_dict": self._data_dict,
            "covariate_dict": self._covariate_dict,
        }

    # 从状态恢复
    def set_state(self, state: Dict) -> NoReturn:
        """
        Restores from a serializable state

        :param state: A dictionary of state which is composed of simple types as well as numpy
            or pandas array types.
        """
        self._metadata = state["metadata"]
        self._data_dict = state["data_dict"]
        self._covariate_dict = state["covariate_dict"]
