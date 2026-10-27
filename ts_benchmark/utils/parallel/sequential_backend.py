# -*- coding: utf-8 -*-

from __future__ import absolute_import

import os
import warnings
from typing import Tuple, Any, NoReturn, Callable, Optional, List, Dict

from ts_benchmark.utils.parallel.base import TaskResult, SharedStorage


class SequentialResult(TaskResult):

    def __init__(self):
        self._result = None

    def result(self) -> Any:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    def put(self, value: Any) -> NoReturn:
        self._result = value


class SequentialSharedStorage(SharedStorage):

    def __init__(self):
        self.storage = {}

    def put(self, name: str, value: Any) -> NoReturn:
        self.storage[name] = value

    def get(self, name: str, default_value: Any = None) -> Any:
        return self.storage.get(name, default_value)


class SequentialBackend:

    def __init__(self, gpu_devices: Optional[List[int]] = None, worker_initializers=None, **kwargs):
        super().__init__()
        self.gpu_devices = gpu_devices if gpu_devices is not None else []
        self.storage = None
        self.worker_initializers = (
            worker_initializers if isinstance(worker_initializers, list)
            else [worker_initializers] if worker_initializers is not None else []
        )

    def init(self) -> NoReturn:
        self.storage = SequentialSharedStorage()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_devices))
        for initializer in self.worker_initializers:
            initializer(self.env)

    def schedule(self, fn: Callable, args: Tuple, timeout: float = -1) -> SequentialResult:
        if timeout != -1:
            warnings.warn("timeout is not supported by SequentialBackend, ignoring")
        res = SequentialResult()
        try:
            res.put(fn(*args))
        except Exception as error:
            res.put(error)
        return res

    def close(self, force: bool = False):
        pass

    @property
    def shared_storage(self) -> SharedStorage:
        return self.storage

    @property
    def env(self) -> Dict:
        return {
            "storage": self.shared_storage,
        }

    def execute_on_workers(self, func: Callable) -> NoReturn:
        func(self.env)

    def add_worker_initializer(self, func: Callable) -> NoReturn:
        pass
