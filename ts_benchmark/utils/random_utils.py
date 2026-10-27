import os
import random
from typing import Optional, NoReturn

import numpy as np
import torch


def fix_random_seed(seed: Optional[int] = 2021) -> NoReturn:
    if seed is None:
        return

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def fix_all_random_seed(seed: Optional[int] = 2021) -> NoReturn:
    if seed is None:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Keep cuDNN available; deterministic algorithms and disabled benchmarking
    # control its reproducibility without disabling GPU convolution acceleration.
    # Python reads this setting at interpreter startup, so this assignment only
    # controls subsequently started child interpreters.
    os.environ["PYTHONHASHSEED"] = str(seed)
