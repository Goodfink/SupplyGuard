import os
import random

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch

from supplyguard.config import DEFAULT_CONFIG

runtime_config = DEFAULT_CONFIG.runtime

random.seed(runtime_config.random_seed)
np.random.seed(runtime_config.random_seed)
torch.manual_seed(runtime_config.random_seed)
torch.set_num_threads(runtime_config.torch_threads)
torch.use_deterministic_algorithms(runtime_config.deterministic_algorithms)
