"""Behavior Cloning discrete agent (minimal fixed structure)."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from ._spec import TorchAgent
