import random
from typing import Any
from typing import Callable

import numpy as np
import torch
from ice_offline.dataset._spec import TorchBuffer


Callback = Callable[[], list[Any]]


def sample_transition(
    batch_size: int,
    obs_dim: int,
    act_dim: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    s = torch.as_tensor(np.random.standard_normal((batch_size, obs_dim)), dtype=torch.float32, device=device)
    a = torch.as_tensor(np.random.standard_normal((batch_size, act_dim)), dtype=torch.float32, device=device)
    r = torch.as_tensor(np.random.standard_normal((batch_size, 1)), dtype=torch.float32, device=device)
    sn = torch.as_tensor(np.random.standard_normal((batch_size, obs_dim)), dtype=torch.float32, device=device)
    d = torch.as_tensor(np.random.randint(0, 2, size=(batch_size, 1)), dtype=torch.float32, device=device)
    return s, a, r, sn, d


def torch_buffer(
    s: torch.Tensor,
    a: torch.Tensor,
    r: torch.Tensor,
    sn: torch.Tensor,
    d: torch.Tensor,
) -> TorchBuffer:
    return TorchBuffer(
        obs_list=s,
        next_obs_list=sn,
        act_list=a,
        rew_list=r,
        done_list=d,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _max_abs_diff(x: Any, y: Any) -> float:
    if torch.is_tensor(x) and torch.is_tensor(y):
        return float((x - y).abs().max().item())
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    return float(np.max(np.abs(x_arr - y_arr)))


def assert_list(ref_list: list[Any], our_list: list[Any], label: str) -> None:
    if len(ref_list) != len(our_list):
        raise SystemExit(f"FAIL: {label} length mismatch ref={len(ref_list)} our={len(our_list)}")

    for idx, (ref_item, our_item) in enumerate(zip(ref_list, our_list)):
        diff = _max_abs_diff(ref_item, our_item)
        if diff != 0.0:
            raise SystemExit(f"FAIL: {label}[{idx}] mismatch max_abs_diff={diff:.12e}")

    print(f"PASS: {label}")


def assert_callback(ref_callback: Callback, our_callback: Callback, label: str, seed: int) -> tuple[list[Any], list[Any]]:
    _set_seed(seed)
    ref_list = ref_callback()
    _set_seed(seed)
    our_list = our_callback()
    assert_list(ref_list, our_list, label=label)
    return ref_list, our_list
