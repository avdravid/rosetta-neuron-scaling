"""Distributed helpers for match_lm."""

from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


def dist_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def dist_barrier() -> None:
    if dist_active():
        dist.barrier()


def init_distributed_from_env():
    """
    If launched with torchrun, env vars RANK/WORLD_SIZE/LOCAL_RANK will exist.
    Returns (distributed, rank, world_size, local_rank, device).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Use 2-hour timeout to handle imbalanced block distribution across ranks
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(hours=4),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return True, rank, world_size, local_rank, device
    else:
        return False, 0, 1, 0, None
