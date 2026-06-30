import torch.distributed as dist


def rank0_print(*args, **kwargs):
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)
