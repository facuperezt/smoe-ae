import torch

def get_model_size_MB(model: torch.nn.Module):
    """
    Returns model size in MB, assumes AdamW optimizer
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    size_all_mb += 8*sum([p.numel() for p in model.parameters()])/1024**2
    # print('model size: {:.3f}MB'.format(size_all_mb))
    return size_all_mb


def get_gpu_memory_usage(device: torch.device):
    """
    Returns the current GPU memory usage with torch.mps.current_allocated_memory()
    """
    if device.type == "mps":
        return (torch.mps.current_allocated_memory() + torch.mps.driver_allocated_memory()) / 1024**2
    elif device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1024**2
    elif device.type == "cpu":
        return 0
    else:
        raise ValueError("Invalid device type")


def clear_gpu_cache(device: torch.device):
    """
    Clears the GPU cache
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cpu":
        pass
    else:
        raise ValueError("Invalid device type")
    
