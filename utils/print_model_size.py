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