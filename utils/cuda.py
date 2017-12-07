import torch


def cudarize(var, **kwargs):
    if torch.cuda.is_available():
        return var.cuda(**kwargs)
    else:
        return var
