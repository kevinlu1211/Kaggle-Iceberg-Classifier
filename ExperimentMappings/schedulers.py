import torch
schedulers = {
    "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau
}