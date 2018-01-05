import torch
import torch.optim.lr_scheduler
optimizers = {
    "SGD": torch.optim.SGD,
    "ADAM": torch.optim.Adam,
    "RMSprop": torch.optim.RMSprop,
    "Adadelta": torch.optim.Adadelta
}

scheduler = {
    "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau
}