import torch
import torch.nn as nn


def init_model_weights(model, init_method="default"):
    """
    Initialize model weights based on the chosen method.
    By default (init_method="default"), we use the built-in reset_parameters()
    so that the behavior is identical to before.
    For 'he_uniform', we apply Kaiming He uniform initialization.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_method == "he_uniform":
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_method == "default":
                m.reset_parameters()
            else:
                raise ValueError("Unknown initialization method: " + init_method)


def initialize_F_proj(shape, init_method="default"):
    """
    Initialize the random projection matrix F_proj.
    For 'default', F_proj is initialized as before.
    For 'he_uniform', we use Kaiming He uniform initialization.
    """
    if init_method == "he_uniform":
        F_proj = torch.empty(*shape)
        nn.init.kaiming_uniform_(F_proj, nonlinearity='relu')
        F_proj = F_proj * 0.005
    elif init_method == "default":
        # Keep same behavior as original: Gaussian with scaling 0.005
        F_proj = torch.randn(*shape) * 0.005
    else:
        raise ValueError("Unknown initialization method: " + init_method)
    return F_proj
