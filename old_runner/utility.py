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
            if not init_method or init_method == "default":
                m.reset_parameters()
            elif init_method == "he_uniform":
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            else:
                raise ValueError("Unknown initialization method: " + init_method)


def get_random_matrix(shape, method="uniform", multiplier=0.005):
    """
    Initialize the random projection matrix F_proj.
    For 'uniform', F_proj is initialized with uniform.
    For 'ones', F_proj is initialized as before.
    For 'he_uniform', we use Kaiming He uniform initialization.
    """

    if not method or method == "uniform":
        F_proj = torch.rand(*shape)
    elif method == "gaussian":
        F_proj = torch.randn(*shape)
    elif method == "ones":
        F_proj = torch.ones(*shape)
    elif method == "he_uniform":
        F_proj = torch.empty(*shape)
        nn.init.kaiming_uniform_(F_proj, nonlinearity='relu')
    else:
        raise ValueError("Unknown initialization method: " + method)
    return F_proj * multiplier
