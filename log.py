import torch

def nan_gradients(v_model):
    for name, param in v_model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN gradient in {name} of v_model")


def check_for_nan_inf(tensor, name="Tensor"):
    if torch.isnan(tensor).any():
        print(f"Warning: NaN values detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Warning: Inf values detected in {name}")
