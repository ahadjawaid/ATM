import torch

def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.norm(a - b, p=2, dim=-1).mean()