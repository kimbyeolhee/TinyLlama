import torch
from torch.nn import init

def rms_norm(x, weight, epsilon):
    return DropoutAddLayerNormFn.apply(
        x, None, weight, None, None, None, 0.0, epsilon, False, False, True
    )

class FusedRMSNorm(torch.nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.dim = dim
        self.reset_parameters()
    
    def reset_parameters(self):
        init.ones_(self.weight) 
    
    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)

class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Lauyer Normalization
    """
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed
    
    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)