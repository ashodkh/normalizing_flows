import torch
import torch.nn as nn

class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim // 2),
            nn.Tanh()
        )
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim // 2)
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        y1 = x1
        y2 = x2 * torch.exp(s) + t
        log_det_jacobian = s.sum(dim=-1)
        return torch.cat([y1, y2], dim=-1), log_det_jacobian

    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=-1)
        s = self.scale_net(y1)
        t = self.translate_net(y1)
        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([x1, x2], dim=-1)
    
class NormalizingFlow(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([AffineCouplingLayer(input_dim) for _ in range(num_layers)])

    def forward(self, x):
        log_det_jacobian = 0
        for i, layer in enumerate(self.layers):
            x, ldj = layer(x)
            log_det_jacobian += ldj
            if i % 2 == 1:  # Swap dimensions after every other layer
                x = x.flip(dims=[-1])
        return x, log_det_jacobian

    def inverse(self, y):
        for i, layer in enumerate(reversed(self.layers)):
            if i % 2 == 1:  # Swap dimensions before feeding into the layer
                y = y.flip(dims=[-1])
            y = layer.inverse(y)
        return y