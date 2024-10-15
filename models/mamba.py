import torch
import torch.nn as nn

class MambaTemporalModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.ipi_output_dim + config.epi_output_dim, config.hidden_dim)
        self.gru = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        self.linear2 = nn.Linear(config.hidden_dim, config.output_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x, _ = self.gru(x)
        return self.linear2(x)
