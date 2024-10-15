import torch
import torch.nn as nn

class MambaTemporalModule(nn.Module):
    """
    MambaTemporalModule for temporal modeling in the AnimateX model.

    This module processes the combined output of the Implicit Pose Indicator (IPI)
    and Explicit Pose Indicator (EPI) through a series of linear layers and a GRU
    to capture temporal dependencies in the pose representations.
    """

    def __init__(self, config):
        """
        Initialize the MambaTemporalModule.

        Args:
            config: Configuration object containing model parameters.
                Expected attributes:
                - ipi_output_dim (int): Output dimension of the Implicit Pose Indicator.
                - epi_output_dim (int): Output dimension of the Explicit Pose Indicator.
                - hidden_dim (int): Dimension of the hidden layers and GRU.
                - output_dim (int): Dimension of the output.
        """
        super().__init__()
        self.linear1 = nn.Linear(config.ipi_output_dim + config.epi_output_dim, config.hidden_dim)
        self.gru = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        self.linear2 = nn.Linear(config.hidden_dim, config.output_dim)
    
    def forward(self, x):
        """
        Forward pass of the MambaTemporalModule.

        Args:
            x (torch.Tensor): Input tensor containing combined IPI and EPI outputs.
                Shape: (batch_size, sequence_length, ipi_output_dim + epi_output_dim)

        Returns:
            torch.Tensor: Processed temporal features.
                Shape: (batch_size, sequence_length, output_dim)
        """
        x = self.linear1(x)
        x, _ = self.gru(x)
        return self.linear2(x)
