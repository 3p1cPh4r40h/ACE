import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiDilationBlock(nn.Module):
    """
    Applies multiple dilated convolutions in parallel to the input
    and concatenates the results. Uses 'same' padding.
    """
    def __init__(self, in_channels, branch_out_channels, dilations, kernel_size=3):
        """
        Args:
            in_channels (int): Number of input channels.
            branch_out_channels (int): Number of output channels *per branch*.
            dilations (list or tuple of int): Dilation rates for each parallel branch.
            kernel_size (int): Kernel size for the convolutions.
        """
        super(MultiDilationBlock, self).__init__()
        self.dilations = dilations
        self.branches = nn.ModuleList()

        for dilation in dilations:
            # Calculate 'same' padding for this branch
            padding = math.floor(((kernel_size - 1) * dilation) / 2)
            self.branches.append(
                nn.Conv2d(in_channels,
                          branch_out_channels,
                          kernel_size=kernel_size,
                          padding=padding,
                          dilation=dilation)
            )

        # The total number of output channels after concatenation
        self.total_out_channels = len(dilations) * branch_out_channels

    def forward(self, x):
        branch_outputs = []
        for branch_conv in self.branches:
            branch_outputs.append(branch_conv(x))

        # Concatenate along the channel dimension (dim=1)
        # Input x shape: (batch, in_channels, H, W)
        # Output shape: (batch, total_out_channels, H, W)
        return torch.cat(branch_outputs, dim=1)