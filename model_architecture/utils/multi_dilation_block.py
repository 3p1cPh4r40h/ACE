import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_architecture.utils.squeeze_excitation import SqueezeExcitationBlock

class MultiDilationBlock(nn.Module):
    """
    Applies multiple dilated convolutions in parallel to the input
    and concatenates the results. Uses 'same' padding.
    Optionally includes attention mechanisms at different stages.
    """
    def __init__(self, in_channels, branch_out_channels, dilations, kernel_size=3,
                 attention_type=None, attention_reduction=16, attention_activation="sigmoid"):
        """
        Args:
            in_channels (int): Number of input channels.
            branch_out_channels (int): Number of output channels *per branch*.
            dilations (list or tuple of int): Dilation rates for each parallel branch.
            kernel_size (int): Kernel size for the convolutions.
            attention_type (str, optional): Type of attention to apply. Can be 'early', 'mid', 'late', or None.
            attention_reduction (int): Reduction ratio for the attention block.
            attention_activation (str): Activation function for attention ('sigmoid' or 'softmax').
        """
        super(MultiDilationBlock, self).__init__()
        self.dilations = dilations
        self.branches = nn.ModuleList()
        self.attention_type = attention_type

        # Initialize attention blocks if specified
        if attention_type == 'early':
            self.early_attention = SqueezeExcitationBlock(
                channels=in_channels,
                reduction=attention_reduction,
                activation=attention_activation
            )
        elif attention_type == 'mid':
            self.mid_attention = SqueezeExcitationBlock(
                channels=branch_out_channels,
                reduction=attention_reduction,
                activation=attention_activation
            )
        elif attention_type == 'late':
            self.late_attention = SqueezeExcitationBlock(
                channels=len(dilations) * branch_out_channels,
                reduction=attention_reduction,
                activation=attention_activation
            )

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
        # Apply early attention if specified
        if self.attention_type == 'early':
            x = self.early_attention(x)

        # Process through branches
        branch_outputs = []
        for i, branch_conv in enumerate(self.branches):
            branch_out = branch_conv(x)
            
            # Apply mid attention if specified (per branch)
            if self.attention_type == 'mid':
                branch_out = self.mid_attention(branch_out)
                
            branch_outputs.append(branch_out)

        # Concatenate along the channel dimension (dim=1)
        x = torch.cat(branch_outputs, dim=1)

        # Apply late attention if specified (after concatenation)
        if self.attention_type == 'late':
            x = self.late_attention(x)

        return x