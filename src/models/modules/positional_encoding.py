# --------------------------------------------------------------------------------
# Copyright (c) 2025 Gabriele Lozupone (University of Cassino and Southern Lazio).
# All rights reserved.
# --------------------------------------------------------------------------------
#
# LICENSE NOTICE
# *************************************************************************************************************
# By downloading/using/running/editing/changing any portion of codes in this package you agree to the license.
# If you do not agree to this license, do not download/use/run/edit/change this code.
# Refer to the LICENSE file in the root directory of this repository for full details.
# *************************************************************************************************************
#
# Contact: Gabriele Lozupone at gabriele.lozupone@unicas.it
# -----------------------------------------------------------------------------


import math
import torch
from torch import nn

"""
Originally ported from here: https://github.com/ckczzj/PDAE/tree/master and adapted for the LDAE framework.
"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 80):
        """
    Args:
      d_model:      dimension of embeddings
      dropout:      randomly zeroes-out some of the input
      max_length:   max sequence length
    """
        # inherit from Module
        super().__init__()
        # initialize dropout
        self.dropout = nn.Dropout(p=dropout)
        # create tensor of 0s
        pe = torch.zeros(max_length, d_model)
        # create position column
        k = torch.arange(0, max_length).unsqueeze(1)
        # calc divisor for positional encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # calc sine on even indices
        pe[:, 0::2] = torch.sin(k * div_term)
        # calc cosine on odd indices
        pe[:, 1::2] = torch.cos(k * div_term)
        # add dimension
        pe = pe.unsqueeze(0)
        # buffers are saved in state_dict but not trained by the optimizer
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
    Args:
      x:        embeddings (batch_size, seq_length, d_model)
    Returns:
                embeddings + positional encodings (batch_size, seq_length, d_model)
    """
        # add positional encoding to the embeddings
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        # perform dropout
        return self.dropout(x)
