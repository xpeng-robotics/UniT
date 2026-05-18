# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Action Encoder with Embodiment-aware Processing

Architecture:
1. 1D Causal Conv (embodiment-aware) for action sequence encoding
2. State MLP (embodiment-aware) for state encoding
3. M-Former (All-in-One) for temporal modeling and query extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass, field
from transformers import PretrainedConfig
from transformers.models.vit.modeling_vit import ViTConfig

from .m_former import MFormer




@dataclass
class ActionEncoderConfig(PretrainedConfig):
    """Action Encoder Configuration"""
    
    action_dim: int = field(default=None, metadata={"help": "Action dimension"})
    state_dim: int = field(default=None, metadata={"help": "State dimension"})
    action_horizon: int = field(default=None, metadata={"help": "Action sequence length (fixed)"})
    hidden_size: int = field(default=768, metadata={"help": "Hidden size"})
    query_num: int = field(default=8, metadata={"help": "Number of output query tokens"})
    
    # Conv encoder config (ResNet)
    num_conv_layers: int = field(default=3, metadata={"help": "Number of conv layers"})
    conv_kernel_size: int = field(default=3, metadata={"help": "Conv kernel size"})
    conv_stride: int = field(default=1, metadata={"help": "Conv stride for downsampling (1=no downsample, 2=halve length)"})
    use_dilation: bool = field(default=False, metadata={"help": "Use dilated convolution"})
    downsample_target_len: int = field(default=None, metadata={"help": "Target sequence length after conv (optional, for verification)"})
    
    # M-Former (All-in-One) config
    m_former_cfg: dict = field(default=None, metadata={"help": "M-Former configuration dict"})
    
    # Embodiment config
    max_num_embodiments: int = field(default=32, metadata={"help": "Maximum number of embodiments"})

    # Dropout
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate"})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

class CategorySpecificCausalConv1D(nn.Module):
    """
    Embodiment-aware 1D Causal Convolution Layer using Grouped Convolution
    No for-loops, fully vectorized.
    """
    
    def __init__(self, num_categories: int, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.num_categories = num_categories
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        # Per-embodiment conv weights (see `CategorySpecificCausalConv1D.forward`).
        self.W = nn.Parameter(
            0.02 * torch.randn(num_categories, out_channels, in_channels, kernel_size)
        )
        self.b = nn.Parameter(torch.zeros(num_categories, out_channels))
        
    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C_in) input tensor
            cat_ids: (B,) embodiment category IDs
        """
        B, T, C_in = x.shape

        x = x.transpose(1, 2)

        if self.padding > 0:
            x = F.pad(x, (self.padding, 0))
        T_padded = x.shape[-1]

        x_grouped = x.reshape(1, B * C_in, T_padded)

        weights = self.W[cat_ids]
        biases = self.b[cat_ids]

        w_grouped = weights.reshape(B * self.out_channels, self.in_channels, self.kernel_size)

        b_grouped = biases.reshape(B * self.out_channels)

        out = F.conv1d(
            x_grouped, 
            w_grouped, 
            bias=b_grouped, 
            stride=self.stride, 
            dilation=self.dilation,
            groups=B,
        )

        out = out.view(B, self.out_channels, -1)

        return out.transpose(1, 2)


class CategorySpecificLinear(nn.Module):
    """Embodiment-aware Linear Layer"""
    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Embodiment-aware MLP"""
    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class CategorySpecificLayerNorm(nn.Module):
    """
    Embodiment-specific LayerNorm.
    Applies standard LayerNorm statistics (mean/std) but uses 
    embodiment-specific affine parameters (gamma/beta).
    """
    def __init__(self, num_categories: int, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.num_categories = num_categories

        self.weight = nn.Parameter(torch.ones(num_categories, normalized_shape))
        self.bias = nn.Parameter(torch.zeros(num_categories, normalized_shape))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) or (B, C)
            cat_ids: (B,)
        """
        out = F.layer_norm(x, self.normalized_shape, weight=None, bias=None, eps=self.eps)

        gamma = self.weight[cat_ids] # (B, C)
        beta = self.bias[cat_ids]    # (B, C)

        if x.dim() == 3:
            gamma = gamma.unsqueeze(1) # (B, 1, C)
            beta = beta.unsqueeze(1)   # (B, 1, C)

        return out * gamma + beta


class CategorySpecificCausalResBlock(nn.Module):
    """
    Pre-Norm Causal Residual Block:
    x = x + Dropout(Conv(Act(Norm(x))))
    """
    def __init__(self, config: ActionEncoderConfig, dilation: int = 1):
        super().__init__()
        
        # 1. Norm (Embodiment-Aware)
        self.norm = CategorySpecificLayerNorm(
            num_categories=config.max_num_embodiments,
            normalized_shape=config.hidden_size
        )
        
        # 2. Activation
        self.activation = nn.GELU()
        
        # 3. Conv (Category-Specific, Causal)
        self.conv = CategorySpecificCausalConv1D(
            num_categories=config.max_num_embodiments,
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conv_kernel_size,
            stride=1,
            dilation=dilation
        )
        
        # 4. Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.norm(x, cat_ids)
        out = self.activation(out)
        out = self.conv(out, cat_ids)
        out = self.dropout(out)

        return residual + out


class CategorySpecificActionResNet(nn.Module):
    """
    ResNet-style Action Conv Encoder:
    Stem -> Stack of Causal ResBlocks -> Final Norm
    """
    def __init__(self, config: ActionEncoderConfig):
        super().__init__()
        self.config = config

        self.stem_conv = CategorySpecificCausalConv1D(
            num_categories=config.max_num_embodiments,
            in_channels=config.action_dim,
            out_channels=config.hidden_size,
            kernel_size=config.conv_kernel_size,
            stride=config.conv_stride,
            dilation=1
        )
        self.stem_norm = CategorySpecificLayerNorm(
            num_categories=config.max_num_embodiments,
            normalized_shape=config.hidden_size
        )

        self.layers = nn.ModuleList()
        num_res_blocks = max(0, config.num_conv_layers - 1)

        for i in range(num_res_blocks):
            dilation = 2**i if config.use_dilation else 1
            self.layers.append(
                CategorySpecificCausalResBlock(config, dilation=dilation)
            )

        self.final_norm = CategorySpecificLayerNorm(
            num_categories=config.max_num_embodiments,
            normalized_shape=config.hidden_size
        )

    def forward(self, actions: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: (B, T, action_dim)
            cat_ids: (B,)
        Returns:
            features: (B, T_downsampled, hidden_size) - Ready for M-Former
        """
        # 1. Stem: Projection & Downsampling & Alignment
        x = self.stem_conv(actions, cat_ids)
        x = self.stem_norm(x, cat_ids)
        
        # 2. Body: Deep Feature Extraction (Residual)
        for layer in self.layers:
            x = layer(x, cat_ids)
            
        # 3. Final Alignment for M-Former
        x = self.final_norm(x, cat_ids)
        
        return x
    
    def get_output_length(self, input_length: int) -> int:
        """Helper to calculate expected output sequence length"""
        # Stem layer stride affects length
        # Formula for Causal Conv with padding: floor((L-1)/stride + 1)
        import math
        return math.floor((input_length - 1) / self.config.conv_stride + 1)


class ActionEncoder(nn.Module):
    """
    Multi-Embodiment Action Encoder
    
    Encodes action sequences and state into fixed number of query tokens.
    Uses M-Former (All-in-One) for both temporal modeling and query extraction.

    Architecture:
    1. Embodiment-aware Causal ResNet Conv: extract local temporal features from actions.
    2. Embodiment-aware MLP: encode state features.
    3. M-Former (All-in-One): temporal modeling + query extraction
       (handles positional encoding internally and token type via cond/target separation).
    """
    
    config_class = ActionEncoderConfig
    
    def __init__(self, config: ActionEncoderConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.query_num = config.query_num

        # 1. Action Conv Encoder (embodiment-aware ResNet)
        self.action_conv_encoder = CategorySpecificActionResNet(config)
        
        # 2. State Encoder (embodiment-aware)
        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.state_dim,
            hidden_dim=config.hidden_size,
            output_dim=config.hidden_size
        )
        
        # 3. M-Former (All-in-One): Temporal modeling + Query extraction
        # Calculate expected sequence length after conv downsampling
        if config.downsample_target_len is not None:
            expected_action_len = config.downsample_target_len
        else:
            # Calculate from stride
            expected_action_len = config.action_horizon // config.conv_stride
        
        # M-Former sequence: [query_tokens, cond, sep, target]
        # = query_num + 1 (state) + 1 (sep) + expected_action_len
        m_former_max_seq_len = config.query_num + 1 + 1 + expected_action_len
        
        if config.m_former_cfg is not None:
            m_former_vit_config = ViTConfig(**config.m_former_cfg)
            # Override critical parameters
            m_former_vit_config.query_num = config.query_num
            m_former_vit_config.hidden_size = config.hidden_size
            m_former_vit_config.input_hidden_size = config.hidden_size
            m_former_vit_config.max_seq_len = m_former_max_seq_len
            m_former_vit_config.num_patches = expected_action_len
            self.m_former = MFormer(m_former_vit_config)
        else:
            # Default M-Former config
            m_former_vit_config = ViTConfig(
                hidden_size=config.hidden_size,
                num_hidden_layers=4,  # Increase layers since M-Former now handles all temporal modeling
                num_attention_heads=config.hidden_size // 64,  # Standard head_dim=64
                intermediate_size=config.hidden_size * 4,
                hidden_dropout_prob=config.dropout,
                attention_probs_dropout_prob=config.dropout,
                query_num=config.query_num,
                input_hidden_size=config.hidden_size,
                max_seq_len=m_former_max_seq_len,
                num_patches=expected_action_len,
            )
            self.m_former = MFormer(m_former_vit_config)
        
        # Store expected length for validation
        self.expected_action_len = expected_action_len
        
        # Unified embodiment ID (for parameter sharing across embodiments).
        # Dynamically set by parent model (e.g., gr00t_n1_tokenizer_unit.py).
        self.unified_embodiment_id = None

        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, 
                actions: torch.Tensor,           # (B, T, action_dim)
                state: torch.Tensor,             # (B, state_dim)
                cat_ids: torch.Tensor,           # (B,) embodiment IDs
                timesteps: Optional[torch.Tensor] = None,  # Not used, kept for compatibility
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode actions and state into query tokens
        
        Args:
            actions: Action sequence (B, T, action_dim) 
                    Note: T may vary, will be adjusted to action_horizon
            state: State vector (B, state_dim)
            cat_ids: Embodiment category IDs (B,)
            timesteps: Not used, kept for compatibility
            
        Returns:
            Tuple of:
                - action_tokens: (B, query_num, hidden_size)
                - state_features: (B, 1, hidden_size) for decoder conditioning
        """
        B, T, _ = actions.shape
        device = actions.device
        
        # [Unified Embodiment] Optionally collapse all samples to a single embodiment id
        # so that category-specific weights are shared across embodiments.
        if self.unified_embodiment_id is not None:
            cat_ids = torch.full_like(cat_ids, self.unified_embodiment_id)
        
        # Step 1: Action Conv Encoding (ResNet)
        action_features = self.action_conv_encoder(actions, cat_ids)  # (B, T_downsampled, H)
        T_downsampled = action_features.shape[1]
        
        # Verify downsampled length matches expected
        if T_downsampled != self.expected_action_len:
            # Use adaptive pooling as fallback
            action_features = action_features.transpose(1, 2)  # (B, H, T_downsampled)
            action_features = F.adaptive_avg_pool1d(action_features, self.expected_action_len)  # (B, H, expected_len)
            action_features = action_features.transpose(1, 2)  # (B, expected_len, H)
            print(f"Warning: Conv output length {T_downsampled} != expected {self.expected_action_len}, "
                  f"adjusted via adaptive pooling")
        
        action_features = self.dropout(action_features)
        # action_features: (B, expected_action_len, hidden_size)
        
        # Step 2: State Encoding
        state_features = self.state_encoder(state, cat_ids)  # (B, 1, H)
        state_features = self.dropout(state_features)
        # state_features: (B, 1, hidden_size)
        
        # Step 3: M-Former (All-in-One) for temporal modeling and query extraction
        # M-Former will handle:
        # - Positional encoding internally (via self.position_embeddings)
        # - Token type distinction (via cond vs target separation)
        # - Temporal modeling (via self-attention across full sequence)
        # - Query extraction (via learnable query tokens)
        m_former_output = self.m_former(
            cond_hidden_states=state_features,      # (B, 1, H) - state as condition
            target_hidden_states=action_features    # (B, expected_action_len, H) - actions as target
        )
        # m_former_output.last_hidden_state: (B, query_num + 1 + 1 + expected_action_len, H)
        # Layout: [query_tokens, cond, sep, target]
        # Extract the query tokens (first query_num tokens)
        action_tokens = m_former_output.last_hidden_state[:, :self.query_num, :]  # (B, query_num, H)

        return action_tokens, state_features
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    
    def get_encoder_info(self) -> dict:
        """Get encoder module information"""
        return {
            "type": "ActionEncoder",
            "action_dim": self.config.action_dim,
            "state_dim": self.config.state_dim,
            "action_horizon": self.config.action_horizon,
            "expected_action_len": self.expected_action_len,
            "hidden_size": self.hidden_size,
            "query_num": self.query_num,
            "num_conv_layers": self.config.num_conv_layers,
            "conv_kernel_size": self.config.conv_kernel_size,
            "conv_stride": self.config.conv_stride,
            "downsample_ratio": self.config.action_horizon / self.expected_action_len,
            "use_dilation": self.config.use_dilation,
            "max_num_embodiments": self.config.max_num_embodiments,
            "m_former_num_layers": self.m_former.config.num_hidden_layers,
            "architecture": "ResNet Conv -> M-Former (All-in-One)",
        }


