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
Action Decoder with embodiment-specific weights (mirrors the encoder stack).

Architecture (symmetric to the encoder):
1. M-Former: map latent tokens to a feature sequence (state-conditioned).
2. ResNet-style blocks: refine features with residuals.
3. Upsample: interpolate + causal conv to actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass, field
from transformers import PretrainedConfig
from transformers.models.vit.modeling_vit import ViTConfig
from transformers.utils import ModelOutput
from transformers.image_processing_utils import BatchFeature

from .m_former import MFormer
from .action_branch_encoder import (
    CategorySpecificLayerNorm,
    CategorySpecificCausalConv1D,
    CategorySpecificCausalResBlock,
)


@dataclass
class ActionDecoderConfig(PretrainedConfig):
    """Action Decoder Configuration (Symmetric to Encoder)"""
    
    action_dim: int = field(default=None, metadata={"help": "Action dimension"})
    # Note: No state_dim - decoder expects pre-encoded state_features from encoder
    action_horizon: int = field(default=None, metadata={"help": "Action sequence length (output)"})
    hidden_size: int = field(default=768, metadata={"help": "Hidden size"})
    query_num: int = field(default=8, metadata={"help": "Number of input latent tokens"})
    
    # ResNet decoder (mirrors encoder layout)
    num_conv_layers: int = field(default=3, metadata={"help": "Number of conv layers (including upsample layer)"})
    conv_kernel_size: int = field(default=3, metadata={"help": "Conv kernel size"})
    upsample_stride: int = field(default=1, metadata={"help": "Final upsample stride (paired with encoder downsample)"})
    use_dilation: bool = field(default=False, metadata={"help": "Use dilated convolution in ResBlocks"})
    
    # M-Former config
    m_former_cfg: dict = field(default=None, metadata={"help": "M-Former configuration dict"})
    
    # Embodiment config
    max_num_embodiments: int = field(default=32, metadata={"help": "Maximum number of embodiments"})
    
    # Dropout
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate"})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class CategorySpecificUpsampleLayer(nn.Module):
    """
    Linear interpolate to `target_length`, then causal conv to `action_dim`.
    """
    def __init__(self, config: ActionDecoderConfig):
        super().__init__()
        self.config = config

        # Pre-upsample Norm
        self.norm = CategorySpecificLayerNorm(
            num_categories=config.max_num_embodiments,
            normalized_shape=config.hidden_size
        )
        
        # Projection Conv: hidden_size -> action_dim (causal)
        self.proj_conv = CategorySpecificCausalConv1D(
            num_categories=config.max_num_embodiments,
            in_channels=config.hidden_size,
            out_channels=config.action_dim,
            kernel_size=config.conv_kernel_size,
            stride=1,
            dilation=1
        )
    
    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Args:
            x: (B, T_in, hidden_size)
            cat_ids: (B,)
            target_length: desired output time steps (``action_horizon``)
        Returns:
            actions: (B, target_length, action_dim)
        """
        # 1. Norm
        x = self.norm(x, cat_ids)
        
        # 2. Interpolate to target length (if needed)
        if x.shape[1] != target_length:
            # (B, T_in, H) -> (B, H, T_in) -> interpolate -> (B, H, T_out) -> (B, T_out, H)
            x = x.transpose(1, 2)  # (B, H, T_in)
            x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
            x = x.transpose(1, 2)  # (B, T_out, H)
        
        # 3. Project to action_dim
        actions = self.proj_conv(x, cat_ids)  # (B, T_out, action_dim)
        
        return actions


class CategorySpecificActionResNetDecoder(nn.Module):
    """
    ResNet-style Action Decoder:
    Input Norm -> Stack of ResBlocks -> Upsample Layer
    """
    def __init__(self, config: ActionDecoderConfig):
        super().__init__()
        self.config = config

        self.input_norm = CategorySpecificLayerNorm(
            num_categories=config.max_num_embodiments,
            normalized_shape=config.hidden_size
        )

        self.layers = nn.ModuleList()
        num_res_blocks = max(0, config.num_conv_layers - 1)

        for _ in range(num_res_blocks):
            dilation = 1
            self.layers.append(
                CategorySpecificCausalResBlock(config, dilation=dilation)
            )

        self.upsample_layer = CategorySpecificUpsampleLayer(config)
    
    def forward(self, features: torch.Tensor, cat_ids: torch.Tensor, 
                target_length: int) -> torch.Tensor:
        """
        Args:
            features: (B, T_intermediate, hidden_size) — M-Former output
            cat_ids: (B,) embodiment ids
            target_length: desired ``action_horizon``
        Returns:
            actions: (B, target_length, action_dim)
        """
        # 1. Input Norm
        x = self.input_norm(features, cat_ids)
        
        # 2. Body: ResBlocks
        for layer in self.layers:
            x = layer(x, cat_ids)
        
        # 3. Upsample Layer: interpolate + project
        actions = self.upsample_layer(x, cat_ids, target_length)
        
        return actions


class ActionDecoder(nn.Module):
    """
    Multi-Embodiment Action Decoder (Symmetric to ActionEncoder)
    
    Decodes latent tokens back to action sequences with state conditioning.
    
    Architecture:
    1. M-Former: Expand latent tokens to intermediate feature sequence (with state condition)
    2. ResNet Decoder: Process features with residual blocks
    3. Upsample Layer: Interpolate + Conv to reconstruct actions
    """
    
    config_class = ActionDecoderConfig
    
    def __init__(self, config: ActionDecoderConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.query_num = config.query_num
        self.action_horizon = config.action_horizon

        # Note: No state_encoder in decoder!
        # State features should be pre-encoded by encoder and passed via action_input['state_features']
        # This avoids redundant encoding and DDP issues with unused parameters
        
        # M-Former mirrors the encoder: cond=state, target=latent tokens.
        m_former_max_seq_len = config.query_num + 1 + 1 + config.query_num

        if config.m_former_cfg is not None:
            m_former_vit_config = ViTConfig(**config.m_former_cfg)
            m_former_vit_config.query_num = config.query_num
            m_former_vit_config.hidden_size = config.hidden_size
            m_former_vit_config.input_hidden_size = config.hidden_size
            m_former_vit_config.max_seq_len = m_former_max_seq_len
            m_former_vit_config.num_patches = config.query_num
            self.m_former = MFormer(m_former_vit_config)
        else:
            m_former_vit_config = ViTConfig(
                hidden_size=config.hidden_size,
                num_hidden_layers=4,
                num_attention_heads=config.hidden_size // 64,
                intermediate_size=config.hidden_size * 4,
                hidden_dropout_prob=config.dropout,
                attention_probs_dropout_prob=config.dropout,
                query_num=config.query_num,
                input_hidden_size=config.hidden_size,
                max_seq_len=m_former_max_seq_len,
                num_patches=config.query_num,
            )
            self.m_former = MFormer(m_former_vit_config)
        
        # 3. ResNet Decoder
        self.resnet_decoder = CategorySpecificActionResNetDecoder(config)
        
        # Unified embodiment ID (for parameter sharing)
        # Dynamically set by parent model (e.g., gr00t_n1_tokenizer_unit.py)
        self.unified_embodiment_id = None
        
        self.dropout = nn.Dropout(config.dropout)
    
    def prepare_input(self, inputs: dict) -> BatchFeature:
        """
        Prepare action inputs from raw inputs
        
        Args:
            inputs: Raw input dictionary containing:
                - action: (B, T, action_dim)
                - state_features: (B, 1, hidden_size) - pre-encoded by encoder
                - embodiment_id: (B,)
                - action_mask: optional (B, T, action_dim)
        
        Returns:
            BatchFeature containing prepared action inputs
        """
        return BatchFeature(data={
            'action': inputs['action'],  # (B, T, action_dim)
            'state': inputs['state'],  # (B, state_dim)
            'embodiment_id': inputs['embodiment_id'],  # (B,)
            'action_mask': inputs.get('action_mask', None)  # (B, T, action_dim) or None
        })
    
    def _decode_internal(self,
                         latent_tokens: torch.Tensor,      # (B, query_num, hidden_size)
                         state_features: torch.Tensor,     # (B, 1, hidden_size) - Pre-encoded state features
                         cat_ids: torch.Tensor,             # (B,) - embodiment IDs
                         ) -> torch.Tensor:
        """
        Internal decode method (shared by forward and get_action)
        
        Args:
            latent_tokens: (B, query_num, hidden_size) - Latent representation from encoder
            state_features: (B, 1, hidden_size) - Pre-encoded state features (from encoder)
            cat_ids: (B,) - Embodiment category IDs
        
        Returns:
            actions: (B, action_horizon, action_dim) - Reconstructed actions
        """
        B = latent_tokens.shape[0]
        device = latent_tokens.device
        
        # [Unified Embodiment] Override cat_ids if unified_embodiment_id is set
        if self.unified_embodiment_id is not None:
            cat_ids = torch.full_like(cat_ids, self.unified_embodiment_id)
        
        # Step 1: State is already encoded upstream (encoder dropout applied there).

        # Step 2: M-Former (same wiring as encoder; targets are latent tokens here).
        m_former_output = self.m_former(
            cond_hidden_states=state_features,
            target_hidden_states=latent_tokens,
        )
        processed_tokens = m_former_output.last_hidden_state[:, :self.query_num, :]

        # Sequence length is restored by the decoder ResNet + upsample block, not M-Former.

        # Step 3: ResNet decoder + upsample to actions
        actions = self.resnet_decoder(
            features=processed_tokens,  # (B, query_num, H)
            cat_ids=cat_ids,
            target_length=self.action_horizon
        )
        # actions: (B, action_horizon, action_dim)
        
        return actions
    
    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Training forward pass - compute loss
        
        Args:
            backbone_output: BatchFeature containing:
                - backbone_features: (B, N, hidden_size) - latent tokens from encoder
                - backbone_attention_mask: (B, N) - attention mask
            action_input: BatchFeature containing:
                - action: (B, T, action_dim) - ground truth actions
                - state_features: (B, 1, hidden_size) - Pre-encoded state (REQUIRED)
                - embodiment_id: (B,) - embodiment IDs
                - action_mask: (B, T, action_dim) or None - valid elements mask
        
        Returns:
            BatchFeature containing {'loss': tensor}
        """
        # Extract inputs
        backbone_features = backbone_output['backbone_features']  # (B, N, H)
        gt_actions = action_input['action']  # (B, T, action_dim)
        cat_ids = action_input['embodiment_id']  # (B,)
        action_mask = action_input.get('action_mask', None)  # (B, T, action_dim) or None
        
        # Get pre-encoded state features (MUST be provided by encoder)
        state_features = backbone_output['state_features']  # (B, 1, H) - pre-encoded by encoder
        # Extract latent tokens (first query_num tokens)
        latent_tokens = backbone_features[:, :self.query_num, :]  # (B, query_num, H)

        gt_targets = gt_actions
        pred_outputs = self._decode_internal(latent_tokens, state_features, cat_ids)  # (B, T, action_dim)
        
        # Compute Smooth L1 Loss
        if action_mask is not None:
            # Masked loss
            loss_per_element = F.smooth_l1_loss(pred_outputs, gt_targets, reduction='none', beta=0.5)  # (B, T, action_dim)
            loss_per_element = loss_per_element * action_mask  # Apply mask
            
            # Per-sample loss (for multi-scenario training)
            loss_per_sample = loss_per_element.sum(dim=[1, 2]) / action_mask.sum(dim=[1, 2]).clamp(min=1.0)  # (B,)
            
            # Overall loss (mean over batch)
            loss = loss_per_sample.mean()
        else:
            # Unmasked loss
            loss_per_element = F.smooth_l1_loss(pred_outputs, gt_targets, reduction='none')  # (B, T, action_dim)
            loss_per_sample = loss_per_element.mean(dim=[1, 2])  # (B,)
            loss = loss_per_sample.mean()
        
        return BatchFeature(data={
            'loss': loss,
            'per_sample_loss': loss_per_sample,  # For multi-scenario grouping
        })
    
    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Inference - get predicted actions
        
        Args:
            backbone_output: BatchFeature containing:
                - backbone_features: (B, N, hidden_size) - latent tokens from encoder
                - backbone_attention_mask: (B, N) - attention mask
            action_input: BatchFeature containing:
                - state_features: (B, 1, hidden_size) - Pre-encoded state (REQUIRED)
                - embodiment_id: (B,) - embodiment IDs

        Returns:
            BatchFeature containing {'action_pred': (B, T, action_dim)}
        """
        # Extract inputs
        backbone_features = backbone_output['backbone_features']  # (B, N, H)
        cat_ids = action_input['embodiment_id']  # (B,)
        
        # Get pre-encoded state features (MUST be provided by encoder)
        state_features = backbone_output['state_features']  # (B, 1, H) - pre-encoded by encoder
        
        # Extract latent tokens (first query_num tokens)
        latent_tokens = backbone_features[:, :self.query_num, :]  # (B, query_num, H)
        
        pred_actions = self._decode_internal(latent_tokens, state_features, cat_ids)  # (B, T, action_dim)

        return BatchFeature(data={'action_pred': pred_actions})
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    
    def get_decoder_info(self) -> dict:
        """Get decoder module information"""
        return {
            "type": "ActionDecoder",
            "action_dim": self.config.action_dim,
            "action_horizon": self.config.action_horizon,
            "hidden_size": self.hidden_size,
            "query_num": self.query_num,
            "num_conv_layers": self.config.num_conv_layers,
            "conv_kernel_size": self.config.conv_kernel_size,
            "upsample_stride": self.config.upsample_stride,
            "upsample_ratio": self.config.action_horizon / self.query_num,
            "use_dilation": self.config.use_dilation,
            "max_num_embodiments": self.config.max_num_embodiments,
            "m_former_num_layers": self.m_former.config.num_hidden_layers,
            "architecture": "M-Former (Process) -> ResNet (Decode) -> Upsample",
            "loss_type": "SmoothL1Loss",
            "state_encoding": "Uses pre-encoded state_features from encoder (no own state_encoder)",
        }
