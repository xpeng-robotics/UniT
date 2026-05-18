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
Vision Branch Encoder: vision_model → M-Former → visual query tokens

Extracts visual query tokens from observation and goal images
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional
from transformers import PretrainedConfig
from transformers.models.vit.modeling_vit import ViTConfig

from .m_former import MFormer


@dataclass
class VisionBranchEncoderConfig(PretrainedConfig):
    """Configuration for Vision Branch Encoder"""
    
    query_num: int = field(default=8, metadata={"help": "Number of query tokens to extract"})
    hidden_size: int = field(default=768, metadata={"help": "Hidden size"})
    m_former_cfg: dict = field(default=None, metadata={"help": "M-Former configuration dict"})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class VisionBranchEncoder(nn.Module):
    """
    Vision Branch Encoder
    
    Architecture:
    1. Vision Model (from backbone): Extract patch-level features from images
    2. M-Former: Extract fixed number of query tokens from patch features
    
    This module handles both observation and goal images through the same pipeline.
    """
    
    config_class = VisionBranchEncoderConfig
    
    def __init__(self, config: VisionBranchEncoderConfig, vision_model: nn.Module):
        """
        Args:
            config: Configuration for the vision branch
            vision_model: Pre-initialized vision backbone (e.g. DINOv2 wrapper).
        """
        super().__init__()
        self.config = config
        self.query_num = config.query_num
        self.hidden_size = config.hidden_size
        self.vision_model = vision_model

        if config.m_former_cfg is not None:
            m_former_vit_config = ViTConfig(**config.m_former_cfg)
            # Override critical parameters
            m_former_vit_config.query_num = config.query_num
            m_former_vit_config.hidden_size = config.hidden_size
            m_former_vit_config.input_hidden_size = config.hidden_size
        else:
            # Default M-Former config
            m_former_vit_config = ViTConfig(
                hidden_size=config.hidden_size,
                num_hidden_layers=4,
                num_attention_heads=max(1, config.hidden_size // 64),
                intermediate_size=config.hidden_size * 4,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                query_num=config.query_num,
                input_hidden_size=config.hidden_size,
            )
        
        self.m_former = MFormer(m_former_vit_config)
    
    def forward(self,
                obs_input,
                goal_input,
                batch_size: int = 1):
        """
        Extract visual query tokens from observation and goal images.

        Args:
            obs_input/goal_input: either a dict of vision-model kwargs (eagle-style)
                or a tensor of pixel values (DINOv2-style).
            batch_size: outer batch size used to reshape patch features.

        Returns:
            visual_tokens: (B, query_num, hidden_size) M-Former query outputs.
            obs_features:  (B, N_patches, hidden_size) patch features for obs.
            goal_features: (B, N_patches, hidden_size) patch features for goal.
        """
        if goal_input is None:
            raise ValueError("Goal input is required for vision branch encoder")

        with torch.no_grad():
            if isinstance(obs_input, dict):
                obs_features = self.vision_model(*obs_input.values()).reshape(batch_size, -1, self.hidden_size)
                goal_features = self.vision_model(*goal_input.values()).reshape(batch_size, -1, self.hidden_size)
            else:
                if obs_input.dim() == 5:
                    obs_input = obs_input.squeeze(1)
                if goal_input.dim() == 5:
                    goal_input = goal_input.squeeze(1)
                obs_features = self.vision_model(obs_input).reshape(batch_size, -1, self.hidden_size)
                goal_features = self.vision_model(goal_input).reshape(batch_size, -1, self.hidden_size)

        # M-Former layout: [query_tokens, cond, sep, target] -- take the first query_num tokens.
        m_former_output = self.m_former(
            cond_hidden_states=obs_features,
            target_hidden_states=goal_features,
        )
        visual_tokens = m_former_output.last_hidden_state[:, :self.query_num, :]
        return visual_tokens, obs_features, goal_features
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    
    def get_encoder_info(self) -> dict:
        """Get encoder module information"""
        return {
            "type": "VisionBranchEncoder",
            "query_num": self.query_num,
            "hidden_size": self.hidden_size,
            "m_former_num_layers": self.m_former.config.num_hidden_layers,
            "architecture": "vision_model -> M-Former(obs_cond, goal_target) -> query_tokens"
        }

