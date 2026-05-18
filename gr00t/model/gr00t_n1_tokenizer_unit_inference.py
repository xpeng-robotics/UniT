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

from dataclasses import dataclass, field
from typing import Tuple, Optional
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature


def denormalize_imagenet(images: torch.Tensor) -> torch.Tensor:
    """Denormalize ImageNet-normalized images to [0, 1]. Expects (B, C, H, W)."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images = images * std + mean
    return torch.clamp(images, 0, 1)


BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


class DINOv2Wrapper(nn.Module):
    """Wraps Dinov2Model: strips CLS; ``layer_index`` selects hidden layer (-1 = last)."""

    def __init__(self, dinov2_model, layer_index=-1):
        super().__init__()
        self.model = dinov2_model
        self.config = dinov2_model.config
        self.layer_index = layer_index
        
        print(f"DINOv2Wrapper: Using layer {layer_index} for features")
    
    def forward(self, pixel_values, **kwargs):
        if self.layer_index != -1:
            kwargs['output_hidden_states'] = True
        
        outputs = self.model(pixel_values, **kwargs)
        
        if self.layer_index == -1:
            hidden_state = outputs.last_hidden_state
        else:
            hidden_state = outputs.hidden_states[self.layer_index]
        
        return hidden_state[:, 1:, :]
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
    
    def requires_grad_(self, requires_grad=True):
        self.model.requires_grad_(requires_grad)
        return self


@dataclass
class GR00T_Tokenizer_Config(PretrainedConfig):
    """VQ tokenizer: vision + action encoders → fusion → VQ → latent motion + action decoders."""

    model_type = "gr00t_tokenizer"
    
    backbone_cfg: dict = field(init=False, metadata={"help": "DINO/backbone"})
    vision_branch_cfg: dict = field(init=False, metadata={"help": "Vision branch"})
    action_encoder_cfg: dict = field(init=False, metadata={"help": "Action encoder"})
    fusion_cfg: dict = field(default_factory=dict, metadata={"help": "Fusion"})
    vq_cfg: dict = field(init=False, metadata={"help": "Vector quantizer"})
    vision_decoder_cfg: dict = field(init=False, metadata={"help": "Latent motion decoder (ViT-based)"})
    action_decoder_cfg: dict = field(init=False, metadata={"help": "ResNet ActionDecoder"})
    
    action_horizon: int = field(init=False, metadata={"help": "Action horizon"})
    action_dim: int = field(init=False, metadata={"help": "Action dimension"})
    state_dim: int = field(init=False, metadata={"help": "State dimension"})
    query_num: int = field(default=8, metadata={"help": "Query tokens"})
    hidden_size: int = field(default=768, metadata={"help": "Hidden size"})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype"})
    
    reconstruction_loss_weights: dict = field(default_factory=lambda: {
        'vision': 1.0,
        'action': 1.0,
        'vq_commitment': 0.25,
        'lpips': 1.0,
    }, metadata={"help": "Loss weights"})
    use_lpips_loss: bool = field(default=False, metadata={"help": "LPIPS in pixel reconstruction"})
    
    default_pv: int = field(default=1, metadata={"help": "Default visual presence flag"})
    default_pa: int = field(default=1, metadata={"help": "Default action presence flag"})
    
    tune_vision_model: bool = field(default=True, metadata={"help": "Train vision backbone"})
    tune_vision_m_former: bool = field(default=True, metadata={"help": "Train vision M-Former"})
    tune_action_encoder: bool = field(default=True, metadata={"help": "Train action encoder"})
    tune_fusion: bool = field(default=True, metadata={"help": "Train fusion"})
    tune_vq: bool = field(default=True, metadata={"help": "Train VQ"})
    tune_vision_decoder: bool = field(default=True, metadata={"help": "Train latent motion decoder"})
    tune_action_decoder_projector: bool = field(default=True, metadata={"help": "Train action decoder (projector)"})
    tune_action_decoder_diffusion: bool = field(default=True, metadata={"help": "Train action decoder (diffusion/head)"})
    unified_embodiment_id: int = field(default=None, metadata={
        "help": "If set, all samples use this embodiment ID in category-specific heads"
    })
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

class GR00T_Tokenizer(PreTrainedModel):
    """VQ tokenizer: vision + action encoders → fusion → RVQ → latent motion + action decoders."""

    supports_gradient_checkpointing = True
    config_class = GR00T_Tokenizer_Config
    
    def __init__(
        self,
        config: GR00T_Tokenizer_Config,
        local_model_path: str = None,
        compute_bridge_loss: bool=True,
        unified_embodiment_id: int = None,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.vision_branch_cfg, dict)
        assert isinstance(config.action_encoder_cfg, dict)
        assert isinstance(config.vq_cfg, dict)
        assert isinstance(config.vision_decoder_cfg, dict)
        assert isinstance(config.action_decoder_cfg, dict)
        
        super().__init__(config)
        self.local_model_path = local_model_path

        from transformers import Dinov2Model
        
        dinov2_path = config.backbone_cfg.get('dinov2_path', 'facebook/dinov2-large')
        dinov2_layer_index = config.backbone_cfg.get('dinov2_layer_index', -2)
        dinov2_model = Dinov2Model.from_pretrained(dinov2_path)
        vision_model = DINOv2Wrapper(dinov2_model, layer_index=dinov2_layer_index)

        tune_visual = config.backbone_cfg.get('tune_visual', False)
        vision_model.requires_grad_(tune_visual)

        self.vision_model_type = "dinov2"
        self.tune_visual = tune_visual
        self.compute_bridge_loss = compute_bridge_loss
        
        from gr00t.model.tokenizer.vision_branch_encoder import (
            VisionBranchEncoder, VisionBranchEncoderConfig
        )
        vision_branch_cfg = VisionBranchEncoderConfig(**config.vision_branch_cfg)
        self.vision_branch = VisionBranchEncoder(
            config=vision_branch_cfg,
            vision_model=vision_model
        )
        
        from gr00t.model.tokenizer.action_branch_encoder import (
            ActionEncoder, ActionEncoderConfig
        )
        action_encoder_cfg = ActionEncoderConfig(**config.action_encoder_cfg)
        self.action_branch = ActionEncoder(action_encoder_cfg)
        
        from gr00t.model.tokenizer.visual_action_fusion import QFormerVisualActionFusion
        self.fusion = QFormerVisualActionFusion(
            hidden_size=config.hidden_size,
            query_num=config.query_num,
            fusion_config=config.fusion_cfg
        )
        
        from gr00t.model.tokenizer.vector_quantizer import (
            ResidualVectorQuantizer, ResidualVectorQuantizerConfig,
            ResidualVQFromLib, ResidualFSQFromLib,
        )
        vq_cfg = ResidualVectorQuantizerConfig(**config.vq_cfg)
        vq_type = config.vq_cfg.get('vq_type', 'default')
        
        if vq_type == 'ema':
            self.vq = ResidualVQFromLib(vq_cfg)
            print(f"✓ Using ResidualVQFromLib (ema mode)")
        elif vq_type == 'fsq':
            self.vq = ResidualFSQFromLib(vq_cfg)
            print(f"✓ Using ResidualFSQFromLib (fsq mode)")
        else:
            self.vq = ResidualVectorQuantizer(vq_cfg)
            print(f"✓ Using ResidualVectorQuantizer (standard RVQ)")
        
        from gr00t.model.tokenizer.latent_motion_decoder import LatentMotionDecoder
        from transformers.models.vit.modeling_vit import ViTConfig
        vision_decoder_vit_cfg = ViTConfig(**config.vision_decoder_cfg)
        self.vision_decoder = LatentMotionDecoder(vision_decoder_vit_cfg)
        
        from gr00t.model.tokenizer.action_branch_decoder import (
            ActionDecoder, ActionDecoderConfig,
        )
        action_decoder_cfg = ActionDecoderConfig(**config.action_decoder_cfg)
        self.action_decoder = ActionDecoder(action_decoder_cfg)
        
        _unified_embodiment_id = unified_embodiment_id if unified_embodiment_id is not None else config.unified_embodiment_id
        if _unified_embodiment_id is not None:
            print(f"✓ Unified embodiment ID: {_unified_embodiment_id} (action encoder/decoder will share parameters)")
            self.action_branch.unified_embodiment_id = _unified_embodiment_id
            self.action_decoder.unified_embodiment_id = _unified_embodiment_id
        self.unified_embodiment_id = _unified_embodiment_id
        
        self.pos_embed = nn.Parameter(torch.zeros(config.query_num, config.hidden_size))
        self.vq_down_resampler = nn.Sequential(
            nn.Linear(config.hidden_size, config.vq_cfg['e_dim']),
        )
        self.bridge_projector = nn.Sequential(
            nn.Linear(config.vq_cfg['e_dim'], config.hidden_size),
        )
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.query_num = config.query_num
        self.hidden_size = config.hidden_size
        self.compute_dtype = config.compute_dtype
        self.loss_weights = config.reconstruction_loss_weights
        self.use_lpips_loss = config.use_lpips_loss if config.use_lpips_loss is not None else True
        
        self.is_dino_mode = config.vision_decoder_cfg.get('is_io_hidden_states', False)
        
        if self.use_lpips_loss:
            try:
                import lpips
                lpips_model = lpips.LPIPS(net='vgg', verbose=False).requires_grad_(False).eval()
                object.__setattr__(self, '_lpips_loss_module', lpips_model)
                print("✓ LPIPS loss initialized (excluded from checkpoints)")
            except ImportError:
                print("⚠️  LPIPS not available. Install with: pip install lpips")
                self.use_lpips_loss = False
                object.__setattr__(self, '_lpips_loss_module', None)
        else:
            object.__setattr__(self, '_lpips_loss_module', None)
        
        self.default_pv = config.default_pv if config.default_pv is not None else 1
        self.default_pa = config.default_pa if config.default_pa is not None else 1
    
    @property
    def lpips_loss(self):
        """
        Property to access LPIPS loss module (excluded from checkpoints)
        
        Lazy loading: Ensures LPIPS is on the correct device when accessed.
        If LPIPS is on meta device (from fast loading), reinitialize it on the target device.
        """
        if self._lpips_loss_module is None:
            return None
        
        try:
            lpips_params = list(self._lpips_loss_module.parameters())
            if len(lpips_params) > 0:
                lpips_device = lpips_params[0].device
                model_device = self.device
                
                if lpips_device.type == 'meta':
                    print(f"⚠️  LPIPS is on meta device, reinitializing on {model_device}")
                    try:
                        import lpips
                        lpips_model = lpips.LPIPS(net='vgg', verbose=False).requires_grad_(False).eval()
                        lpips_model = lpips_model.to(model_device)
                        object.__setattr__(self, '_lpips_loss_module', lpips_model)
                        print(f"✓ LPIPS reinitialized on {model_device}")
                    except Exception as reinit_e:
                        print(f"⚠️  Failed to reinitialize LPIPS: {reinit_e}")
                        object.__setattr__(self, '_lpips_loss_module', None)
                        return None
                
                elif lpips_device != model_device:
                    print(f"Moving LPIPS from {lpips_device} to {model_device}")
                    self._lpips_loss_module = self._lpips_loss_module.to(model_device)
        except Exception as e:
            print(f"Warning: Failed to ensure LPIPS device: {e}")            
            return None
        
        return self._lpips_loss_module
    
    def state_dict(self, *args, **kwargs):
        """Override state_dict to exclude LPIPS loss module"""
        state_dict = super().state_dict(*args, **kwargs)
        keys_to_remove = [k for k in state_dict.keys() if '_lpips_loss_module' in k]
        for key in keys_to_remove:
            del state_dict[key]
        return state_dict
    
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Allow missing LPIPS keys."""
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
    
    def prepare_input(self, inputs: dict):
        """Map batch dict to tokenizer tensors (ImageNet-style obs/goal stacks)."""

        obs_input = inputs["imagenet_obs_images"]
        goal_input = inputs["imagenet_goal_images"]

        action_inputs = self.action_decoder.prepare_input(inputs)
        batch_size = inputs["state"].shape[0]
        device = inputs["state"].device if torch.is_tensor(inputs["state"]) else None

        if "pv" in inputs:
            pv = inputs["pv"].to(device) if hasattr(inputs["pv"], "to") else torch.tensor(inputs["pv"], device=device, dtype=torch.long)
        else:
            pv = torch.full((batch_size,), self.default_pv, dtype=torch.long, device=device)

        if "pa" in inputs:
            pa = inputs["pa"].to(device) if hasattr(inputs["pa"], "to") else torch.tensor(inputs["pa"], device=device, dtype=torch.long)
        else:
            pa = torch.full((batch_size,), self.default_pa, dtype=torch.long, device=device)

        tokenizer_inputs = BatchFeature(data={
            "target_action": inputs["action"],
            "pv": pv,
            "pa": pa,
        })

        def to_device_with_maybe_dtype(x):
            if not isinstance(x, torch.Tensor):
                return x
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            return x.to(self.device)

        obs_input = tree.map_structure(to_device_with_maybe_dtype, obs_input)
        goal_input = tree.map_structure(to_device_with_maybe_dtype, goal_input)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        tokenizer_inputs = tree.map_structure(to_device_with_maybe_dtype, tokenizer_inputs)

        return obs_input, goal_input, action_inputs, tokenizer_inputs
    
    def forward(
        self, 
        inputs: dict, 
        action_mode: bool = False,
        return_motion_token_ids_only: bool = False
    ) -> BatchFeature:
        """
        Forward pass

        Inputs require ``imagenet_obs_images``, ``imagenet_goal_images``, action/state/embodiment,
        optional ``pv`` / ``pa``, and optionally ``global_step`` (stripped before processing).
        """
        inputs.pop("global_step", None)

        obs_input, goal_input, action_inputs, tokenizer_inputs = self.prepare_input(inputs)
        batch_size = inputs["state"].shape[0]

        # Vision / action branches: per-query features (B, query_num, hidden_size)
        vision_query_features, obs_embeds, goal_embeds_dino = self.vision_branch(
            obs_input=obs_input,
            goal_input=goal_input,
            batch_size=batch_size
        )

        action_query_features, state_features = self.action_branch(
            actions=action_inputs["action"],
            state=action_inputs["state"],
            cat_ids=action_inputs["embodiment_id"],
        )

        # Routed unit tokens (hidden_size); then linear map to VQ dim (`unit_tokens_down`).
        unit_tokens = self.fusion(
            visual_tokens=vision_query_features,
            action_tokens=action_query_features,
            pv=tokenizer_inputs["pv"],
            pa=tokenizer_inputs["pa"]
        )

        unit_tokens_down = self.vq_down_resampler(unit_tokens)
        quantized_tokens, vq_indices, vq_loss = self.vq(unit_tokens_down)
        quantized_tokens_up = self.bridge_projector(quantized_tokens)

        if return_motion_token_ids_only:
            return BatchFeature(data={
                "indices": vq_indices,
                "quant": quantized_tokens,
                "before_quant": unit_tokens_down,
            })

        goal_embeds = quantized_tokens_up + self.pos_embed.unsqueeze(0)

        output_dict = {}
        vision_recon_loss = torch.tensor(
            0.0, device=quantized_tokens.device, dtype=torch.float32
        )

        if self.compute_bridge_loss:
            if obs_input is not None and goal_input is not None:
                is_dino_mode = getattr(self, "is_dino_mode", False)

                if is_dino_mode:
                    cond_dino = obs_embeds
                    target_dino = goal_embeds_dino
                    reconstructed_dino = self.vision_decoder(
                        cond_input=cond_dino,
                        latent_motion_tokens=quantized_tokens_up
                    )
                    cos_sim = F.cosine_similarity(reconstructed_dino, target_dino, dim=-1)
                    cos_loss_per_sample = 1 - cos_sim.mean(dim=-1)
                    cos_loss = cos_loss_per_sample.mean()
                    vision_recon_loss = cos_loss

                    output_dict["vision_cos_loss"] = cos_loss
                    output_dict["vision_cos_sim"] = cos_sim.mean()
                    output_dict["vision_recon_loss"] = vision_recon_loss
                else:
                    cond_images = obs_input.squeeze(1)
                    target_images = goal_input.squeeze(1)
                    reconstructed_images = self.vision_decoder(
                        cond_input=cond_images,
                        latent_motion_tokens=quantized_tokens_up
                    )
                    mse_per_sample = F.mse_loss(
                        reconstructed_images, target_images, reduction="none"
                    ).mean(dim=[1, 2, 3])
                    mse_loss = mse_per_sample.mean()
                    vision_recon_loss = mse_loss
                    output_dict["vision_mse_loss"] = mse_loss

                    if self.use_lpips_loss and self.lpips_loss is not None:
                        lpips_per_sample = self.lpips_loss(
                            denormalize_imagenet(reconstructed_images) * 2 - 1,
                            denormalize_imagenet(target_images) * 2 - 1,
                        ).view(-1)
                        lpips_loss = lpips_per_sample.mean()
                        vision_recon_loss = (
                            vision_recon_loss
                            + self.loss_weights.get("lpips", 0.1) * lpips_loss
                        )
                        output_dict["vision_lpips_loss"] = lpips_loss

                    output_dict["vision_recon_loss"] = vision_recon_loss
            else:
                vision_recon_loss = torch.tensor(
                    0.0, device=quantized_tokens.device, dtype=torch.float32
                )
                output_dict["vision_recon_loss"] = vision_recon_loss
        else:
            output_dict["vision_recon_loss"] = vision_recon_loss

        if not action_mode:
            action_decoder_input = BatchFeature(data={
                "backbone_features": goal_embeds,
                "state_features": state_features,
            })

            action_head_outputs = self.action_decoder(action_decoder_input, action_inputs)
            action_recon_loss = action_head_outputs["loss"]
            output_dict["action_recon_loss"] = action_recon_loss

            total_loss = (
                self.loss_weights["vision"] * output_dict["vision_recon_loss"]
                + self.loss_weights["action"] * action_recon_loss
                + self.loss_weights["vq_commitment"] * vq_loss
            )
            output_dict["loss"] = total_loss
            output_dict["vq_loss"] = vq_loss

            if self.training:
                with torch.no_grad():
                    if hasattr(self.vq, "layers"):
                        for layer_idx in range(len(self.vq.layers)):
                            layer_indices = vq_indices[..., layer_idx]
                            unique_codes = torch.unique(layer_indices).numel()
                            output_dict[f"vq_active_codes_layer_{layer_idx}"] = unique_codes
                    else:
                        unique_codes = torch.unique(vq_indices).numel()
                        output_dict["vq_active_codes"] = unique_codes

            return BatchFeature(data=output_dict)

        action_decoder_input = BatchFeature(data={
            "backbone_features": goal_embeds,
            "state_features": state_features,
        })
        action_pred = self.action_decoder.get_action(action_decoder_input, action_inputs)
        action_pred["vq_indices"] = vq_indices
        action_pred.update(output_dict)
        return action_pred

    def set_trainable_parameters(self,
                                tune_vision_model: bool = True,
                                tune_vision_m_former: bool = True,
                                tune_bridge_projector: bool = True,
                                tune_action_encoder: bool = True,
                                tune_fusion: bool = True,
                                tune_vq: bool = True,
                                tune_vision_decoder: bool = True,
                                tune_action_decoder_projector: bool = True,
                                tune_action_decoder_diffusion: bool = True):
        """Set tune_* flags then apply ``requires_grad_`` on subtrees."""
        self.tune_vision_model = tune_vision_model
        self.tune_vision_m_former = tune_vision_m_former
        self.tune_bridge_projector = tune_bridge_projector
        self.tune_action_encoder = tune_action_encoder
        self.tune_fusion = tune_fusion
        self.tune_vq = tune_vq
        self.tune_vision_decoder = tune_vision_decoder
        
        if tune_vision_model:
            self.vision_branch.vision_model.requires_grad_(True)
        else:
            self.vision_branch.vision_model.requires_grad_(False)
        
        if tune_vision_m_former:
            self.vision_branch.m_former.requires_grad_(True)
        else:
            self.vision_branch.m_former.requires_grad_(False)
        
        if tune_bridge_projector:
            self.bridge_projector.requires_grad_(True)
        else:
            self.bridge_projector.requires_grad_(False)
        
        # Action Encoder
        if tune_action_encoder:
            self.action_branch.requires_grad_(True)
        else:
            self.action_branch.requires_grad_(False)
        
        if tune_fusion:
            self.fusion.requires_grad_(True)
        else:
            self.fusion.requires_grad_(False)
        
        if tune_vq:
            self.vq.requires_grad_(True)
        else:
            self.vq.requires_grad_(False)
        
        if tune_vision_decoder:
            self.vision_decoder.requires_grad_(True)
        else:
            self.vision_decoder.requires_grad_(False)
        
        if hasattr(self.action_decoder, 'set_trainable_parameters'):
            self.action_decoder.set_trainable_parameters(
                tune_projector=tune_action_decoder_projector,
                tune_diffusion_model=tune_action_decoder_diffusion
            )
        else:
            if tune_action_decoder_projector or tune_action_decoder_diffusion:
                self.action_decoder.requires_grad_(True)
            else:
                self.action_decoder.requires_grad_(False)

        print(f"Tune vision model (shared): {tune_vision_model}")
        print(f"Tune vision M-Former: {tune_vision_m_former}")
        print(f"Tune bridge projector: {tune_bridge_projector}")
        print(f"Tune action encoder: {tune_action_encoder}")
        print(f"Tune fusion: {tune_fusion}")
        print(f"Tune VQ: {tune_vq}")
        print(f"Tune vision decoder: {tune_vision_decoder}")
        print(f"Tune action decoder projector: {tune_action_decoder_projector}")
        print(f"Tune action decoder diffusion: {tune_action_decoder_diffusion}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """HF load + optional dinov2 path override."""
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        kwargs.pop("output_dir", None)
        kwargs.pop("tune_image_type_embedding", None)

        dinov2_path_override = kwargs.pop("dinov2_path_override", None)
        if dinov2_path_override is not None:
            print(f">> Overriding tokenizer dinov2_path: {dinov2_path_override}")
            model_config.backbone_cfg['dinov2_path'] = dinov2_path_override
        unified_embodiment_id = kwargs.pop("unified_embodiment_id", getattr(model_config, 'unified_embodiment_id', None))
        tune_vision_model = kwargs.pop("tune_vision_model", True)
        tune_vision_m_former = kwargs.pop("tune_vision_m_former", True)
        tune_bridge_projector = kwargs.pop("tune_bridge_projector", True)
        tune_action_encoder = kwargs.pop("tune_action_encoder", True)
        tune_fusion = kwargs.pop("tune_fusion", True)
        tune_vq = kwargs.pop("tune_vq", True)
        tune_vision_decoder = kwargs.pop("tune_vision_decoder", True)
        tune_action_decoder_projector = kwargs.pop("tune_action_decoder_projector", True)
        tune_action_decoder_diffusion = kwargs.pop("tune_action_decoder_diffusion", True)

        print(f"Loading pretrained tokenizer from {pretrained_model_name_or_path}")
        print(f"Tune vision model (shared): {tune_vision_model}")
        print(f"Tune vision M-Former: {tune_vision_m_former}")
        print(f"Tune bridge projector: {tune_bridge_projector}")
        print(f"Tune action encoder: {tune_action_encoder}")
        print(f"Tune fusion: {tune_fusion}")
        print(f"Tune VQ: {tune_vq}")
        print(f"Tune vision decoder: {tune_vision_decoder}")
        print(f"Tune action decoder projector: {tune_action_decoder_projector}")
        print(f"Tune action decoder diffusion: {tune_action_decoder_diffusion}")
        print(f"Unified embodiment ID: {unified_embodiment_id}")
        
        try:
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            print(f"Model not found in HF hub. Loading from local path: {pretrained_model_name_or_path}")
            local_model_path = pretrained_model_name_or_path
            
        try:
            pretrained_model = super(GR00T_Tokenizer, cls).from_pretrained(
                local_model_path, 
                local_model_path=local_model_path,
                unified_embodiment_id=unified_embodiment_id,
                **kwargs
            )
        except Exception as e:
            print(f"Load pretrained model error: {e}")
            print("Initializing model from config...")
            config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
            pretrained_model = cls(config, local_model_path=local_model_path, unified_embodiment_id=unified_embodiment_id)
        
        pretrained_model.set_trainable_parameters(
            tune_vision_model=tune_vision_model,
            tune_vision_m_former=tune_vision_m_former,
            tune_bridge_projector=tune_bridge_projector,
            tune_action_encoder=tune_action_encoder,
            tune_fusion=tune_fusion,
            tune_vq=tune_vq,
            tune_vision_decoder=tune_vision_decoder,
            tune_action_decoder_projector=tune_action_decoder_projector,
            tune_action_decoder_diffusion=tune_action_decoder_diffusion,
        )
        
        pretrained_model.config.tune_vision_model = tune_vision_model
        pretrained_model.config.tune_vision_m_former = tune_vision_m_former
        pretrained_model.config.tune_bridge_projector = tune_bridge_projector
        pretrained_model.config.tune_action_encoder = tune_action_encoder
        pretrained_model.config.tune_fusion = tune_fusion
        pretrained_model.config.tune_vq = tune_vq
        pretrained_model.config.tune_vision_decoder = tune_vision_decoder
        pretrained_model.config.tune_action_decoder_projector = tune_action_decoder_projector
        pretrained_model.config.tune_action_decoder_diffusion = tune_action_decoder_diffusion
        pretrained_model.config.unified_embodiment_id = unified_embodiment_id
        
        return pretrained_model
    
    def get_action(self, inputs: dict) -> BatchFeature:
        return self.forward(inputs=inputs, action_mode=True)
    
    def decode_from_embedding(
        self,
        cond_pixel_values: torch.Tensor,
        given_motion_embeddings: torch.Tensor,
        action_inputs: BatchFeature = None,
        obs_embeds: torch.Tensor = None,
        state_features: torch.Tensor = None
    ) -> dict:
        """Decode motion tokens → pixels + optional action (bridge policy / tokenizer eval).

        Unused in dual-system VQ-ID path. ``obs_embeds`` ignored (signature compat)."""
        _ = obs_embeds
        device = given_motion_embeddings.device

        cond_pixel_values = cond_pixel_values.to(device, dtype=self.dtype)
        given_motion_embeddings = given_motion_embeddings.to(device, dtype=self.dtype)

        projected_embeddings = self.bridge_projector(given_motion_embeddings)

        reconstructed_images = self.vision_decoder(
            cond_input=cond_pixel_values,
            latent_motion_tokens=projected_embeddings
        )

        output_dict = {
            'recons_pixel_values': reconstructed_images
        }

        if action_inputs is not None and self.default_pa == 1:
            if state_features is None:
                with torch.no_grad():
                    embodiment_id = action_inputs['embodiment_id']
                    if self.unified_embodiment_id is not None:
                        embodiment_id = torch.full_like(embodiment_id, self.unified_embodiment_id)
                    state_features = self.action_branch.state_encoder(
                        action_inputs['state'].to(self.dtype), embodiment_id)

            goal_embeds = projected_embeddings + self.pos_embed.unsqueeze(0)
            action_decoder_input = BatchFeature(data={
                'backbone_features': goal_embeds,
                'state_features': state_features,
            })

            with torch.no_grad():
                action_pred_outputs = self.action_decoder.get_action(action_decoder_input, action_inputs)

            output_dict['action_pred'] = action_pred_outputs['action_pred']

        return output_dict
    
    @property
    def codebook_weights(self):
        """VQ embedding rows per layer for latent_bridge (FSQ: implicit codebook projected)."""
        vq_type = getattr(self.config, 'vq_cfg', {}).get('vq_type', 'default')
        if vq_type == 'fsq':
            weights = []
            for layer in self.vq.layers:
                if hasattr(layer, '_fsq') and hasattr(layer._fsq, 'implicit_codebook'):
                    fsq = layer._fsq
                    w = fsq.implicit_codebook
                    with torch.no_grad():
                        w = fsq.project_out(w)
                else:
                    codebook_size = self.vq.n_e
                    codebook_dim = self.vq.e_dim
                    w = torch.zeros(codebook_size, codebook_dim, device=self.vq.device)
                weights.append(w)
            return weights
        
        if hasattr(self.vq, 'layers'):
            weights = []
            for layer in self.vq.layers:
                if hasattr(layer, 'embedding'):
                    w = layer.embedding.weight
                    if getattr(layer.config, 'l2_norm', False):
                        w = F.normalize(w, p=2, dim=-1)
                    weights.append(w)
                else:
                    raise AttributeError(f"Layer {layer} does not have embedding attribute")
            return weights
        else:
            raise AttributeError("VQ structure not recognized. Expected self.vq.layers")
    
    @property
    def num_codebooks(self):
        """Residual VQ depth (``len(vq.layers)``)."""
        if hasattr(self.vq, 'layers'):
            return len(self.vq.layers)
        else:
            raise AttributeError("VQ structure not recognized. Expected self.vq.layers")
    
    @property
    def num_bridge_tokens(self):
        """Number of motion query tokens (= ``query_num``)."""
        return self.query_num
    
    def set_frozen_modules_to_eval_mode(self):
        """Put frozen subtrees in eval mode while global ``train()`` stays on."""
        if self.training:
            if not getattr(self, 'tune_vision_model', True):
                self.vision_branch.vision_model.eval()
            if not getattr(self, 'tune_vision_m_former', True):
                self.vision_branch.m_former.eval()
            if not getattr(self, 'tune_bridge_projector', True):
                self.bridge_projector.eval()
            if not getattr(self, 'tune_action_encoder', True):
                self.action_branch.eval()
            if not getattr(self, 'tune_fusion', True):
                self.fusion.eval()
            if not getattr(self, 'tune_vq', True):
                self.vq.eval()
            if not getattr(self, 'tune_vision_decoder', True):
                self.vision_decoder.eval()
            if hasattr(self.action_decoder, 'set_frozen_modules_to_eval_mode'):
                self.action_decoder.set_frozen_modules_to_eval_mode()
    
    def to(self, *args, **kwargs):
        """Override to() to also move LPIPS loss module"""
        result = super().to(*args, **kwargs)
        # Move LPIPS loss module if it exists
        # Note: LPIPS is only needed during training, skip if it's on meta device or fails to move
        if hasattr(self, '_lpips_loss_module') and self._lpips_loss_module is not None:
            try:
                # Check if LPIPS is on meta device (happens with fast loading)
                # If so, skip moving it as we don't need it for inference
                lpips_params = list(self._lpips_loss_module.parameters())
                if len(lpips_params) > 0 and lpips_params[0].device.type != 'meta':
                    self._lpips_loss_module = self._lpips_loss_module.to(*args, **kwargs)
            except (RuntimeError, NotImplementedError) as e:
                # If moving fails (e.g., meta tensor issue), skip it
                # LPIPS is only needed for training, not inference
                pass
        return result
    
    def cuda(self, device=None):
        """Override cuda() to also move LPIPS loss module"""
        result = super().cuda(device)
        if hasattr(self, '_lpips_loss_module') and self._lpips_loss_module is not None:
            try:
                # Check if LPIPS is on meta device
                lpips_params = list(self._lpips_loss_module.parameters())
                if len(lpips_params) > 0 and lpips_params[0].device.type != 'meta':
                    self._lpips_loss_module = self._lpips_loss_module.cuda(device)
            except (RuntimeError, NotImplementedError):
                # Skip if moving fails (LPIPS only needed for training)
                pass
        return result
    
    def cpu(self):
        """Override cpu() to also move LPIPS loss module"""
        result = super().cpu()
        if hasattr(self, '_lpips_loss_module') and self._lpips_loss_module is not None:
            try:
                # Check if LPIPS is on meta device
                lpips_params = list(self._lpips_loss_module.parameters())
                if len(lpips_params) > 0 and lpips_params[0].device.type != 'meta':
                    self._lpips_loss_module = self._lpips_loss_module.cpu()
            except (RuntimeError, NotImplementedError):
                # Skip if moving fails (LPIPS only needed for training)
                pass
        return result
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return self.action_decoder.dtype


# Register Tokenizer
AutoConfig.register("gr00t_tokenizer", GR00T_Tokenizer_Config)
AutoModel.register(GR00T_Tokenizer_Config, GR00T_Tokenizer)
