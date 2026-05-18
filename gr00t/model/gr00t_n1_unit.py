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
from typing import Tuple

import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .action_head.flow_matching_action_head_unit import (
    FlowmatchingActionHeadUniT,
    FlowmatchingActionHeadUniTConfig,
)
from .backbone import EagleBackbone, EagleBackboneUniT
from .backbone.eagle_backbone_unit import qwen_vl_visual_module
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import os
from pathlib import Path

BACKBONE_FEATURE_KEY = "backbone_features"
BACKBONE_SEQ_KEYS = frozenset({"backbone_features", "backbone_attention_mask"})
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


def load_gr00t_tokenizer(tokenizer_path: str, dinov2_path_override: str = None):
    """Load a frozen GR00T_Tokenizer from a checkpoint directory.

    Args:
        tokenizer_path: HuggingFace-style checkpoint dir containing the tokenizer.
        dinov2_path_override: Optional override of the DINOv2 weight path (deployment-only).
    """
    print(f"Loading GR00T tokenizer from {tokenizer_path}")
    if dinov2_path_override is not None:
        print(f"  - dinov2_path_override: {dinov2_path_override}")

    from .gr00t_n1_tokenizer_unit_inference import GR00T_Tokenizer

    try:
        tokenizer = GR00T_Tokenizer.from_pretrained(
            tokenizer_path,
            dinov2_path_override=dinov2_path_override,
            tune_vision_model=False,
            tune_vision_m_former=False,
            tune_bridge_projector=False,
            tune_action_encoder=False,
            tune_fusion=False,
            tune_vq=False,
            tune_vision_decoder=False,
            tune_action_decoder_projector=False,
            tune_action_decoder_diffusion=False,
        )

        # Older tokenizer checkpoints stored unified_embodiment_id only in config; older from_pretrained
        # paths did not propagate it into action_branch / action_decoder. Re-inject it so per-embodiment
        # parameter sharing keeps working for those checkpoints.
        config_unified_id = getattr(tokenizer.config, 'unified_embodiment_id', None)
        if config_unified_id is not None:
            if getattr(tokenizer.action_branch, 'unified_embodiment_id', None) is None:
                tokenizer.action_branch.unified_embodiment_id = config_unified_id
            if getattr(tokenizer.action_decoder, 'unified_embodiment_id', None) is None:
                tokenizer.action_decoder.unified_embodiment_id = config_unified_id
            if getattr(tokenizer, 'unified_embodiment_id', None) is None:
                tokenizer.unified_embodiment_id = config_unified_id

        print("Loaded GR00T tokenizer.")
        return tokenizer

    except Exception as e:
        raise RuntimeError(f"Failed to load GR00T tokenizer: {e}")

def compute_global_mean_std_patch(feat: torch.Tensor):
    """Per-patch mean/std over the global (cross-rank) batch.

    feat: [B, P, D] -> (global_mean, global_std), each [P, D].
    Distributed reduction sums local count/sum/sum_sq across ranks before computing statistics,
    so the result is identical regardless of world size.
    """
    x = feat.float()
    B, P, D = x.shape
    device = x.device

    distributed = dist.is_available() and dist.is_initialized()

    local_count = torch.tensor([B], device=device, dtype=torch.float32)
    local_sum = x.sum(dim=0)               # [P, D]
    local_sum_sq = (x * x).sum(dim=0)      # [P, D]

    if distributed:
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_sum_sq, op=dist.ReduceOp.SUM)

    global_mean = local_sum / local_count
    global_var = local_sum_sq / local_count - global_mean ** 2
    global_std = torch.sqrt(global_var)

    return global_mean.to(feat.device).detach(), global_std.to(feat.device).detach()


@dataclass
class GR00T_N1_5_UniT_Config(PretrainedConfig):
    model_type = "gr00t_n1_5_unit"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    unit_cfg: dict = field(init=False, metadata={"help": "Bridge configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class GR00T_N1_5_UniT(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_UniT_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1_5_UniT_Config,
        local_model_path: str,
        tokenizer_len: int=None,
        bridge_type: str="vision_lang_obs",
        compute_bridge_loss: bool=False,
        select_layer: int=None,
        bridge_loss_type: str="ce",
        use_image_type_embedding: bool=False,
        use_vl_mask: bool=False,
        use_correct_attn_mask: bool=False,  # Convert HF-style mask to SDPA-style
        action_only_one_obs: bool=False,

        noise_tau: float=0,
        omit_image_type_embedding_for_goal: bool=False,
        reweight_noise: bool=False,

        groot_tokenizer_path: str=None,
        action_loss_weight: float=1.0,  # Weight for action loss (0.0 to disable)
        bridge_loss_weight: float=0.1,  # Global scale on bridge_loss; total = (action_loss_weight*action_loss + bridge_loss_weight*bridge_loss) / 2
        unified_embodiment_id: int=None,  # If set, all samples use this embodiment ID
        detach_vl_for_action: bool=False,  # If True, detach VL features before passing to action head
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        if select_layer is not None:
            config.backbone_cfg['select_layer'] = select_layer

        config.action_head_cfg['use_vl_mask'] = use_vl_mask
        config.action_head_cfg['use_correct_attn_mask'] = use_correct_attn_mask

        super().__init__(config)
        self.local_model_path = local_model_path

        # tokenizer_len = Eagle backbone vocab size; for Qwen/Eagle this is fixed at 151729.
        if tokenizer_len is None:
            raise ValueError("tokenizer_len (Eagle vocab size) must be provided as parameter")
        
        print(f"[INFO] tokenizer_len (Eagle vocab size) for backbone: {tokenizer_len}")
        # num_bridge_tokens is the authoritative slice/registration size; the
        # backbone needs it to align its embedding-grad hook with the same
        # range of token ids that the data transform appended via get_bridge_str.
        if 'num_bridge_tokens' not in config.unit_cfg:
            raise ValueError("config.unit_cfg must contain 'num_bridge_tokens'")
        self.backbone = EagleBackboneUniT(
            **config.backbone_cfg,
            tokenizer_len=tokenizer_len,
            num_bridge_tokens=config.unit_cfg['num_bridge_tokens'],
        )
        action_head_cfg = FlowmatchingActionHeadUniTConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHeadUniT(action_head_cfg)

        # When set, all samples share a single embodiment_id inside the action head,
        # tying per-embodiment parameters together. Required for cross-embodiment co-training.
        if unified_embodiment_id is not None:
            print(f"Unified embodiment ID: {unified_embodiment_id}")
            self.action_head.unified_embodiment_id = unified_embodiment_id
        self.unified_embodiment_id = unified_embodiment_id

        print(f"Use bridge: {self.config.unit_cfg['use_bridge']}")
        self.use_bridge = self.config.unit_cfg['use_bridge']
        self.groot_tokenizer_path = groot_tokenizer_path

        # bridge_type must be set before _init_bridge_modules_tokenizer_mode (used inside).
        self.bridge_type = bridge_type
        self.compute_bridge_loss = compute_bridge_loss
        self.bridge_loss_type = bridge_loss_type
        self.action_loss_weight = action_loss_weight
        self.bridge_loss_weight = bridge_loss_weight
        self.detach_vl_for_action = detach_vl_for_action
        if detach_vl_for_action:
            print("VL features will be detached before action head (action loss will not update VLM via VL).")

        if self.config.unit_cfg['use_bridge']:
            self._init_bridge_modules_tokenizer_mode()

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

        # CE loss is the only supported bridge loss in tokenizer mode.
        if bridge_loss_type not in ["ce", "cross_entropy"]:
            print(f"WARNING: bridge_loss_type={bridge_loss_type} is ignored; CE loss is always used")

        self.use_image_type_embedding = use_image_type_embedding
        self.omit_image_type_embedding_for_goal = omit_image_type_embedding_for_goal
        if use_image_type_embedding:
            self.image_type_embedding = nn.Embedding(3, self.backbone.eagle_model.config.hidden_size)
            nn.init.normal_(self.image_type_embedding.weight, mean=0.0, std=0.02)
        
        self.action_only_one_obs = action_only_one_obs
        self.noise_tau = noise_tau
        self.reweight_noise = reweight_noise

    def validate_inputs(self, inputs):
        # NOTE: ideally enforced inside backbone/action_head themselves; kept here to avoid breaking
        # the existing public API.
        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def _init_bridge_modules_tokenizer_mode(self):
        """Initialize bridge modules using a pretrained GR00T_Tokenizer for goal encoding."""
        print("Using GR00T tokenizer for goal encoding")
        print(f"Bridge type: {self.bridge_type}")

        self._needs_obs_embeds = self.bridge_type in ["vision_lang_obs", "vision_lang_obs_e2e"]
        self._needs_vision_lang_features = self.bridge_type in ["vision_lang", "vision_lang_obs", "vision_lang_obs_e2e"]

        self.bridge_vision_model = None
        if self._needs_obs_embeds:
            qwen_vl_visual_module(self.backbone.eagle_model)

        # Tokenizer loading is deferred to from_pretrained to avoid loading external checkpoints
        # under HuggingFace's meta-device init context.
        if self.groot_tokenizer_path is None:
            raise ValueError("groot_tokenizer_path must be provided")
        self.groot_tokenizer = None

        hidden_size = self.backbone.eagle_model.config.hidden_size  # 2048 for GR00T-N1.5
        tokenizer_cfg = self.config.unit_cfg.get('tokenizer_cfg', {})

        # num_codebooks / codebook_size mirror the loaded GR00T_Tokenizer; kept in config for compatibility.
        self.num_codebooks = tokenizer_cfg.get('num_codebooks', 2)
        codebook_size = tokenizer_cfg.get('codebook_size', 128)

        if 'num_bridge_tokens' in self.config.unit_cfg:
            self.num_bridge_tokens = self.config.unit_cfg['num_bridge_tokens']
        else:
            raise ValueError("num_bridge_tokens must be specified in unit_cfg")
        print(f"num_bridge_tokens={self.num_bridge_tokens}, num_codebooks={self.num_codebooks}")

        # One independent CE head per RVQ codebook layer; predicts goal token indices
        # from bridge hidden states. Supervised by CE loss only (no soft embedding output).
        self.bridge_ce_predictors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, codebook_size)
            )
            for _ in range(self.num_codebooks)
        ])

        # Per-codebook CE loss weight; default schedule downweights deeper RVQ residuals.
        default_weights = [1.0 / (i + 1) for i in range(self.num_codebooks)]  # [1.0, 0.5, 0.33, ...]
        self.ce_loss_weights = tokenizer_cfg.get('ce_loss_weights', default_weights)
        assert len(self.ce_loss_weights) == self.num_codebooks, \
            f"ce_loss_weights length ({len(self.ce_loss_weights)}) must match num_codebooks ({self.num_codebooks})"
        self.label_smoothing = tokenizer_cfg.get('label_smoothing', 0.0)
        print(f"Tokenizer CE loss config: weights={self.ce_loss_weights}, label_smoothing={self.label_smoothing}")

    def forward(
        self,
        inputs: dict,
        action_mode: bool=False
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        cached_image_embeds = backbone_outputs.get("cached_image_embeds")
        if "cached_image_embeds" in backbone_outputs:
            del backbone_outputs["cached_image_embeds"]
        batch_size = inputs['state'].shape[0]

        # ImageNet-normalized obs/goal frames; shape [B, V*T, C, H, W]. Required for tokenizer GT.
        if "imagenet_obs_images" in inputs:
            imagenet_obs_images = inputs["imagenet_obs_images"]
            imagenet_goal_images = inputs.get("imagenet_goal_images", None)
        else:
            imagenet_obs_images = None
            imagenet_goal_images = None

        output_dict = {}
        if self.use_bridge:
            # Layout of backbone_outputs along the token axis: [..VL tokens.., ..bridge tokens..]
            # bridge tokens are appended in `EagleBackboneUniT` and always sit at the tail.
            #
            # Three knobs that are decoupled inside this block:
            #   - needs_vl: whether the action head consumes VL context (all bridge_types here do)
            #   - include_bridge_in_action_context: keep the bridge segment inside the VL context
            #     fed to the action head. True only for the "*_e2e" variant.
            #   - needs_obs_embeds: prepend Qwen .visual patch embeds to the action context.
            nt = self.num_bridge_tokens
            bridge_segment_features = backbone_outputs["backbone_features"][:, -nt:]
            bridge_segment_mask = backbone_outputs["backbone_attention_mask"][:, -nt:]

            include_bridge_in_action_context = self.bridge_type == "vision_lang_obs_e2e"
            if include_bridge_in_action_context:
                vl_features = backbone_outputs["backbone_features"]
                vl_attention_mask = backbone_outputs["backbone_attention_mask"]
            else:
                vl_features = backbone_outputs["backbone_features"][:, :-nt]
                vl_attention_mask = backbone_outputs["backbone_attention_mask"][:, :-nt]

            # CE prediction always runs on the bridge segment, regardless of bridge_type.
            logits_list = self.compute_logits_from_bridge_features(bridge_segment_features)

            # obs_embeds: Qwen .visual output captured once inside EagleBackboneUniT (cached_image_embeds).
            obs_embeds = None
            if self._needs_obs_embeds:
                if cached_image_embeds is None:
                    raise RuntimeError(
                        "Obs image embeds missing: expected backbone outputs['cached_image_embeds'] "
                        "(EagleBackboneUniT visual forward hook). Ensure eagle batch includes pixel_values "
                        "and backbone uses a supported Qwen2.5-VL-style .visual tower."
                    )
                obs_embeds = cached_image_embeds
                hidden_size = obs_embeds.shape[-1]
                if self.action_only_one_obs:
                    obs_embeds = obs_embeds.reshape(batch_size, -1, self.num_bridge_tokens, hidden_size)
                    obs_embeds = obs_embeds[:, -1]
                else:
                    obs_embeds = obs_embeds.reshape(batch_size, -1, hidden_size)

            if self.compute_bridge_loss:
                if imagenet_goal_images is None:
                    raise RuntimeError("imagenet_goal_images is required for bridge loss computation")
                if imagenet_obs_images is None:
                    raise RuntimeError("imagenet_obs_images is required for bridge loss computation")

                tokenizer_gt = self.compute_tokenizer_ground_truth(inputs)
                gt_indices = tokenizer_gt['gt_indices']  # [B, num_tokens*num_codebooks]

                loss_outputs = self.compute_tokenizer_ce_loss(logits_list, gt_indices)
                output_dict['bridge_loss'] = loss_outputs['bridge_loss']
                for i in range(self.num_codebooks):
                    output_dict[f'ce_loss_layer{i+1}'] = loss_outputs[f'ce_loss_layer{i+1}']

            # Build the cross-attention context fed to the action head.
            # detach_vl_for_action severs the gradient path from action loss into the VLM tokens
            # while leaving the CE-loss path through the bridge tokens intact.
            if self.detach_vl_for_action:
                vl_features = vl_features.detach()

            if self.bridge_type == "vision_lang":
                action_context_features = vl_features
                action_context_mask = vl_attention_mask
            elif self.bridge_type in ("vision_lang_obs", "vision_lang_obs_e2e"):
                # VL features already carry VLM positional encoding; image_type_embedding marks
                # obs_embeds apart in the cross-attention context.
                assert obs_embeds is not None, (
                    f"bridge_type={self.bridge_type} requires obs_embeds; "
                    "this should have been produced from cached_image_embeds above."
                )
                if self.use_image_type_embedding:
                    obs_embeds = obs_embeds + self.image_type_embedding.weight[1]
                obs_attention_mask = torch.ones(obs_embeds.shape[:-1], device=obs_embeds.device)
                action_context_features = torch.cat([vl_features, obs_embeds], dim=1)
                action_context_mask = torch.cat([vl_attention_mask, obs_attention_mask], dim=1)
            else:
                raise NotImplementedError(f"Invalid bridge_type: {self.bridge_type}")

            bridge_outputs = BatchFeature(
                data={
                    "backbone_features": action_context_features,
                    "backbone_attention_mask": action_context_mask,
                }
            )
        else:
            bridge_outputs = backbone_outputs

        if not action_mode:
            action_head_outputs = self.action_head(bridge_outputs, action_inputs)
            self.validate_data(action_head_outputs, backbone_outputs, is_training=True)

            output_dict['action_loss'] = action_head_outputs['loss']

            # action_loss_weight=0 disables the action gradient path.
            # bridge_loss is globally scaled by bridge_loss_weight (default 0.1).
            action_loss = self.action_loss_weight * action_head_outputs['loss']
            if 'bridge_loss' in output_dict:
                loss = (action_loss + self.bridge_loss_weight * output_dict['bridge_loss']) / 2
            else:
                loss = action_loss
            output_dict['loss'] = loss

            return BatchFeature(data=output_dict)

        else:
            action_head_outputs = self.action_head.get_action(bridge_outputs, action_inputs)
            self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
            action_head_outputs.update(output_dict)
            return action_head_outputs


    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        return self.forward(inputs=inputs, action_mode=True)
    
    def state_dict(self, *args, **kwargs):
        """Exclude the frozen groot_tokenizer params from the saved state_dict.

        The tokenizer is loaded from an external checkpoint and never updated. Excluding
        it also avoids shared-tensor issues raised by the LPIPS loss inside the tokenizer.
        """
        state_dict = super().state_dict(*args, **kwargs)
        if hasattr(self, 'groot_tokenizer') and self.groot_tokenizer is not None:
            keys_to_remove = [k for k in state_dict.keys() if k.startswith('groot_tokenizer.')]
            for key in keys_to_remove:
                del state_dict[key]
            if len(keys_to_remove) > 0:
                print(f"[INFO] Excluded {len(keys_to_remove)} GR00T tokenizer parameters from state_dict")
        return state_dict

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            if not isinstance(x, torch.Tensor):
                return x
            # Cast to the action head's compute dtype only for floating tensors;
            # integer tensors (token ids, indices) keep their original dtype.
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, resume_pretrained_option: str="all", **kwargs):
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        tune_visual = kwargs.pop("tune_visual", model_config.backbone_cfg['tune_visual'])
        tune_llm = kwargs.pop("tune_llm", model_config.backbone_cfg['tune_llm'])
        tune_bridge_embedding = kwargs.pop("tune_bridge_embedding", model_config.backbone_cfg['tune_bridge_embedding'])
        tokenizer_len = kwargs.pop("tokenizer_len", None)
        tune_projector = kwargs.pop("tune_projector", model_config.action_head_cfg['tune_projector'])
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", model_config.action_head_cfg['tune_diffusion_model'])

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune backbone bridge embedding: {tune_bridge_embedding}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        try:
            bridge_type = kwargs.pop("bridge_type", model_config.unit_cfg.get('bridge_type', "vision_lang_obs"))
            compute_bridge_loss = kwargs.pop("compute_bridge_loss", model_config.unit_cfg.get('compute_bridge_loss', False))
            bridge_loss_type = kwargs.pop("bridge_loss_type", model_config.unit_cfg.get('bridge_loss_type', 'ce'))
            tune_all_llm_embedding = kwargs.pop("tune_all_llm_embedding", model_config.unit_cfg.get('tune_all_llm_embedding', False))
            use_image_type_embedding = kwargs.pop("use_image_type_embedding", model_config.unit_cfg.get('use_image_type_embedding', False))
            omit_image_type_embedding_for_goal = kwargs.pop("omit_image_type_embedding_for_goal", model_config.unit_cfg.get('omit_image_type_embedding_for_goal', False))
            action_only_one_obs = kwargs.pop("action_only_one_obs", model_config.unit_cfg.get('action_only_one_obs', False))
            noise_tau = kwargs.pop("noise_tau", model_config.unit_cfg.get('noise_tau', 0))
            reweight_noise = kwargs.pop("reweight_noise", model_config.unit_cfg.get('reweight_noise', False))
            groot_tokenizer_path = kwargs.pop("groot_tokenizer_path", model_config.unit_cfg.get('groot_tokenizer_path', None))
            action_loss_weight = kwargs.pop("action_loss_weight", model_config.unit_cfg.get('action_loss_weight', 1.0))
            bridge_loss_weight = kwargs.pop("bridge_loss_weight", model_config.unit_cfg.get('bridge_loss_weight', 0.1))
            unified_embodiment_id = kwargs.pop("unified_embodiment_id", model_config.unit_cfg.get('unified_embodiment_id', None))
            detach_vl_for_action = kwargs.pop("detach_vl_for_action", model_config.unit_cfg.get('detach_vl_for_action', False))
            # Optional override of the DINOv2 weight path inside the loaded GR00T tokenizer (deployment-time).
            dinov2_path_override = kwargs.pop("dinov2_path_override", None)
        except Exception as e:
            print(kwargs)
            raise e
        print(f"Bridge type: {bridge_type}")
        print(f"Compute bridge loss: {compute_bridge_loss}")
        print(f"Bridge loss type: {bridge_loss_type}")
        print(f"Tune all llm token embeddings: {tune_all_llm_embedding}")
        print(f"Use image type embeddings: {use_image_type_embedding}")
        print(f"Omit image type embeddings for goal images: {omit_image_type_embedding_for_goal}")
        print(f"Action head using only one obs: {action_only_one_obs}")
        print(f"Noise Tau: {noise_tau}")
        print(f"Reweight Noise: {reweight_noise}")
        print(f"GR00T tokenizer path: {groot_tokenizer_path}")
        print(f"Action loss weight: {action_loss_weight}")
        print(f"Bridge loss weight: {bridge_loss_weight}")
        print(f"Unified embodiment ID: {unified_embodiment_id}")
        print(f"Detach VL for action: {detach_vl_for_action}")
        if dinov2_path_override is not None:
            print(f"DINOv2 path override (for tokenizer): {dinov2_path_override}")

        select_layer = kwargs.pop("select_layer", model_config.backbone_cfg.get('select_layer', None))

        tune_bridge_visual = kwargs.pop("tune_bridge_visual", model_config.unit_cfg['tune_bridge_visual'])
        tune_image_type_embedding = kwargs.pop("tune_image_type_embedding", model_config.unit_cfg.get('tune_image_type_embedding', True))
        print(f"Tune bridge vision model: {tune_bridge_visual}")
        print(f"Tune image type embeddings: {tune_image_type_embedding}")

        use_vl_mask = kwargs.pop("use_vl_mask", model_config.action_head_cfg.get('use_vl_mask', True))
        print(f"Use VL mask: {use_vl_mask}")

        # use_correct_attn_mask: convert HF-style attention mask to SDPA-style inside action head
        use_correct_attn_mask = kwargs.pop(
            "use_correct_attn_mask", model_config.action_head_cfg.get('use_correct_attn_mask', True)
        )
        print(f"Use correct attn mask format: {use_correct_attn_mask}")

        # snapshot_download returns local cache path under ~/.cache/huggingface/hub/;
        # falls back to treating the argument as a local path if it is not a valid hub repo id.
        try:
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        customized_kwargs = {
            "tokenizer_len": tokenizer_len,
            "bridge_type": bridge_type,
            "compute_bridge_loss": compute_bridge_loss,
            "select_layer": select_layer,
            "bridge_loss_type": bridge_loss_type,
            "use_image_type_embedding": use_image_type_embedding,
            "use_vl_mask": use_vl_mask,
            "use_correct_attn_mask": use_correct_attn_mask,
            "action_only_one_obs": action_only_one_obs,
            "noise_tau": noise_tau,
            "omit_image_type_embedding_for_goal": omit_image_type_embedding_for_goal,
            "reweight_noise": reweight_noise,
            "groot_tokenizer_path": groot_tokenizer_path,
            "action_loss_weight": action_loss_weight,
            "bridge_loss_weight": bridge_loss_weight,
            "unified_embodiment_id": unified_embodiment_id,
            "detach_vl_for_action": detach_vl_for_action,
        }

        try:
            import os

            pretrained_model = super().from_pretrained(
                local_model_path, local_model_path=local_model_path,
                **customized_kwargs,
                **kwargs
            )

            if resume_pretrained_option != "all":
                config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
                scratch_model = cls(
                    config, local_model_path=None,
                    **customized_kwargs,
                )
                print(f"Resuming only {resume_pretrained_option} ...")
                if resume_pretrained_option == "backbone+bridge_projector":
                    pretrained_model.action_head = scratch_model.action_head
                elif resume_pretrained_option == "action_head":
                    pretrained_model.backbone = scratch_model.backbone
                    pretrained_model.bridge_projector = scratch_model.bridge_projector
                else:
                    raise NotImplementedError

        except Exception as e:
            print(f"Load pretrained model error: {e}")
            config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
            pretrained_model = cls(
                config, local_model_path=None,
                **customized_kwargs,
            )

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm, 
            tune_bridge_embedding=tune_bridge_embedding,
            tokenizer_len=tokenizer_len,
            tune_all_llm_embedding=tune_all_llm_embedding,
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )

        if pretrained_model.use_bridge:
            pretrained_model.set_trainable_parameters(
                tune_bridge_visual=tune_bridge_visual,
                tune_image_type_embedding=tune_image_type_embedding
            )

        pretrained_model.config.backbone_cfg['tune_visual'] = tune_visual
        pretrained_model.config.backbone_cfg['tune_llm'] = tune_llm
        pretrained_model.config.backbone_cfg['tune_bridge_embedding'] = tune_bridge_embedding
        pretrained_model.config.action_head_cfg['tune_projector'] = tune_projector
        pretrained_model.config.action_head_cfg['tune_diffusion_model'] = tune_diffusion_model
        pretrained_model.config.action_head_cfg['use_vl_mask'] = use_vl_mask
        pretrained_model.config.action_head_cfg['use_correct_attn_mask'] = use_correct_attn_mask
        pretrained_model.config.unit_cfg['tune_bridge_visual'] = tune_bridge_visual
        pretrained_model.config.unit_cfg['tokenizer_len'] = tokenizer_len
        pretrained_model.config.unit_cfg['bridge_type'] = bridge_type
        pretrained_model.config.unit_cfg['compute_bridge_loss'] = compute_bridge_loss
        pretrained_model.config.unit_cfg['bridge_loss_type'] = bridge_loss_type
        pretrained_model.config.backbone_cfg['tune_all_llm_embedding'] = tune_all_llm_embedding
        pretrained_model.config.unit_cfg['use_image_type_embedding'] = use_image_type_embedding
        pretrained_model.config.unit_cfg['action_only_one_obs'] = action_only_one_obs
        pretrained_model.config.unit_cfg['noise_tau'] = noise_tau
        pretrained_model.config.unit_cfg['reweight_noise'] = reweight_noise
        pretrained_model.config.unit_cfg['omit_image_type_embedding_for_goal'] = omit_image_type_embedding_for_goal
        pretrained_model.config.unit_cfg['tune_image_type_embedding'] = tune_image_type_embedding
        pretrained_model.config.unit_cfg['groot_tokenizer_path'] = groot_tokenizer_path
        pretrained_model.config.unit_cfg['unified_embodiment_id'] = unified_embodiment_id
        pretrained_model.config.unit_cfg['bridge_loss_weight'] = bridge_loss_weight

        # Load the tokenizer after from_pretrained finishes so it does not get materialized
        # under HuggingFace's meta-device init context. Prefer a checkpoint-local copy
        # ({checkpoint}/tokenizer) over the config-supplied path so saved runs are self-contained.
        if pretrained_model.groot_tokenizer is None:
            local_tokenizer_path = os.path.join(pretrained_model_name_or_path, "tokenizer")
            if os.path.exists(local_tokenizer_path):
                tokenizer_path_to_use = local_tokenizer_path
            else:
                tokenizer_path_to_use = pretrained_model.groot_tokenizer_path
            print(f"Using tokenizer at: {tokenizer_path_to_use}")

            pretrained_model.groot_tokenizer = load_gr00t_tokenizer(
                tokenizer_path_to_use,
                dinov2_path_override=dinov2_path_override
            )
            pretrained_model.groot_tokenizer.to(pretrained_model.device)
            pretrained_model.groot_tokenizer.requires_grad_(False)
            pretrained_model.groot_tokenizer.eval()

            print(
                f"GR00T tokenizer: num_codebooks={pretrained_model.groot_tokenizer.num_codebooks}, "
                f"num_bridge_tokens={pretrained_model.groot_tokenizer.num_bridge_tokens}, "
                f"codebook shapes={[tuple(w.shape) for w in pretrained_model.groot_tokenizer.codebook_weights]}"
            )

        return pretrained_model

    def set_trainable_parameters(self, tune_bridge_visual: bool, tune_image_type_embedding: bool):
        self.tune_bridge_visual = tune_bridge_visual
        self.tune_image_type_embedding = tune_image_type_embedding

        # Tokenizer comes from a frozen external checkpoint; never updated.
        if self.groot_tokenizer is not None:
            self.groot_tokenizer.requires_grad_(False)

        # CE predictors are the only bridge-side trainable head.
        self.bridge_ce_predictors.requires_grad_(True)
        print(f"[DEBUG] CE predictors ({self.num_codebooks} layers) trainable ({self.bridge_type})")

        if self.use_image_type_embedding:
            self.image_type_embedding.requires_grad_(self.tune_image_type_embedding)


    def _ensure_tokenizer_eval_mode(self):
        """Recursively force the frozen tokenizer (and all submodules) into eval().

        Required before every tokenizer forward pass: keeping it in train() would
        re-enable VQ dead-code restart and Dropout/BatchNorm training paths inside
        the tokenizer.
        """
        if self.groot_tokenizer is not None:
            def recursive_eval(module):
                module.eval()
                for child in module.children():
                    recursive_eval(child)
            recursive_eval(self.groot_tokenizer)

    def set_frozen_modules_to_eval_mode(self):
        """Re-apply eval() to frozen submodules.

        HuggingFace's Trainer calls model.train() at the start of every training_step,
        which would otherwise flip Dropout/BatchNorm in our frozen modules back on.
        """
        if self.training:
            self._ensure_tokenizer_eval_mode()
            if self.use_image_type_embedding and not self.tune_image_type_embedding:
                self.image_type_embedding.eval()


    def compute_tokenizer_ground_truth(self, inputs):
        """Run the frozen GR00T tokenizer once to obtain GT goal-token indices.

        Returns:
            gt_indices: [B, num_tokens*num_codebooks] - codebook indices concatenated over RVQ layers.
        """
        self._ensure_tokenizer_eval_mode()
        with torch.no_grad():
            # GR00T_Tokenizer expects 'imagenet_obs_images' / 'imagenet_goal_images' inside `inputs`.
            outputs = self.groot_tokenizer.forward(
                inputs=inputs,
                return_motion_token_ids_only=True
            )
            indices = outputs["indices"]
            return {'gt_indices': indices}
    
    def compute_logits_from_bridge_features(self, bridge_features):
        """Run each per-RVQ-layer CE head on the bridge tokens.

        Args:
            bridge_features: [B, num_bridge_tokens, hidden_size]
        Returns:
            logits_list[i]: [B, num_bridge_tokens, codebook_size] for the i-th RVQ layer.
        """
        return [predictor(bridge_features) for predictor in self.bridge_ce_predictors]

    def compute_tokenizer_ce_loss(self, logits_list, gt_indices):
        """Weighted-sum cross-entropy across RVQ layers.

        Returns:
            bridge_loss: scalar, sum_i ce_loss_weights[i] * CE(logits_list[i], gt_indices[..., i])
            ce_loss_layer{i+1}: per-layer CE for logging.
        """
        total_loss = 0
        layer_losses = {}
        for i in range(self.num_codebooks):
            gt_indices_i = gt_indices[..., i]                        # [B, Q]
            logits_i = logits_list[i]                                # [B, Q, codebook_size]
            logits_flat = logits_i.reshape(-1, logits_i.size(-1))    # [B*Q, codebook_size]
            targets_flat = gt_indices_i.reshape(-1)                  # [B*Q]
            ce_loss_i = F.cross_entropy(logits_flat, targets_flat, label_smoothing=self.label_smoothing)
            layer_losses[f'ce_loss_layer{i+1}'] = ce_loss_i
            total_loss = total_loss + ce_loss_i * self.ce_loss_weights[i]
        return {'bridge_loss': total_loss, **layer_losses}

AutoConfig.register("gr00t_n1_5_unit", GR00T_N1_5_UniT_Config)
AutoModel.register(GR00T_N1_5_UniT_Config, GR00T_N1_5_UniT)
