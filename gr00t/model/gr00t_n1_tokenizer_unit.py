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
from typing import Optional

import torch
import torch.nn.functional as F
import tree
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature


def denormalize_imagenet(images: torch.Tensor) -> torch.Tensor:
    """Denormalize ImageNet-normalized images to [0, 1]. Expects (B, C, H, W)."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    images = images * std + mean
    return torch.clamp(images, 0, 1)


class DINOv2Wrapper(nn.Module):
    """
    Wrapper for DINOv2 model to match the interface expected by VisionBranchEncoder
    
    DINOv2 outputs include a CLS token as the first token, which we need to remove
    to get only patch features for compatibility with the rest of the pipeline.
    
    Supports selecting features from specific layers (e.g., last layer or second-to-last layer)
    
    Input: pixel_values (B, C, H, W)
    Output: patch_features (B, N_patches, hidden_size) - without CLS token
    """
    
    def __init__(self, dinov2_model, layer_index=-1):
        """
        Args:
            dinov2_model: The DINOv2 model instance
            layer_index: Which layer to extract features from
                        -1 = last layer (default)
                        -2 = second-to-last layer
                        0 to N-1 = specific layer index
        """
        super().__init__()
        self.model = dinov2_model
        self.config = dinov2_model.config
        self.layer_index = layer_index
        
        print(f"DINOv2Wrapper: Using layer {layer_index} for features")
    
    def forward(self, pixel_values, **kwargs):
        """
        Forward pass through DINOv2 with CLS token removal
        
        Args:
            pixel_values: (B, C, H, W) input images
            **kwargs: Additional arguments passed to DINOv2
        
        Returns:
            patch_features: (B, N_patches, hidden_size) without CLS token
        """
        # Force output_hidden_states if we need intermediate layers
        if self.layer_index != -1:
            kwargs['output_hidden_states'] = True
        
        outputs = self.model(pixel_values, **kwargs)
        
        # Select the appropriate layer
        if self.layer_index == -1:
            # Use last layer (default behavior)
            hidden_state = outputs.last_hidden_state
        else:
            # Use specific layer from hidden_states tuple
            # hidden_states: tuple of (B, 1 + N_patches, hidden_size) for each layer
            hidden_state = outputs.hidden_states[self.layer_index]
        
        # DINOv2 output shape: (B, 1 + N_patches, hidden_size)
        # First token is CLS token, rest are patch features
        # Remove CLS token: [:, 1:, :]
        patch_features = hidden_state[:, 1:, :]
        return patch_features
    
    def __getattr__(self, name):
        """Forward attribute access to the wrapped model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
    
    def requires_grad_(self, requires_grad=True):
        """Forward requires_grad_ to the wrapped model"""
        self.model.requires_grad_(requires_grad)
        return self


# ==================== Tokenizer Config ====================
@dataclass
class GR00T_Tokenizer_Config(PretrainedConfig):
    """
    Configuration for GR00T Tokenizer (VQ-based reconstruction model)
    
    Architecture:
    - Vision Branch: vision_model → M-Former → visual query tokens
    - Action Branch: ActionEncoder → action query tokens
    - Fusion: Visual-Action Fusion → unit tokens (pv/pa-routed latent queries)
    - VQ: Vector Quantization → quantized tokens
    - Decoders: Vision Decoder + Action Decoder
    """
    model_type = "gr00t_tokenizer"
    
    # ========== Backbone Configuration (for vision_model) ==========
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration (vision model only, no LLM)"})
    
    # ========== Encoder Configurations ==========
    vision_branch_cfg: dict = field(init=False, metadata={"help": "Vision Branch Encoder configuration"})
    action_encoder_cfg: dict = field(init=False, metadata={"help": "Action Encoder configuration"})
    
    # ========== Fusion & VQ Configurations ==========
    fusion_cfg: dict = field(default_factory=dict, metadata={"help": "Visual-Action Fusion configuration"})
    vq_cfg: dict = field(init=False, metadata={"help": "Vector Quantizer configuration"})
    
    # ========== Decoder Configurations ==========
    vision_decoder_cfg: dict = field(init=False, metadata={"help": "Vision Decoder (LatentMotionDecoder) configuration"})
    action_decoder_cfg: dict = field(init=False, metadata={"help": "Action Decoder (FlowMatchingActionHeadBridge) configuration"})
    
    # ========== Basic Parameters ==========
    action_horizon: int = field(init=False, metadata={"help": "Action horizon"})
    action_dim: int = field(init=False, metadata={"help": "Action dimension"})
    state_dim: int = field(init=False, metadata={"help": "State dimension"})
    query_num: int = field(default=8, metadata={"help": "Number of query tokens"})
    hidden_size: int = field(default=768, metadata={"help": "Hidden size"})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype"})
    
    # ========== Training Configuration ==========
    reconstruction_loss_weights: dict = field(default_factory=lambda: {
        'vision': 1.0,
        'action': 1.0,
        'vq_commitment': 0.25,
        'lpips': 1.0,
    }, metadata={
        "help": "Loss weights. Optional 'unit_fusion_distill' (>0): multi-scenario only — "
                "MSE( unit_tokens[vision_only], unit_tokens[both].detach() ) + "
                "MSE( unit_tokens[action_only], unit_tokens[both].detach() ); training only."
    })
    use_lpips_loss: bool = field(default=False, metadata={"help": "Whether to use LPIPS perceptual loss"})
    
    # ========== Modality Presence Configuration ==========
    default_pv: int = field(default=1, metadata={"help": "Default visual presence (0: no visual, 1: has visual)"})
    default_pa: int = field(default=1, metadata={"help": "Default action presence (0: no action, 1: has action)"})
    
    # ========== Multi-Scenario Training Configuration ==========
    use_multi_scenario_training: bool = field(default=False, metadata={"help": "Enable multi-scenario training with 3x batch expansion"})

    # ========== Trainable Flags (for from_pretrained) ==========
    tune_vision_model: bool = field(default=True, metadata={"help": "Whether to tune vision model"})
    tune_vision_m_former: bool = field(default=True, metadata={"help": "Whether to tune vision M-Former"})
    tune_action_encoder: bool = field(default=True, metadata={"help": "Whether to tune action encoder"})
    tune_fusion: bool = field(default=True, metadata={"help": "Whether to tune fusion module"})
    tune_vq: bool = field(default=True, metadata={"help": "Whether to tune VQ module"})
    tune_vision_decoder: bool = field(default=True, metadata={"help": "Whether to tune vision decoder"})
    tune_action_decoder_projector: bool = field(default=True, metadata={"help": "Whether to tune action decoder projector"})
    tune_action_decoder_diffusion: bool = field(default=True, metadata={"help": "Whether to tune action decoder diffusion model"})

    # ========== Unified Embodiment Configuration ==========
    unified_embodiment_id: int = field(default=None, metadata={
        "help": "If set, all samples use this embodiment ID for action encoder/decoder (share parameters)"
    })
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

class GR00T_Tokenizer(PreTrainedModel):
    """
    GR00T Tokenizer: VQ-based Multimodal Reconstruction Model
    
    Architecture:
    1. Vision Branch: vision_model → M-Former → visual query tokens
    2. Action Branch: ActionEncoder → action query tokens
    3. Fusion: Visual-Action Fusion (M-Former based) → unit tokens (routed per pv/pa)
    4. VQ: Residual Vector Quantizer → quantized tokens
    5. Decoders:
       - Vision Decoder: LatentMotionDecoder → reconstructed vision
       - Action Decoder: FlowMatchingActionHeadBridge → reconstructed action
    
    Training: reconstruction + VQ commitment; optional ``unit_fusion_distill`` (multi-scenario only)
        aligns vision_only / action_only ``unit_tokens`` rows to the both-row slice (teacher detached).
    Inference: Can be used for action prediction or vision synthesis
    """
    supports_gradient_checkpointing = True
    config_class = GR00T_Tokenizer_Config
    
    def __init__(
        self,
        config: GR00T_Tokenizer_Config,
        local_model_path: str = None,
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
        
        # ========== 1. Load DINOv2 Vision Model ==========
        from transformers import Dinov2Model

        dinov2_path = config.backbone_cfg.get('dinov2_path', 'facebook/dinov2-large')
        dinov2_layer_index = config.backbone_cfg.get('dinov2_layer_index', -2)
        dinov2_model = Dinov2Model.from_pretrained(dinov2_path)
        # Wrap DINOv2 to remove CLS token (first token in output)
        vision_model = DINOv2Wrapper(dinov2_model, layer_index=dinov2_layer_index)

        tune_visual = config.backbone_cfg.get('tune_visual', False)
        if tune_visual:
            vision_model.requires_grad_(True)
        else:
            vision_model.requires_grad_(False)
        
        self.tune_visual = tune_visual
        # Vision reconstruction runs in forward() when obs_input and goal_input are present;
        # no separate compute_bridge_loss (from_pretrained still pops it for checkpoint/call compat).
        # ========== 2. Vision Branch Encoder ==========
        from gr00t.model.tokenizer.vision_branch_encoder import (
            VisionBranchEncoder, VisionBranchEncoderConfig
        )
        vision_branch_cfg = VisionBranchEncoderConfig(**config.vision_branch_cfg)
        self.vision_branch = VisionBranchEncoder(
            config=vision_branch_cfg,
            vision_model=vision_model
        )
        
        # ========== 3. Action Branch Encoder ==========
        from gr00t.model.tokenizer.action_branch_encoder import (
            ActionEncoder, ActionEncoderConfig
        )
        action_encoder_cfg = ActionEncoderConfig(**config.action_encoder_cfg)
        self.action_branch = ActionEncoder(action_encoder_cfg)
        
        # ========== 4. Visual-Action Fusion ==========
        from gr00t.model.tokenizer.visual_action_fusion import QFormerVisualActionFusion
        self.fusion = QFormerVisualActionFusion(
            hidden_size=config.hidden_size,
            query_num=config.query_num,
            fusion_config=config.fusion_cfg
        )
        
        # ========== 5. Vector Quantizer ==========
        # Support different VQ types via config.vq_cfg['vq_type']
        # Available types:
        #   - "default" or None: Standard Residual VQ (original implementation)
        #   - "ema": EMA update + rotation trick (from vector-quantize-pytorch)
        #            Supports kmeans_init and dead code restart
        
        from gr00t.model.tokenizer.vector_quantizer import (
            ResidualVectorQuantizer, ResidualVectorQuantizerConfig, ResidualVQFromLib, ResidualFSQFromLib
        )
        
        vq_cfg = ResidualVectorQuantizerConfig(**config.vq_cfg)
        vq_type = config.vq_cfg.get('vq_type', 'default')
        
        if vq_type == 'ema':
            # Use vector-quantize-pytorch's ResidualVQ with EMA + rotation trick
            self.vq = ResidualVQFromLib(vq_cfg)
            print(f"✓ Using ResidualVQFromLib (ema mode)")
        elif vq_type == 'fsq':
            # Use vector-quantize-pytorch's ResidualFSQ (Finite Scalar Quantization)
            self.vq = ResidualFSQFromLib(vq_cfg)
            print(f"✓ Using ResidualFSQFromLib (fsq mode)")
        else:
            # Use original ResidualVectorQuantizer (default)
            self.vq = ResidualVectorQuantizer(vq_cfg)
            print(f"✓ Using ResidualVectorQuantizer (standard RVQ)")
        
        # ========== 6. Vision Decoder ==========
        from gr00t.model.tokenizer.latent_motion_decoder import LatentMotionDecoder
        from transformers.models.vit.modeling_vit import ViTConfig
        vision_decoder_vit_cfg = ViTConfig(**config.vision_decoder_cfg)
        self.vision_decoder = LatentMotionDecoder(vision_decoder_vit_cfg)
        
        # ========== 7. Action Decoder (ResNet-based, direct reconstruction) ==========
        from gr00t.model.tokenizer.action_branch_decoder import (
            ActionDecoder, ActionDecoderConfig
        )
        action_decoder_cfg = ActionDecoderConfig(**config.action_decoder_cfg)
        self.action_decoder = ActionDecoder(action_decoder_cfg)
        
        # ========== Unified Embodiment ID (Dynamic Injection) ==========
        # Following gr00t_n1_bridge.py pattern: inject unified_embodiment_id to encoder/decoder
        # This allows all samples to share the same embodiment parameters
        _unified_embodiment_id = unified_embodiment_id if unified_embodiment_id is not None else config.unified_embodiment_id
        if _unified_embodiment_id is not None:
            print(f"✓ Unified embodiment ID: {_unified_embodiment_id} (action encoder/decoder will share parameters)")
            self.action_branch.unified_embodiment_id = _unified_embodiment_id
            self.action_decoder.unified_embodiment_id = _unified_embodiment_id
        self.unified_embodiment_id = _unified_embodiment_id
        
        # ========== 8. Bridge Projector ==========
        # Down-sample unit tokens to VQ embedding dim, then up-sample quantized
        # tokens back to hidden_size for downstream concat with obs features.
        # Sequential wrapper kept for state_dict key compatibility with older ckpts.
        self.pos_embed = nn.Parameter(torch.zeros(config.query_num, config.hidden_size))
        self.vq_down_resampler = nn.Sequential(
            nn.Linear(config.hidden_size, config.vq_cfg['e_dim'])
        )
        self.bridge_projector = nn.Sequential(
            nn.Linear(config.vq_cfg['e_dim'], config.hidden_size)
        )
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        
        # ========== Configuration ==========
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.query_num = config.query_num
        self.hidden_size = config.hidden_size
        self.compute_dtype = config.compute_dtype
        self.loss_weights = config.reconstruction_loss_weights
        self.use_lpips_loss = config.use_lpips_loss if config.use_lpips_loss is not None else True
        
        # DINO mode: use cosine similarity loss instead of MSE+LPIPS for vision reconstruction
        # Read from vision_decoder_cfg (can be overridden after model loading)
        self.is_dino_mode = config.vision_decoder_cfg.get('is_io_hidden_states', False)

        # Initialize LPIPS loss if needed
        # NOTE: LPIPS must be excluded from state_dict to avoid shared tensor issues
        # We use a special storage mechanism that PyTorch won't track
        if self.use_lpips_loss:
            try:
                import lpips
                # Create LPIPS model
                lpips_model = lpips.LPIPS(net='vgg', verbose=False).requires_grad_(False).eval()
                # Store in a way that won't be tracked by PyTorch's state_dict
                # Using object.__setattr__ to bypass nn.Module's __setattr__
                object.__setattr__(self, '_lpips_loss_module', lpips_model)
                print("✓ LPIPS loss initialized (excluded from checkpoints)")
            except ImportError:
                print("⚠️  LPIPS not available. Install with: pip install lpips")
                self.use_lpips_loss = False
                object.__setattr__(self, '_lpips_loss_module', None)
        else:
            object.__setattr__(self, '_lpips_loss_module', None)
        
        # Modality presence defaults (can be overridden in forward)
        self.default_pv = config.default_pv if config.default_pv is not None else 1  # 1: visual present by default
        self.default_pa = config.default_pa if config.default_pa is not None else 1  # 1: action present by default

    @property
    def lpips_loss(self):
        """
        Property to access LPIPS loss module (excluded from checkpoints)
        
        Lazy loading: Ensures LPIPS is on the correct device when accessed.
        If LPIPS is on meta device (from fast loading), reinitialize it on the target device.
        """
        if self._lpips_loss_module is None:
            return None
        
        # Check if LPIPS is on the correct device (should match model device)
        try:
            lpips_params = list(self._lpips_loss_module.parameters())
            if len(lpips_params) > 0:
                lpips_device = lpips_params[0].device
                model_device = self.device
                
                # If LPIPS is on meta device, reinitialize it on the target device
                if lpips_device.type == 'meta':
                    print(f"⚠️  LPIPS is on meta device, reinitializing on {model_device}")
                    try:
                        import lpips
                        # Reinitialize LPIPS directly on target device
                        lpips_model = lpips.LPIPS(net='vgg', verbose=False).requires_grad_(False).eval()
                        lpips_model = lpips_model.to(model_device)
                        object.__setattr__(self, '_lpips_loss_module', lpips_model)
                        print(f"✓ LPIPS reinitialized on {model_device}")
                    except Exception as reinit_e:
                        print(f"⚠️  Failed to reinitialize LPIPS: {reinit_e}")
                        object.__setattr__(self, '_lpips_loss_module', None)
                        return None
                # If devices don't match but not meta, move LPIPS to model device
                elif lpips_device != model_device:
                    print(f"Moving LPIPS from {lpips_device} to {model_device}")
                    self._lpips_loss_module = self._lpips_loss_module.to(model_device)
        except Exception as e:
            print(f"Warning: Failed to ensure LPIPS device: {e}")
            # Return None if LPIPS can't be used
            return None
        
        return self._lpips_loss_module
    
    def state_dict(self, *args, **kwargs):
        """Override state_dict to exclude LPIPS loss module"""
        state_dict = super().state_dict(*args, **kwargs)
        # Remove all LPIPS loss related keys
        keys_to_remove = [k for k in state_dict.keys() if '_lpips_loss_module' in k]
        for key in keys_to_remove:
            del state_dict[key]
        return state_dict
    
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Override to handle missing LPIPS loss keys gracefully"""
        # LPIPS loss will be re-initialized in __init__, so missing keys are expected
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
    
    def _split_3scenario_batch(self, tensor_3b, original_batch_size):
        """
        Split 3x expanded batch into 3 individual scenario tensors
        
        Args:
            tensor_3b: (3B, ...) expanded tensor
            original_batch_size: B, the original batch size
        
        Returns:
            list of 3 tensors, each with shape (B, ...)
            [scenario0_tensor, scenario1_tensor, scenario2_tensor]
        """
        B = original_batch_size
        return [
            tensor_3b[0:B],        # scenario 0: both
            tensor_3b[B:2*B],      # scenario 1: vision_only
            tensor_3b[2*B:3*B]     # scenario 2: action_only
        ]
    
    def prepare_input(self, inputs: dict):
        """
        Prepare inputs for tokenizer forward pass
        
        Extracts:
        - obs_input: observation vision inputs (eagle_ prefixed)
        - goal_input: goal vision inputs (goal_image_ prefixed)
        - action_inputs: action and state for action encoder
        - tokenizer_inputs: reconstruction targets
        
        Args:
            inputs: Input dictionary, can optionally contain 'pv' and 'pa' keys
                   to control visual/action presence masks
        """
        # DINOv2 path: directly use ImageNet-normalized images
        obs_input = inputs["imagenet_obs_images"]  # (B, V*T, C, H, W)
        goal_input = inputs["imagenet_goal_images"]  # (B, V, C, H, W)

        action_inputs = self.action_decoder.prepare_input(inputs)

        # Tokenizer-specific inputs (reconstruction targets + presence masks)
        batch_size = inputs['state'].shape[0]
        device = inputs['state'].device if torch.is_tensor(inputs['state']) else None

        # pv/pa override > model default
        if 'pv' in inputs:
            pv = inputs['pv'].to(device) if hasattr(inputs['pv'], 'to') else torch.tensor(inputs['pv'], device=device, dtype=torch.long)
        else:
            pv = torch.full((batch_size,), self.default_pv, dtype=torch.long, device=device)

        if 'pa' in inputs:
            pa = inputs['pa'].to(device) if hasattr(inputs['pa'], 'to') else torch.tensor(inputs['pa'], device=device, dtype=torch.long)
        else:
            pa = torch.full((batch_size,), self.default_pa, dtype=torch.long, device=device)

        tokenizer_inputs = BatchFeature(data={
            'target_action': inputs['action'],
            'pv': pv,  # 1: visual present, 0: absent
            'pa': pa,  # 1: action present, 0: absent
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
    
    def forward(self, inputs: dict, action_mode: bool = False) -> BatchFeature:
        """
        Forward pass.

        Args:
            inputs: Input dictionary containing:
                - imagenet_obs_images: (B, V*T, C, H, W) ImageNet-normalized obs images
                - imagenet_goal_images: (B, V, C, H, W) ImageNet-normalized goal images
                - action: (B, T, action_dim)
                - state: (B, state_dim)
                - embodiment_id: (B,)
            action_mode: If True, perform action prediction; if False, train reconstruction.
        """
        # Get global_step from Trainer (passed via inputs), then remove it from inputs
        # to avoid it being processed as model input data (logging uses output_dict metrics).
        inputs.pop('global_step', None)

        # ========== Multi-Scenario Training ==========
        original_batch_size = None
        scenario_names = None
        use_multi_scenario = (
            self.config.use_multi_scenario_training and
            self.training and
            not action_mode
        )

        # Prepare inputs on original B-sized batch (encoders only need B)
        obs_input, goal_input, action_inputs, tokenizer_inputs = self.prepare_input(inputs)
        batch_size = inputs['state'].shape[0]

        # ========== Step 1: Encoder Forward (on B, not 3B) ==========

        # 1.1 Vision Branch: obs + goal → vision query features (per-query, hidden_size)
        vision_query_features, obs_embeds, goal_embeds_dino = self.vision_branch(
            obs_input=obs_input,
            goal_input=goal_input,
            batch_size=batch_size
        )  # (B, query_num, H), (B, N, H), (B, N, H)

        # 1.2 Action Branch: action + state → action query features
        action_query_features, state_features = self.action_branch(
            actions=action_inputs['action'],  # (B, T, action_dim)
            state=action_inputs['state'],  # (B, state_dim)
            cat_ids=action_inputs['embodiment_id'],  # (B,)
        )

        # ========== Multi-Scenario Expansion (post-encoder, pre-fusion) ==========
        if use_multi_scenario:
            original_batch_size = batch_size
            scenario_names = ['both', 'vision_only', 'action_only']
            device = vision_query_features.device

            # Expand encoder outputs: (B, ...) → (3B, ...)
            vision_query_features = vision_query_features.repeat(3, 1, 1)
            action_query_features = action_query_features.repeat(3, 1, 1)
            obs_embeds = obs_embeds.repeat(3, 1, 1)
            goal_embeds_dino = goal_embeds_dino.repeat(3, 1, 1)
            state_features = state_features.repeat(3, 1, 1)

            # Expand decoder inputs: obs/goal images and action_inputs
            obs_input = obs_input.repeat(3, *([1] * (obs_input.ndim - 1)))
            goal_input = goal_input.repeat(3, *([1] * (goal_input.ndim - 1)))
            action_inputs = {
                k: v.repeat(3, *([1] * (v.ndim - 1))) if torch.is_tensor(v) else v
                for k, v in action_inputs.items()
            }

            # Construct 3B pv/pa masks
            B = original_batch_size
            tokenizer_inputs['pv'] = torch.cat([
                torch.ones(B, dtype=torch.long, device=device),   # both
                torch.ones(B, dtype=torch.long, device=device),   # vision_only
                torch.zeros(B, dtype=torch.long, device=device),  # action_only
            ])
            tokenizer_inputs['pa'] = torch.cat([
                torch.ones(B, dtype=torch.long, device=device),   # both
                torch.zeros(B, dtype=torch.long, device=device),  # vision_only
                torch.ones(B, dtype=torch.long, device=device),   # action_only
            ])

            batch_size = batch_size * 3

        # ========== Step 2: Fusion (pv/pa-routed) → unit tokens ==========
        unit_tokens = self.fusion(
            visual_tokens=vision_query_features,
            action_tokens=action_query_features,
            pv=tokenizer_inputs['pv'],
            pa=tokenizer_inputs['pa']
        )  # (B or 3B, query_num, H)

        # Optional: multi-scenario only — distill routed unit_tokens for ablated pv/pa rows
        # toward the full (pv=1,pa=1) unit slice. All regimes live in unit_tokens, not in encoder features.
        unit_fusion_distill_loss = None
        unit_fusion_distill_w = None
        if self.training and not action_mode and use_multi_scenario:
            _ufd_w = self.loss_weights.get("unit_fusion_distill")
            if _ufd_w is not None and float(_ufd_w) > 0.0:
                unit_fusion_distill_w = float(_ufd_w)
                B0 = original_batch_size
                u_teacher = unit_tokens[:B0].detach()
                u_vis_only = unit_tokens[B0 : 2 * B0]
                u_act_only = unit_tokens[2 * B0 : 3 * B0]
                unit_fusion_distill_loss = F.mse_loss(u_vis_only, u_teacher) + F.mse_loss(
                    u_act_only, u_teacher
                )
        
        # ========== Step 3: Vector Quantization ==========
        unit_tokens_down = self.vq_down_resampler(unit_tokens)
        quantized_tokens, vq_indices, vq_loss = self.vq(unit_tokens_down)
        quantized_tokens_up = self.bridge_projector(quantized_tokens)

        # ========== Step 4: Prepare Action Decoder Input ==========
        # ResNet ActionDecoder consumes only the latent goal tokens (no obs).
        goal_embeds = quantized_tokens_up + self.pos_embed.unsqueeze(0)  # (B, query_num, H)

        output_dict = {}

        # ========== Vision reconstruction (mix: when obs+goal present) ==========
        if obs_input is not None and goal_input is not None:
            is_dino_mode = getattr(self, 'is_dino_mode', False)

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

                output_dict['vision_cos_loss'] = cos_loss
                output_dict['vision_cos_sim'] = cos_sim.mean()
                output_dict['vision_recon_loss'] = vision_recon_loss

                if use_multi_scenario:
                    for i, name in enumerate(scenario_names):
                        start, end = i * original_batch_size, (i + 1) * original_batch_size
                        output_dict[f'{name}_vision_cos_loss'] = cos_loss_per_sample[start:end].mean()
                        output_dict[f'{name}_vision_recon_loss'] = output_dict[f'{name}_vision_cos_loss']

            else:
                cond_images = obs_input.squeeze(1)
                target_images = goal_input.squeeze(1)

                reconstructed_images = self.vision_decoder(
                    cond_input=cond_images,
                    latent_motion_tokens=quantized_tokens_up
                )

                mse_per_sample = F.mse_loss(reconstructed_images, target_images, reduction='none').mean(dim=[1, 2, 3])
                mse_loss = mse_per_sample.mean()
                vision_recon_loss = mse_loss
                output_dict['vision_mse_loss'] = mse_loss

                if self.use_lpips_loss and self.lpips_loss is not None:
                    lpips_per_sample = self.lpips_loss(
                        denormalize_imagenet(reconstructed_images) * 2 - 1,
                        denormalize_imagenet(target_images) * 2 - 1,
                    ).view(-1)
                    lpips_loss = lpips_per_sample.mean()
                    vision_recon_loss = vision_recon_loss + self.loss_weights.get('lpips', 0.1) * lpips_loss
                    output_dict['vision_lpips_loss'] = lpips_loss

                output_dict['vision_recon_loss'] = vision_recon_loss

                if use_multi_scenario:
                    for i, name in enumerate(scenario_names):
                        start, end = i * original_batch_size, (i + 1) * original_batch_size
                        output_dict[f'{name}_vision_mse_loss'] = mse_per_sample[start:end].mean()
                        output_dict[f'{name}_vision_recon_loss'] = output_dict[f'{name}_vision_mse_loss']

                        if self.use_lpips_loss and self.lpips_loss is not None:
                            scenario_lpips = lpips_per_sample[start:end].mean()
                            output_dict[f'{name}_vision_lpips_loss'] = scenario_lpips
                            output_dict[f'{name}_vision_recon_loss'] = (
                                output_dict[f'{name}_vision_recon_loss']
                                + self.loss_weights.get('lpips', 0.1) * scenario_lpips
                            )
        else:
            vision_recon_loss = torch.tensor(0.0, device=quantized_tokens.device, dtype=torch.float32)
            output_dict['vision_recon_loss'] = vision_recon_loss

        if not action_mode:
            # Step 4b: Action Reconstruction (ResNet decoder consumes goal tokens only)
            action_decoder_input = BatchFeature(data={
                'backbone_features': goal_embeds,    # (B, query_num, H)
                'state_features': state_features,    # (B, 1, H)
            })

            action_head_outputs = self.action_decoder(action_decoder_input, action_inputs)
            action_recon_loss = action_head_outputs['loss']
            output_dict['action_recon_loss'] = action_recon_loss
            
            # Per-scenario action loss (simple split and mean)
            if use_multi_scenario and 'per_sample_loss' in action_head_outputs:
                action_per_sample = action_head_outputs['per_sample_loss']  # (3B,)
                for i, name in enumerate(scenario_names):
                    start, end = i * original_batch_size, (i + 1) * original_batch_size
                    output_dict[f'{name}_action_recon_loss'] = action_per_sample[start:end].mean()
            
            # Step 5: Combine Losses
            total_loss = (
                self.loss_weights['vision'] * vision_recon_loss +
                self.loss_weights['action'] * action_recon_loss +
                self.loss_weights['vq_commitment'] * vq_loss
            )
            if unit_fusion_distill_loss is not None:
                total_loss = total_loss + unit_fusion_distill_w * unit_fusion_distill_loss
                output_dict['unit_fusion_distill_loss'] = unit_fusion_distill_loss
            
            output_dict['loss'] = total_loss
            output_dict['vq_loss'] = vq_loss


            # Step 6: Compute VQ codebook usage (always, for TensorBoard logging)
            if self.training:
                with torch.no_grad():
                    if use_multi_scenario:
                        # Multi-scenario: compute codebook usage per scenario
                        vq_indices_list = self._split_3scenario_batch(vq_indices, original_batch_size)
                        
                        for scenario_idx, scenario_name in enumerate(scenario_names):
                            scenario_vq_indices = vq_indices_list[scenario_idx]  # (B, query_num) or (B, query_num, n_layers)
                            
                            if hasattr(self.vq, 'layers'):
                                # RVQ: multiple layers
                                for layer_idx in range(len(self.vq.layers)):
                                    layer_indices = scenario_vq_indices[..., layer_idx]  # (B, query_num)
                                    unique_codes = torch.unique(layer_indices).numel()
                                    output_dict[f'{scenario_name}_vq_active_codes_layer_{layer_idx}'] = unique_codes
                            else:
                                # Single VQ
                                unique_codes = torch.unique(scenario_vq_indices).numel()
                                output_dict[f'{scenario_name}_vq_active_codes'] = unique_codes
                    else:
                        # Single scenario mode
                        if hasattr(self.vq, 'layers'):
                            # RVQ: multiple layers
                            for layer_idx in range(len(self.vq.layers)):
                                layer_indices = vq_indices[..., layer_idx]  # (B, query_num)
                                unique_codes = torch.unique(layer_indices).numel()
                                output_dict[f'vq_active_codes_layer_{layer_idx}'] = unique_codes
                        else:
                            # Single VQ
                            unique_codes = torch.unique(vq_indices).numel()
                            output_dict['vq_active_codes'] = unique_codes

            return BatchFeature(data=output_dict)
        
        else:
            # ========== Inference Mode: Action Prediction ==========
            action_decoder_input = BatchFeature(data={
                'backbone_features': goal_embeds,    # (B, query_num, H)
                'state_features': state_features,    # (B, 1, H)
            })
            
            # Get action prediction
            action_pred = self.action_decoder.get_action(action_decoder_input, action_inputs)
            action_pred['vq_indices'] = vq_indices
    
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
        """Set trainable parameters"""
        # Store training flags for set_frozen_modules_to_eval_mode
        self.tune_vision_model = tune_vision_model
        self.tune_vision_m_former = tune_vision_m_former
        self.tune_bridge_projector = tune_bridge_projector
        self.tune_action_encoder = tune_action_encoder
        self.tune_fusion = tune_fusion
        self.tune_vq = tune_vq
        self.tune_vision_decoder = tune_vision_decoder

        # Vision Branch (shared for both goal and obs)
        if tune_vision_model:
            self.vision_branch.vision_model.requires_grad_(True)
        else:
            self.vision_branch.vision_model.requires_grad_(False)
        
        if tune_vision_m_former:
            self.vision_branch.m_former.requires_grad_(True)
        else:
            self.vision_branch.m_former.requires_grad_(False)
        
        # Bridge Projector (feature alignment)
        if tune_bridge_projector:
            self.bridge_projector.requires_grad_(True)
        else:
            self.bridge_projector.requires_grad_(False)
        
        # Action Encoder
        if tune_action_encoder:
            self.action_branch.requires_grad_(True)
        else:
            self.action_branch.requires_grad_(False)
        
        # Fusion
        if tune_fusion:
            self.fusion.requires_grad_(True)
        else:
            self.fusion.requires_grad_(False)
        
        # VQ
        if tune_vq:
            self.vq.requires_grad_(True)
        else:
            self.vq.requires_grad_(False)
        
        # Vision Decoder
        if tune_vision_decoder:
            self.vision_decoder.requires_grad_(True)
        else:
            self.vision_decoder.requires_grad_(False)
        
        # Action Decoder (ResNet): single requires_grad switch.
        # Two flags kept for from_pretrained API compat; OR them together.
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
        """
        Load pretrained tokenizer model
        
        Args:
            pretrained_model_name_or_path: Path or HF model ID
            **kwargs: Training configuration flags
        
        Returns:
            Loaded tokenizer model with configured trainable parameters
        """
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        # Extract training parameters
        tune_vision_model = kwargs.pop("tune_vision_model", True)
        tune_vision_m_former = kwargs.pop("tune_vision_m_former", True)
        tune_bridge_projector = kwargs.pop("tune_bridge_projector", True)
        tune_action_encoder = kwargs.pop("tune_action_encoder", True)
        tune_fusion = kwargs.pop("tune_fusion", True)
        tune_vq = kwargs.pop("tune_vq", True)
        tune_vision_decoder = kwargs.pop("tune_vision_decoder", True)
        tune_action_decoder_projector = kwargs.pop("tune_action_decoder_projector", True)
        tune_action_decoder_diffusion = kwargs.pop("tune_action_decoder_diffusion", True)

        # Extract unified_embodiment_id (for parameter sharing)
        unified_embodiment_id = kwargs.pop("unified_embodiment_id", getattr(model_config, 'unified_embodiment_id', None))
        # Mix training always uses vision recon when batch has images; ignore legacy flag from checkpoints / callers.
        kwargs.pop("compute_bridge_loss", None)
        
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
        
        # Download model
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            print(f"Model not found in HF hub. Loading from local path: {pretrained_model_name_or_path}")
            local_model_path = pretrained_model_name_or_path
        
        # Load model
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
        
        # Set trainable parameters
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
        
        # Update config
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
        """
        Inference method for action prediction
        
        Args:
            inputs: Input dictionary
        
        Returns:
            BatchFeature with action predictions
        """
        return self.forward(inputs=inputs, action_mode=True)
    
    def set_frozen_modules_to_eval_mode(self):
        """
        Set frozen modules to eval mode
        
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            # Vision Branch (shared for both goal and obs)
            if not getattr(self, 'tune_vision_model', True):
                self.vision_branch.vision_model.eval()
            if not getattr(self, 'tune_vision_m_former', True):
                self.vision_branch.m_former.eval()
            
            # Bridge Projector
            if not getattr(self, 'tune_bridge_projector', True):
                self.bridge_projector.eval()
            
            # Action Encoder
            if not getattr(self, 'tune_action_encoder', True):
                self.action_branch.eval()
            
            # Fusion
            if not getattr(self, 'tune_fusion', True):
                self.fusion.eval()
            
            # VQ
            if not getattr(self, 'tune_vq', True):
                self.vq.eval()
            
            # Vision Decoder
            if not getattr(self, 'tune_vision_decoder', True):
                self.vision_decoder.eval()
            
            
            # Image Type Embedding
    
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
        # Use action_decoder's dtype as reference (following Bridge pattern)
        return self.action_decoder.dtype


# Register Tokenizer
AutoConfig.register("gr00t_tokenizer", GR00T_Tokenizer_Config)
AutoModel.register(GR00T_Tokenizer_Config, GR00T_Tokenizer)
