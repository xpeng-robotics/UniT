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
Vector Quantization modules for tokenization

Implements VectorQuantizer2 and ResidualVectorQuantizer (RVQ) with 
VQ-VAE compatible interfaces for learned discrete representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from transformers import PretrainedConfig
from torch import einsum
from einops import rearrange
import torch.distributed as dist

@dataclass
class VectorQuantizerConfig(PretrainedConfig):
    """Configuration for VectorQuantizer2"""
    
    n_e: int = field(default=512, metadata={"help": "Number of embeddings in codebook"})
    e_dim: int = field(default=256, metadata={"help": "Dimension of each embedding"})
    beta: float = field(default=0.25, metadata={"help": "Commitment loss weight"})
    remap: Optional[str] = field(default=None, metadata={"help": "Path to remapping file"})
    unknown_index: str = field(default="random", metadata={"help": "How to handle unknown indices (random/extra/closest)"})
    sane_index_shape: bool = field(default=False, metadata={"help": "Reshape indices to match input shape"})
    legacy: bool = field(default=True, metadata={"help": "Use legacy (buggy) loss for backward compatibility"})
    code_restart: bool = field(default=True, metadata={"help": "Enable dead code restart"})
    restart_interval: int = field(default=100, metadata={"help": "Steps between restart checks"})
    max_restart_steps: int = field(default=50000, metadata={"help": "Maximum steps to perform restarts"})
    l2_norm: bool = field(default=False, metadata={"help": "Use L2 norm for distance calculation"})
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class VQStageConfig:
    """Configuration for a single RVQ stage"""
    n_e: int = 512
    e_dim: int = 256
    beta: float = 0.25
    weight: float = 1.0
    remap: Optional[str] = None
    unknown_index: str = "random"
    legacy: bool = True
    code_restart: bool = True
    sane_index_shape: bool = False
    restart_interval: int = 100
    max_restart_steps: int = 50000
    l2_norm: bool = False

@dataclass  
class ResidualVectorQuantizerConfig(PretrainedConfig):
    """Configuration for Residual Vector Quantizer"""
    
    stages: List[Dict[str, Any]] = field(default=None, metadata={"help": "List of stage configurations"})
    sane_index_shape: bool = field(default=False, metadata={"help": "Reshape indices to match input shape"})
    legacy: bool = field(default=True, metadata={"help": "Use legacy loss for backward compatibility"})
    
    # Fallback to single-stage VQ or multi-stage with same config
    n_e: Optional[int] = field(default=None, metadata={"help": "Fallback: number of embeddings"})
    e_dim: Optional[int] = field(default=None, metadata={"help": "Fallback: embedding dimension"})
    beta: Optional[float] = field(default=None, metadata={"help": "Fallback: commitment loss weight"})
    n_q: Optional[int] = field(default=None, metadata={"help": "Fallback: number of quantizer layers"})
    
    # VQ type selection: "default" (current impl), "ema" (EMA + rotation trick), or "fsq" (Finite Scalar Quantization)
    vq_type: str = field(default="default", metadata={"help": "VQ type: 'default', 'ema', or 'fsq'"})
    
    # FSQ specific params
    fsq_levels: Optional[List[int]] = field(default=None, metadata={"help": "FSQ levels per dimension, e.g. [8, 5, 5, 5] (fsq mode)"})
    
    # Additional params for ema mode
    rotation_trick: bool = field(default=True, metadata={"help": "Use rotation trick for gradient (ema mode)"})
    kmeans_init: bool = field(default=False, metadata={"help": "Use K-means initialization (ema mode)"})
    threshold_ema_dead_code: int = field(default=2, metadata={"help": "Threshold for dead code expiration (ema mode)"})
    commitment_weight: Optional[float] = field(default=None, metadata={"help": "Commitment weight (ema mode, defaults to beta)"})
    
    # EMA decay
    decay: float = field(default=0.8, metadata={"help": "EMA decay rate (ema mode)"})
    
    # Quantize dropout (from Encodec)
    quantize_dropout: bool = field(default=False, metadata={"help": "Enable quantize dropout for RVQ (ema mode)"})
    quantize_dropout_cutoff_index: int = field(default=1, metadata={"help": "Minimum layers to keep when dropout (ema mode)"})
    quantize_dropout_prob: float = field(default=1.0, metadata={"help": "Probability of applying dropout when enabled (ema mode)"})
    
    # Codebook regularization
    orthogonal_reg_weight: float = field(default=0.0, metadata={"help": "Orthogonal regularization weight (ema mode)"})
    codebook_diversity_loss_weight: float = field(default=0.0, metadata={"help": "Codebook diversity loss weight (ema mode)"})
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Modern Vector Quantizer
    - Uses torch.cdist for fast distance calculation.
    - Includes 'Dead Code Restart' for high codebook usage.
    - Removes legacy 'remap' logic for clean, efficient training.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_e = int(config.n_e)
        self.e_dim = int(config.e_dim)
        self.beta = float(config.beta)
        
        # Legacy loss computation mode (for backward compatibility)
        self.legacy = getattr(config, 'legacy', True)
        
        # Index shape behavior
        self.sane_index_shape = getattr(config, 'sane_index_shape', False)
        
        # Core Codebook
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if getattr(config, 'l2_norm', False):
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=1)
        # self.embedding.weight.data.normal_(0, 1)

        # Training Stability: Code Restart
        # Default to True unless config explicitly says False
        code_restart = getattr(config, 'code_restart', True)
        self.code_restart = True if code_restart is None else bool(code_restart)
        self.register_buffer("usage", torch.zeros(self.n_e), persistent=False)
        self.register_buffer("internal_step", torch.tensor(0, dtype=torch.long), persistent=False)
        
        # Ensure restart_interval and max_restart_steps are not None
        restart_interval = getattr(config, 'restart_interval', 100)
        self.restart_interval = 100 if restart_interval is None else int(restart_interval)
        
        max_restart_steps = getattr(config, 'max_restart_steps', 50000)
        self.max_restart_steps = 50000 if max_restart_steps is None else int(max_restart_steps)

    def forward(self, z: torch.Tensor, temp=None, rescale_logits: bool = False, return_logits: bool = False):
        """
        Args:
            z: Input tensor (B, ..., D)
        """
        assert temp is None or temp == 1.0, "Temperature not supported"
        assert rescale_logits is False
        assert return_logits is False

        # 1. Flatten input; normalize z before restart sampling when using spherical VQ.
        input_shape = z.shape
        z_flattened = z.reshape(-1, self.e_dim)

        # With L2-normalized codes, restart samples must live on the same manifold as `z_flattened`.
        if getattr(self.config, 'l2_norm', False):
            z_flattened = F.normalize(z_flattened, p=2, dim=1)

        # Dead-code restart must run *before* building `embedding_weight`.
        # Restart writes `self.embedding.weight` in place; if we had already wrapped the
        # weight in `F.normalize(...)`, autograd would see a mutated input (in-place error).
        if self.training and self.code_restart:
            self.internal_step += 1

            if self.internal_step % self.restart_interval == 0 and \
               self.internal_step < self.max_restart_steps:

                self._perform_restart_ddp_safe(z_flattened)

                self.reset_usage()

        # 2. Code vectors for the distance step (fresh tensor if L2-normalized).
        if getattr(self.config, 'l2_norm', False):
            embedding_weight = F.normalize(self.embedding.weight, p=2, dim=1)
        else:
            embedding_weight = self.embedding.weight

        d = torch.cdist(z_flattened, embedding_weight)

        min_encoding_indices = torch.argmin(d, dim=1)

        z_q = F.embedding(min_encoding_indices, embedding_weight).view(input_shape)

        if self.training and self.code_restart:
            self.update_usage(min_encoding_indices)

        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()

        indices = min_encoding_indices.reshape(input_shape[:-1])

        return z_q, indices, loss
        
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Get embeddings for indices, with optional L2 normalization"""
        embeddings = self.embedding(indices)
        if getattr(self.config, 'l2_norm', False):
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    # --- Dead Code Restart Helpers ---
    
    def update_usage(self, min_enc):
        """
        Accumulate usage counts
        
        Safety: Only runs in training mode with code_restart enabled.
        """
        if not self.training or not self.code_restart:
            return
        
        min_enc = min_enc.reshape(-1)
        updates = torch.bincount(min_enc, minlength=self.n_e).type_as(self.usage)
        self.usage += updates

    def reset_usage(self):
        """
        Reset usage stats (Call at epoch start)
        
        Safety: Only runs in training mode with code_restart enabled.
        """
        if self.training and self.code_restart:
            self.usage.zero_()

    def _perform_restart_ddp_safe(self, batch_z: torch.Tensor):
        """
        Restarts dead codes by sampling from the current batch.
        Guarantees consistency across DDP ranks by letting Rank 0 choose and broadcast.
        """
        if not self.training or not self.code_restart:
            return

        # 1. Aggregate Usage across all GPUs
        if dist.is_initialized():
            dist.all_reduce(self.usage, op=dist.ReduceOp.SUM)

        # 2. Identify Dead Codes
        dead_codes_indices = torch.nonzero(self.usage < 1).squeeze(1)
        num_dead = len(dead_codes_indices)

        target_dtype = self.embedding.weight.dtype

        if num_dead > 0:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[VectorQuantizer] Step {self.internal_step.item()}: Restarting {num_dead}/{self.n_e} dead codes with batch sampling.")

            # Buffer dtype must match `embedding` for broadcast and assignment on all ranks.
            replacement_vectors = torch.zeros(num_dead, self.e_dim, device=self.device, dtype=target_dtype)

            if not dist.is_initialized() or dist.get_rank() == 0:
                n_batch = batch_z.shape[0]

                if n_batch >= num_dead:
                    rand_idx = torch.randperm(n_batch, device=self.device)[:num_dead]
                else:
                    rand_idx = torch.randint(0, n_batch, (num_dead,), device=self.device)

                sampled_vectors = batch_z[rand_idx].detach()
                replacement_vectors = sampled_vectors.to(dtype=target_dtype)

            if dist.is_initialized():
                dist.broadcast(replacement_vectors, src=0)

            with torch.no_grad():
                self.embedding.weight[dead_codes_indices] = replacement_vectors

    @property
    def device(self):
        return self.embedding.weight.device
    
    @property
    def dtype(self):
        return self.embedding.weight.dtype

# class L2ProjectedVectorQuantizer(VectorQuantizer):
#     """
#     L2-Norm Projected Vector Quantizer (Spherical VQ).
    
#     Inherits from VectorQuantizer to keep:
#     - Dead Code Restart logic
#     - Usage tracking
#     - Configuration handling
    
#     Changes:
#     - Normalizes input (z) and codebook (e) to unit sphere.
#     - Uses Cosine Similarity (Dot Product) for distance.
#     - Rescales quantized output by input magnitude (z_q = e_norm * |z|).
#     """
#     def forward(self, z: torch.Tensor, temp=None, rescale_logits: bool = False, return_logits: bool = False):
#         # Compatibility assertions
#         assert temp is None or temp == 1.0, "Temperature not supported in this version"
#         assert rescale_logits is False
#         assert return_logits is False

#         # --- 1. Debug Logging (Only Rank 0) ---
#         if dist.is_initialized() and dist.get_rank() == 0:
#             # Calculate magnitudes for monitoring
#             # print(f"L2VQ: z.mean(): {z.mean()}, z.std(): {z.std()}")
#             # print(f"L2VQ: self.embedding.weight.mean(): {self.embedding.weight.mean()}, self.embedding.weight.std(): {self.embedding.weight.std()}")
#             z_mag_debug = z.norm(p=2, dim=-1).mean()
#             emb_mag_debug = self.embedding.weight.norm(p=2, dim=-1).mean()
#             # print(f"L2VQ: z.shape: {z.mean(),z.std()}, self.embedding.weight.shape: {self.embedding.weight.mean(),self.embedding.weight.std()}")
#             # # print(f"L2VQ: z_mag: {z_mag_debug:.4f}, codebook_mag: {emb_mag_debug:.4f}")
#             # print(f"L2VQ: z_mag: {z_mag_debug:.4f}, codebook_mag: {emb_mag_debug:.4f}")
#         # --- 2. Training Stability: Code Restart (Inherited Logic) ---
#         if self.training and self.code_restart:
#             self.internal_step += 1
#             if self.internal_step % self.restart_interval == 0 and \
#                self.internal_step < self.max_restart_steps:
#                 self._perform_restart_ddp_safe()
#                 self.reset_usage()

#         # --- 3. Pre-processing & Flattening ---
#         input_shape = z.shape
#         z_flattened = z.reshape(-1, self.e_dim) # (N, D)

#         # --- 4. L2 Normalization & Magnitude Extraction ---
#         # A. Extract Input Magnitude (Scale)
#         # Add epsilon to prevent division by zero
#         z_mag = torch.norm(z_flattened, p=2, dim=1, keepdim=True) + 1e-6
        
#         # B. Normalize Input (Project to Sphere)
#         z_normalized = z_flattened / z_mag

#         # C. Normalize Codebook (On-the-fly)
#         # We don't change the weights permanently, just normalize a view for calculation
#         w_normalized = F.normalize(self.embedding.weight, p=2, dim=1)

#         # --- 5. Compute Distances (Cosine Similarity) ---
#         # On sphere, minimizing Euclidean distance <=> Maximizing Dot Product
#         # (N, D) @ (D, Codes) -> (N, Codes)
#         dists = torch.matmul(z_normalized, w_normalized.t())

#         # --- 6. Find Nearest Neighbors ---
#         # We want MAX similarity (ArgMax)
#         min_encoding_indices = torch.argmax(dists, dim=1)
        
#         # --- 7. Get Quantized Vectors & Rescale ---
#         # A. Retrieve Unit Vectors from Codebook
#         z_q_unit = F.embedding(min_encoding_indices, w_normalized)
        
#         # B. Rescale: Combine Vision Direction (Codebook) with Action Force (Input Magnitude)
#         # z_q = e_norm * |z|
#         z_q = z_q_unit * z_mag
        
#         # Reshape back to original dimensions
#         z_q = z_q.view(input_shape)
#         z_q_unit = z_q_unit.view(input_shape)
#         z_normalized = z_normalized.view(input_shape)

#         # --- 8. Usage Tracking (Inherited Logic) ---
#         if self.training and self.code_restart:
#             self.update_usage(min_encoding_indices)

#         # --- 9. Compute Loss (On the Sphere) ---
#         # Critical: Compute loss on normalized vectors to fix scale mismatch issues.
#         # This ensures gradients only affect direction, not magnitude.
        
#         if not self.legacy:
#             # Modern formula
#             loss = self.beta * torch.mean((z_q_unit.detach() - z_normalized) ** 2) + \
#                    torch.mean((z_q_unit - z_normalized.detach()) ** 2)
#         else:
#             # Legacy formula
#             loss = torch.mean((z_q_unit.detach() - z_normalized) ** 2) + self.beta * \
#                    torch.mean((z_q_unit - z_normalized.detach()) ** 2)

#         # --- 10. Straight Through Estimator (STE) ---
#         # Forward pass uses z_q (rescaled), Backward pass flows to z
#         z_q = z + (z_q - z).detach()

#         # --- 11. Reshape Indices ---
#         if self.sane_index_shape:
#             indices = min_encoding_indices.reshape(input_shape[:-1])
#         else:
#             indices = min_encoding_indices

#         return z_q, indices, loss

#     def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
#         """
#         Get embeddings for indices.
#         For L2VQ, this returns the NORMALIZED embeddings (directions).
#         """
#         w_normalized = F.normalize(self.embedding.weight, p=2, dim=1)
#         return F.embedding(indices, w_normalized)

class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer (RVQ)
    Applies multiple stages of Vector Quantization to the residual.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Parse stages from config
        # If config.stages is a list of dicts, use it. 
        # Otherwise create n_q layers based on global params.
        if hasattr(config, 'stages') and config.stages is not None:
            # Inherit global restart params to each stage if not specified
            stages_cfg = []
            for stage in config.stages:
                stage_copy = dict(stage) if isinstance(stage, dict) else vars(stage).copy()
                if 'code_restart' not in stage_copy:
                    stage_copy['code_restart'] = getattr(config, 'code_restart', True)
                if 'restart_interval' not in stage_copy:
                    stage_copy['restart_interval'] = getattr(config, 'restart_interval', 100)
                if 'max_restart_steps' not in stage_copy:
                    stage_copy['max_restart_steps'] = getattr(config, 'max_restart_steps', 50000)
                stages_cfg.append(stage_copy)
        else:
            # Fallback for simple config
            n_q = getattr(config, 'n_q', 1)
            # Create n_q copies of the config
            stages_cfg = []
            for _ in range(n_q):
                stage_dict = {
                    'n_e': getattr(config, 'n_e', 512),
                    'e_dim': getattr(config, 'e_dim', 256),
                    'beta': getattr(config, 'beta', 0.25),
                    'legacy': getattr(config, 'legacy', True),
                    'sane_index_shape': False,  # Internal VQs don't need this
                    'code_restart': getattr(config, 'code_restart', True),
                    'restart_interval': getattr(config, 'restart_interval', 100),
                    'max_restart_steps': getattr(config, 'max_restart_steps', 50000),
                }
                stages_cfg.append(stage_dict)

        self.layers = nn.ModuleList()
        
        for i, stage_conf in enumerate(stages_cfg):
            # Ensure stage_conf is an object with attributes, or wrap it if it's a dict
            if isinstance(stage_conf, dict):
                # Quick wrapper class to simulate config object
                class DictConfig:
                    def __init__(self, d): 
                        self.__dict__.update(d)
                    def __getattr__(self, key): 
                        return self.__dict__.get(key)
                stage_conf_obj = DictConfig(stage_conf)
            else:
                stage_conf_obj = stage_conf
            
            self.layers.append(VectorQuantizer(stage_conf_obj))

    def forward(self, z: torch.Tensor, n_q: int = None):
        """
        Args:
            z: Input (B, ..., D)
            n_q: Optional, number of quantizers to use (for progressive training)
        """
        residual = z
        quantized_out = 0.0
        total_loss = 0.0
        all_indices = []

        # Determine how many layers to run
        num_layers = len(self.layers) if n_q is None else min(n_q, len(self.layers))

        for i in range(num_layers):
            layer = self.layers[i]
            
            # 1. Quantize the residual
            # z_q_k comes with STE: z_q_k = residual + (e_k - residual).detach()
            z_q_k, indices_k, loss_k = layer(residual)
            
            # 2. Accumulate output
            # We want the final output to be sum(e_k). 
            # Since z_q_k roughly equals e_k (numerically), we can accumulate it.
            # But for the residual update, we must be careful.
            quantized_out = quantized_out + z_q_k
            
            # 3. Update residual for next layer
            # CRITICAL: detach() to stop gradient flow from layer k+1 to layer k's codebook
            # We want layer k+1 to learn the "fixed" error of layer k.
            # e_k = z_q_k (numerically, if we ignore the STE graph for a moment)
            # residual_new = residual_old - e_k
            residual = residual - z_q_k.detach()
            
            total_loss = total_loss + loss_k
            all_indices.append(indices_k)

        # 4. Final STE adjustment (Optional but clean)
        # The quantized_out already has gradients flowing to all codebooks via sum.
        # But we want the gradient from the decoder to flow directly to input z as well.
        # z_q_final = z + (sum(e_k) - z).detach()
        # Since we accumulated z_q_k (which are roughly residual_k), 
        # quantized_out is effectively sum(e_k) + (z - sum(e_k)).detach() ? No.
        
        # Let's stick to the cleanest SoundStream method:
        # z_q = z + (quantized_out - z).detach() 
        # Note: quantized_out constructed above currently contains STE chains.
        # To avoid double STE or graph complexity, usually RVQ implementations just return quantized_out 
        # because z_q_k already bridges the gradient.
        
        z_q = quantized_out
        # print("all_indices: ", all_indices)
        # Stack indices: (B, ..., num_layers)
        indices = torch.stack(all_indices, dim=-1)

        return z_q, indices, total_loss
    
    def get_codebook_entry(self, indices: torch.Tensor, layer_idx: int = None):
        """
        Get codebook embeddings for given indices
        
        Args:
            indices: Stacked indices (B, ..., num_layers)
            layer_idx: Optional, if provided, only get embeddings from specific layer
        
        Returns:
            z_q: Quantized embeddings (sum of per-stage embeddings)
        """
        if layer_idx is not None:
            # Get from specific layer only
            return self.layers[layer_idx].get_codebook_entry(indices)
        
        # indices shape: (B, ..., num_layers)
        # Split along last dimension to get per-stage indices
        per_stage_indices = [indices[..., i] for i in range(len(self.layers))]
        
        # Sum embeddings from all stages
        embeds = []
        for k, vq in enumerate(self.layers):
            idx_k = per_stage_indices[k]
            e_k = vq.get_codebook_entry(idx_k)
            embeds.append(e_k)
        
        z_q = torch.stack(embeds, dim=0).sum(dim=0)
        return z_q
    
    @property
    def device(self):
        """Get device of the module"""
        return self.layers[0].device if len(self.layers) > 0 else None
    
    @property
    def dtype(self):
        """Get dtype of the module"""
        return self.layers[0].dtype if len(self.layers) > 0 else None


# =============================================================================
# ResidualVQ Wrapper for vector-quantize-pytorch library (ema mode)
# =============================================================================

class _VQLayerWrapper(nn.Module):
    """
    Wrapper to make vector-quantize-pytorch's VectorQuantize compatible
    with the expected interface (n_e, embedding.weight, config.l2_norm)
    
    This is an internal class used by ResidualVQFromLib.
    """
    def __init__(self, vq_layer, use_cosine_sim: bool):
        super().__init__()
        self._vq = vq_layer
        self._use_cosine_sim = use_cosine_sim
        
        # Create a fake config object for l2_norm access
        class FakeConfig:
            def __init__(self, l2_norm):
                self.l2_norm = l2_norm
        self.config = FakeConfig(use_cosine_sim)
    
    @property
    def n_e(self):
        """Codebook size (alias for codebook_size)"""
        return self._vq.codebook_size
    
    @property
    def embedding(self):
        """
        Fake embedding module that provides .weight attribute
        Returns an object with .weight property that gives codebook weights
        """
        vq = self._vq
        use_cosine_sim = self._use_cosine_sim
        
        class FakeEmbedding:
            @property
            def weight(self):
                # _codebook.embed shape: (1, codebook_size, dim)
                # Need to return (codebook_size, dim)
                w = vq._codebook.embed[0]  # (codebook_size, dim)
                if use_cosine_sim:
                    w = F.normalize(w, p=2, dim=-1)
                return w
        
        return FakeEmbedding()
    
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Get embeddings for indices"""
        # Use the wrapper's embedding.weight for consistency
        w = self.embedding.weight  # (codebook_size, dim)
        return F.embedding(indices, w)
    
    @property
    def device(self):
        return self._vq._codebook.embed.device
    
    @property
    def dtype(self):
        return self._vq._codebook.embed.dtype


class ResidualVQFromLib(nn.Module):
    """
    Wrapper around vector-quantize-pytorch's ResidualVQ
    
    Provides interface compatibility with existing ResidualVectorQuantizer:
    - Forward returns (quantized, indices, total_loss_scalar)
    - self.layers provides access to individual VQ layers
    - layer.n_e, layer.embedding.weight, layer.config.l2_norm available
    - get_codebook_entry(indices, layer_idx) method
    
    Key features:
    - Uses cosine similarity for distance calculation
    - Uses EMA for codebook updates (no learnable codebook)
    - Supports rotation trick for better gradient flow
    
    Reference: vector-quantize-pytorch by lucidrains
    Paper: SoundStream (arXiv:2107.03312)
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Lazy import to avoid dependency issues
        try:
            from gr00t.model.tokenizer.vector_quantize_pytorch import ResidualVQ
        except ImportError:
            raise ImportError(
                "vector-quantize-pytorch is required for vq_type='ema'. "
                "Please ensure gr00t/model/tokenizer/vector_quantize_pytorch/ exists."
            )
        
        # Parse config
        num_quantizers = getattr(config, 'n_q', 1)
        codebook_size = getattr(config, 'n_e', 512)
        dim = getattr(config, 'e_dim', 256)
        use_cosine_sim = getattr(config, 'l2_norm', False)  # Default False for ema mode (L2 distance)
        
        # Commitment weight: use config.commitment_weight if set, else fall back to beta
        commitment_weight = getattr(config, 'commitment_weight', None)
        if commitment_weight is None:
            commitment_weight = getattr(config, 'beta', 0.25)
        
        # Additional options
        rotation_trick = getattr(config, 'rotation_trick', True)
        kmeans_init = getattr(config, 'kmeans_init', False)
        threshold_ema_dead_code = getattr(config, 'threshold_ema_dead_code', 2)
        decay = getattr(config, 'decay', 0.8)
        
        # Quantize dropout (from Encodec)
        quantize_dropout = getattr(config, 'quantize_dropout', False)
        quantize_dropout_cutoff_index = getattr(config, 'quantize_dropout_cutoff_index', 1)
        quantize_dropout_prob = getattr(config, 'quantize_dropout_prob', 1.0)
        
        # Codebook regularization
        orthogonal_reg_weight = getattr(config, 'orthogonal_reg_weight', 0.0)
        codebook_diversity_loss_weight = getattr(config, 'codebook_diversity_loss_weight', 0.0)
        
        # Create ResidualVQ from library
        # Key settings:
        # - ema_update=True: EMA codebook update
        # - rotation_trick=True: Better gradient flow
        # - quantize_dropout: Random dropout of later RVQ layers during training
        self._rvq = ResidualVQ(
            dim=dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            use_cosine_sim=use_cosine_sim,
            commitment_weight=commitment_weight,
            rotation_trick=rotation_trick,
            ema_update=True,
            kmeans_init=kmeans_init,
            threshold_ema_dead_code=threshold_ema_dead_code,
            decay=decay,
            quantize_dropout=quantize_dropout,
            quantize_dropout_cutoff_index=quantize_dropout_cutoff_index,
            quantize_dropout_prob=quantize_dropout_prob,
            orthogonal_reg_weight=orthogonal_reg_weight,
            codebook_diversity_loss_weight=codebook_diversity_loss_weight,
        )
        
        # Save config for external access
        self._config = config
        self._use_cosine_sim = use_cosine_sim
        self._num_quantizers = num_quantizers
        
        # Create layer wrappers for interface compatibility
        # Note: Use plain list instead of nn.ModuleList to avoid shared tensor issues
        # when saving checkpoints (the actual modules are already in self._rvq.layers)
        self._layer_wrappers = [
            _VQLayerWrapper(vq, use_cosine_sim) 
            for vq in self._rvq.layers
        ]
        
        print(f"[ResidualVQFromLib] Initialized with:")
        print(f"  - num_quantizers: {num_quantizers}")
        print(f"  - codebook_size: {codebook_size}")
        print(f"  - dim: {dim}")
        print(f"  - use_cosine_sim: {use_cosine_sim}")
        print(f"  - commitment_weight: {commitment_weight}")
        print(f"  - rotation_trick: {rotation_trick}")
        print(f"  - decay: {decay}")
        print(f"  - kmeans_init: {kmeans_init}")
        print(f"  - threshold_ema_dead_code: {threshold_ema_dead_code}")
        print(f"  - quantize_dropout: {quantize_dropout}")
        print(f"  - quantize_dropout_cutoff_index: {quantize_dropout_cutoff_index}")
        print(f"  - quantize_dropout_prob: {quantize_dropout_prob}")
        print(f"  - orthogonal_reg_weight: {orthogonal_reg_weight}")
        print(f"  - codebook_diversity_loss_weight: {codebook_diversity_loss_weight}")
    
    def forward(self, z: torch.Tensor, n_q: int = None):
        """
        Forward pass
        
        Args:
            z: Input (B, ..., D)
            n_q: Optional, number of quantizers to use (NOT supported in this wrapper)
        
        Returns:
            quantized: Quantized output (B, ..., D)
            indices: Indices (B, ..., num_layers)
            total_loss: Scalar loss
        """
        if n_q is not None and n_q != self._num_quantizers:
            print(f"[Warning] ResidualVQFromLib does not support progressive n_q. Using all {self._num_quantizers} quantizers.")
        
        # Forward through the library's ResidualVQ
        quantized, indices, losses = self._rvq(z)
        
        # Convert losses from (num_quantizers,) tensor to scalar
        # This matches the expected interface
        total_loss = losses.sum()
        
        return quantized, indices, total_loss
    
    def get_codebook_entry(self, indices: torch.Tensor, layer_idx: int = None):
        """
        Get codebook embeddings for given indices
        
        Args:
            indices: Stacked indices (B, ..., num_layers)
            layer_idx: Optional, get embeddings from specific layer only
        
        Returns:
            z_q: Quantized embeddings (sum of per-stage embeddings if layer_idx=None)
        """
        if layer_idx is not None:
            # Get from specific layer
            if indices.shape[-1] == self._num_quantizers:
                layer_indices = indices[..., layer_idx]
            else:
                layer_indices = indices
            
            # Use the layer wrapper's get_codebook_entry
            return self._layer_wrappers[layer_idx].get_codebook_entry(layer_indices)
        
        # Get from all layers and sum
        # Use library's get_codes_from_indices for efficiency
        all_codes = self._rvq.get_codes_from_indices(indices)
        # all_codes shape: (num_quantizers, B, ..., dim)
        return all_codes.sum(dim=0)
    
    @property
    def layers(self):
        """Layer wrappers for backward compatibility (read-only)"""
        return self._layer_wrappers
    
    @property
    def device(self):
        """Get device of the module"""
        return self._rvq.layers[0]._codebook.embed.device
    
    @property
    def dtype(self):
        """Get dtype of the module"""
        return self._rvq.layers[0]._codebook.embed.dtype
    
    @property
    def n_e(self):
        """Codebook size of first layer (for backward compatibility)"""
        return self._layer_wrappers[0].n_e
    
    @property
    def e_dim(self):
        """Embedding dimension"""
        return self._rvq.codebook_dim


class ResidualFSQFromLib(nn.Module):
    """
    Wrapper for ResidualFSQ from vector-quantize-pytorch library.
    Provides a compatible interface with ResidualVectorQuantizer.
    
    FSQ (Finite Scalar Quantization) quantizes each dimension independently
    to a finite set of levels, without using an explicit codebook.
    """
    
    def __init__(self, config: ResidualVectorQuantizerConfig):
        super().__init__()
        
        from .vector_quantize_pytorch import ResidualFSQ
        
        # Get basic parameters
        dim = config.e_dim if config.e_dim else config.stages[0]['e_dim']
        num_quantizers = config.n_q if config.n_q else len(config.stages)
        
        # FSQ specific: levels
        levels = config.fsq_levels
        if levels is None:
            # Default: approximately match codebook_size=128
            levels = [8, 4, 4]
            print(f"[ResidualFSQFromLib] No fsq_levels specified, using default: {levels}")
        
        # Quantize dropout params
        quantize_dropout = getattr(config, 'quantize_dropout', False)
        quantize_dropout_cutoff_index = getattr(config, 'quantize_dropout_cutoff_index', 0)
        quantize_dropout_prob = getattr(config, 'quantize_dropout_prob', 1.0)
        
        # Create ResidualFSQ from library
        self._fsq = ResidualFSQ(
            levels=levels,
            num_quantizers=num_quantizers,
            dim=dim,
            quantize_dropout=quantize_dropout,
            quantize_dropout_cutoff_index=quantize_dropout_cutoff_index,
            quantize_dropout_prob=quantize_dropout_prob,
        )
        
        # Save config for external access
        self._config = config
        self._num_quantizers = num_quantizers
        self._levels = levels
        self._dim = dim
        
        # Compute implicit codebook size
        codebook_size = 1
        for l in levels:
            codebook_size *= l
        self._codebook_size = codebook_size
        
        # Create layer wrappers for interface compatibility (simplified)
        # Note: Use plain list instead of nn.ModuleList to avoid shared tensor issues
        # when saving checkpoints (the actual modules are already in self._fsq.layers)
        self._layer_wrappers = [
            _FSQLayerWrapper(fsq_layer, codebook_size) 
            for fsq_layer in self._fsq.layers
        ]
        
        print(f"[ResidualFSQFromLib] Initialized with:")
        print(f"  - num_quantizers: {num_quantizers}")
        print(f"  - levels: {levels}")
        print(f"  - codebook_size (implicit): {codebook_size}")
        print(f"  - dim: {dim}")
        print(f"  - quantize_dim: {len(levels)}")
        print(f"  - quantize_dropout: {quantize_dropout}")
        print(f"  - quantize_dropout_cutoff_index: {quantize_dropout_cutoff_index}")
        print(f"  - quantize_dropout_prob: {quantize_dropout_prob}")
    
    def forward(self, z: torch.Tensor, n_q: int = None):
        """
        Forward pass
        
        Args:
            z: Input (B, ..., D)
            n_q: Optional, number of quantizers to use (NOT supported)
        
        Returns:
            quantized: Quantized output (B, ..., D)
            indices: Indices (B, ..., num_layers)
            total_loss: Scalar loss (always 0 for FSQ, no commitment loss)
        """
        if n_q is not None and n_q != self._num_quantizers:
            print(f"[Warning] ResidualFSQFromLib does not support progressive n_q. Using all {self._num_quantizers} quantizers.")
        
        # Forward through the library's ResidualFSQ
        quantized, indices = self._fsq(z)
        
        # Convert indices to long (FSQ returns int32, but cross_entropy needs int64/long)
        indices = indices.long()
        
        # FSQ has no commitment loss, return 0
        # Use a tensor that requires grad to avoid DDP issues
        loss = z.new_zeros(())
        
        return quantized, indices, loss
    
    def get_codebook_entry(self, indices: torch.Tensor, layer_idx: int = None):
        """
        Get codebook embeddings for given indices
        
        Args:
            indices: Stacked indices (B, ..., num_layers)
            layer_idx: Optional, get embeddings from specific layer only
        
        Returns:
            z_q: Quantized embeddings (sum of per-stage embeddings if layer_idx=None)
        """
        if layer_idx is not None:
            # Get from specific layer
            if indices.shape[-1] == self._num_quantizers:
                layer_indices = indices[..., layer_idx]
            else:
                layer_indices = indices
            
            # Use the layer wrapper's get_codebook_entry
            return self._layer_wrappers[layer_idx].get_codebook_entry(layer_indices)
        
        # Get from all layers and sum
        all_codes = self._fsq.get_codes_from_indices(indices)
        # all_codes shape: (num_quantizers, B, ..., dim)
        return all_codes.sum(dim=0)
    
    @property
    def layers(self):
        """Layer wrappers for backward compatibility (read-only)"""
        return self._layer_wrappers
    
    @property
    def device(self):
        """Get device of the module"""
        return self._fsq.project_in.weight.device if self._fsq.has_projections else next(self._fsq.parameters()).device
    
    @property
    def dtype(self):
        """Get dtype of the module"""
        return self._fsq.project_in.weight.dtype if self._fsq.has_projections else next(self._fsq.parameters()).dtype
    
    @property
    def n_e(self):
        """Codebook size (implicit, for backward compatibility)"""
        return self._codebook_size
    
    @property
    def e_dim(self):
        """Embedding dimension"""
        return self._dim


class _FSQLayerWrapper(nn.Module):
    """
    Wrapper for a single FSQ layer to provide compatible interface.
    """
    
    def __init__(self, fsq_layer, codebook_size: int):
        super().__init__()
        self._fsq = fsq_layer
        self._n_e = codebook_size
    
    @property
    def n_e(self):
        """Codebook size"""
        return self._n_e
    
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get codebook embeddings for given indices.
        
        Args:
            indices: Indices tensor
        
        Returns:
            Quantized embeddings from implicit codebook
        """
        # Use FSQ's indices_to_codes method
        codes = self._fsq.indices_to_codes(indices)
        return codes


# Note: To use ResidualVQFromLib (ema mode) or ResidualFSQFromLib (fsq mode), 
# import them directly in tokenizer and select based on config.vq_type. Example:
#
#   from gr00t.model.tokenizer.vector_quantizer import (
#       ResidualVectorQuantizer,
#       ResidualVQFromLib,
#       ResidualFSQFromLib,
#       ResidualVectorQuantizerConfig
#   )
#   
#   vq_cfg = ResidualVectorQuantizerConfig(**config.vq_cfg)
#   if vq_cfg.vq_type == "ema":
#       self.vq = ResidualVQFromLib(vq_cfg)
#   elif vq_cfg.vq_type == "fsq":
#       self.vq = ResidualFSQFromLib(vq_cfg)
#   else:
#       self.vq = ResidualVectorQuantizer(vq_cfg)



