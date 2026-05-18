"""
Visual-Action Fusion Module (Q-Former based, three-path soft routing).
"""
import torch
import torch.nn as nn
from typing import Optional


class QFormerVisualActionFusion(nn.Module):
    """
    Three-Path Visual-Action Fusion with M-Former
    
    Architecture:
    1. Path 1 (visual-only): Direct visual processing for pv=1, pa=0
    2. Path 2 (action-only): Direct action processing for pv=0, pa=1
    3. Path 3 (fused): M-Former fusion for pv=1, pa=1
    
    Key improvements:
    - Uses M-Former's cond/target interface (same as ActionEncoder)
    - Avoids complex key-value sequence construction
    - M-Former handles positional encoding and token type internally
    - Three-way routing based on presence flags
    """
    
    def __init__(self, 
                 hidden_size: int = 768,
                 num_heads: int = 8,
                 query_num: int = 8,
                 dropout: float = 0.1,
                 fusion_config: Optional[dict] = None):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.query_num = query_num
        self.dropout = dropout
        self.fusion_config = fusion_config or {}
        
        # M-Former specific configs (for fused path)
        self.num_layers = int(self.fusion_config.get('num_layers', 4))  # Increase default layers
        
        # === Path 1: Visual-only processing ===
        self.visual_only_path = nn.Identity()  # Visual tokens already processed by M-Former
        
        # === Path 2: Action-only processing ===
        self.action_only_path = nn.Identity()  # Action tokens already processed by A-Former
        
        # === Path 3: M-Former fusion (for both modalities) ===
        # Use M-Former with cond/target interface (same as ActionEncoder)
        from transformers.models.vit.modeling_vit import ViTConfig
        from .m_former import MFormer
        
        # M-Former sequence: [query_tokens, cond, sep, target]
        # = query_num + query_num (visual) + 1 (sep) + query_num (action)
        m_former_max_seq_len = query_num + query_num + 1 + query_num
        
        # Get M-Former config from fusion_config or use defaults
        m_former_cfg = self.fusion_config.get('m_former_cfg', None)
        if m_former_cfg is not None:
            m_former_vit_config = ViTConfig(**m_former_cfg)
            # Override critical parameters
            m_former_vit_config.query_num = query_num
            m_former_vit_config.hidden_size = hidden_size
            m_former_vit_config.input_hidden_size = hidden_size
            m_former_vit_config.max_seq_len = m_former_max_seq_len
            m_former_vit_config.num_patches = query_num  # For backward compatibility
        else:
            # Default M-Former config
            m_former_vit_config = ViTConfig(
                hidden_size=hidden_size,
                num_hidden_layers=self.num_layers,
                num_attention_heads=max(1, hidden_size // 64),  # Standard head_dim=64
                intermediate_size=hidden_size * 4,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
                query_num=query_num,
                input_hidden_size=hidden_size,
                max_seq_len=m_former_max_seq_len,
                num_patches=query_num,
            )
        
        self.m_former = MFormer(m_former_vit_config)
        
        # === Three independent alignment layers ===
        self.align_visual = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.align_action = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.align_fused = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        
        # === Shared projection (final alignment) ===
        self.shared_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
    
    def _compute_mformer_path(self, visual_tokens, action_tokens):
        """
        Compute the fused path using M-Former
        Uses visual as condition, action as target (following ActionEncoder pattern)
        """
        # M-Former expects:
        # - cond_hidden_states: (B, query_num, hidden_size) - visual tokens
        # - target_hidden_states: (B, query_num, hidden_size) - action tokens
        # M-Former will handle positional encoding and token type internally
        m_former_output = self.m_former(
            cond_hidden_states=visual_tokens,    # (B, query_num, H)
            target_hidden_states=action_tokens   # (B, query_num, H)
        )
        # m_former_output.last_hidden_state: (B, query_num + query_num + 1 + query_num, H)
        # Layout: [query_tokens, cond, sep, target]
        # Extract the query tokens (first query_num tokens)
        fused_tokens = m_former_output.last_hidden_state[:, :self.query_num, :]
        
        return fused_tokens

    def forward(self,
                visual_tokens: torch.Tensor,   # (B, query_num, hidden_size)
                action_tokens: torch.Tensor,   # (B, query_num, hidden_size)
                pv: torch.Tensor,              # (B,) visual presence
                pa: torch.Tensor               # (B,) action presence
                ) -> torch.Tensor:
        """
        Three-path forward pass with routing
        
        Args:
            visual_tokens: Visual features from M-Former
            action_tokens: Action features from A-Former  
            pv: Visual presence flags (0 or 1)
            pa: Action presence flags (0 or 1)
            
        Returns:
            output: (B, query_num, hidden_size) fused features via three-way routing
        """
        B = visual_tokens.shape[0]
        
        # === Path 1: Visual-only ===
        v_only = self.visual_only_path(visual_tokens)
        v_only = self.align_visual(v_only)
        
        # === Path 2: Action-only ===
        a_only = self.action_only_path(action_tokens)
        a_only = self.align_action(a_only)
        
        # === Path 3: Fused (M-Former) ===
        # Always compute fused path to ensure all parameters receive gradients
        fused = self._compute_mformer_path(visual_tokens, action_tokens)
        fused = self.align_fused(fused)
        
        # === Shared projection for all three paths ===
        v_only = self.shared_projection(v_only)
        a_only = self.shared_projection(a_only)
        fused = self.shared_projection(fused)
        
        # === Three-way soft routing ===
        pv_f = pv.view(B, 1, 1).float()
        pa_f = pa.view(B, 1, 1).float()
        
        w_both = pv_f * pa_f              # pv=1, pa=1 -> use fused
        w_vis = pv_f * (1.0 - pa_f)       # pv=1, pa=0 -> use visual-only
        w_act = (1.0 - pv_f) * pa_f       # pv=0, pa=1 -> use action-only
        
        output = w_both * fused + w_vis * v_only + w_act * a_only
        
        return output
    
    def get_fusion_info(self) -> dict:
        """Get Three-Path M-Former fusion module information"""
        return {
            "type": "ThreePathMFormerFusion",
            "architecture": "visual_only | action_only | mformer_fused",
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "query_num": self.query_num,
            "num_layers": self.num_layers,
            "m_former_max_seq_len": self.query_num * 3 + 1,  # query + visual + sep + action
            "has_align_layers": True,
            "has_shared_projection": True,
            "routing_strategy": "soft_three_way",
            "paths": {
                "visual_only": "pv=1, pa=0 -> direct visual",
                "action_only": "pv=0, pa=1 -> direct action",
                "fused": "pv=1, pa=1 -> M-Former(visual_cond, action_target)"
            },
            "fusion_method": "M-Former with cond/target interface (same as ActionEncoder)"
        }
