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
import os

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

import gr00t

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)


def qwen_vl_visual_module(eagle_model: nn.Module) -> nn.Module:
    """Vision tower for Qwen2.5-VL loaded via ``AutoModel`` (`Qwen2_5_VLModel` or wrapped)."""
    if getattr(eagle_model, "visual", None) is not None:
        return eagle_model.visual
    inner = getattr(eagle_model, "model", None)
    if inner is not None and getattr(inner, "visual", None) is not None:
        return inner.visual
    raise ValueError(
        "EagleBackboneUniT is Qwen2.5-VL only: expected eagle_model.visual "
        "or eagle_model.model.visual."
    )


class EagleBackboneUniT(nn.Module):

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str | None = None,
        project_to_dim: int = 1536,
        tune_bridge_embedding: bool = False,
        tokenizer_len: bool = None,
        tune_all_llm_embedding: bool = False,
        num_bridge_tokens: int = None,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        self.eagle_model = AutoModel.from_pretrained(eagle_path, trust_remote_code=True)

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        print(f"Selected LLM Layer: {select_layer}")
        if hasattr(self.eagle_model.language_model, "model"):
            while len(self.eagle_model.language_model.model.layers) > select_layer:
                self.eagle_model.language_model.model.layers.pop(-1)
        else:
            while len(self.eagle_model.language_model.layers) > select_layer:
                self.eagle_model.language_model.layers.pop(-1)

        self.select_layer = select_layer
        self.num_bridge_tokens = num_bridge_tokens
        self.set_trainable_parameters(
            tune_llm, tune_visual,
            tune_bridge_embedding, tokenizer_len,
            tune_all_llm_embedding,
        )

    def set_trainable_parameters(self,
            tune_llm: bool,
            tune_visual: bool,
            tune_bridge_embedding: bool,
            tokenizer_len: int = None,
            tune_all_llm_embedding: bool = False,
        ):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        if tune_visual:
            # Vision-tower finetuning is no longer supported. The previous code
            # path additionally required a DDP-unused-param workaround that is
            # now removed; re-enable it only after restoring proper DDP support.
            raise NotImplementedError(
                "EagleBackboneUniT no longer supports tune_visual=True. "
                "Set tune_visual=False (vision tower is always frozen)."
            )
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
        qwen_vl_visual_module(self.eagle_model).requires_grad_(False)
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")

        # Ensure a slot for the gradient-hook handle exists across re-entries.
        if not hasattr(self, "_embed_tokens_hook_handle"):
            self._embed_tokens_hook_handle = None

        if tune_bridge_embedding and (not tune_all_llm_embedding):
            if hasattr(self.eagle_model.language_model, "model"):
                embed_tokens = self.eagle_model.language_model.model.embed_tokens
            else:
                embed_tokens = self.eagle_model.language_model.embed_tokens

            embed_tokens.weight.requires_grad = True

            # Remove any previous hook before registering a new one.
            if self._embed_tokens_hook_handle is not None:
                self._embed_tokens_hook_handle.remove()
                self._embed_tokens_hook_handle = None

            if self.num_bridge_tokens is None:
                raise ValueError(
                    "EagleBackboneUniT.tune_bridge_embedding=True requires "
                    "num_bridge_tokens to be set. Pass it from the model's "
                    "unit_cfg.num_bridge_tokens at construction time."
                )
            bridge_token_ids = torch.arange(
                tokenizer_len - self.num_bridge_tokens, tokenizer_len,
                device=embed_tokens.weight.device,
            )
            print(f"start_bridge_token_id: {bridge_token_ids[0]}, end_bridge_token_id: {bridge_token_ids[-1]}")

            # [vocab_size, 1] mask broadcasts against embed_tokens.weight.grad.
            self._embed_tokens_hook_mask = torch.zeros(embed_tokens.weight.shape[0], device=embed_tokens.weight.device)
            self._embed_tokens_hook_mask[bridge_token_ids] = 1.0
            self._embed_tokens_hook_mask = self._embed_tokens_hook_mask.view(-1, 1)

            def grad_hook(grad):
                if self._embed_tokens_hook_mask.device != grad.device:
                    self._embed_tokens_hook_mask = self._embed_tokens_hook_mask.to(grad.device)
                return grad * self._embed_tokens_hook_mask

            self._embed_tokens_hook_handle = embed_tokens.weight.register_hook(grad_hook)

        else:
            # Drop a previously registered hook if any.
            if self._embed_tokens_hook_handle is not None:
                self._embed_tokens_hook_handle.remove()
                self._embed_tokens_hook_handle = None

        print(f"Tune backbone bridge embedding: {tune_bridge_embedding}")
        print(f"Tune all llm token embeddings: {tune_all_llm_embedding}")

        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if not self.tune_visual:
                qwen_vl_visual_module(self.eagle_model).eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward_eagle(self, vl_input: BatchFeature) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        if "image_sizes" in eagle_input:
            del eagle_input["image_sizes"]

        visual_mod = None
        hook_handle = None
        hook_captures: list[torch.Tensor] = []

        if eagle_input.get("pixel_values") is not None:
            visual_mod = qwen_vl_visual_module(self.eagle_model)

            def _visual_hook(_module, _inp, output):
                hook_captures.append(output)

            hook_handle = visual_mod.register_forward_hook(_visual_hook)

        try:
            eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        if len(hook_captures) > 1:
            raise RuntimeError(
                f"Multiple visual forward calls in one eagle pass ({len(hook_captures)}); "
                "cached_image_embeds is ambiguous. Extend policy for image+video mixes."
            )

        cached_image_embeds = hook_captures[0] if hook_captures else None

        # print(f"(eagle_output.hidden_states): {len(eagle_output.hidden_states)}")
        eagle_features = eagle_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        return eagle_features, eagle_input["attention_mask"], cached_image_embeds

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask, cached_image_embeds = self.forward_eagle(vl_input)

        out = {
            "backbone_features": eagle_embeds,
            "backbone_attention_mask": eagle_mask,
        }
        if cached_image_embeds is not None:
            out["cached_image_embeds"] = cached_image_embeds
        return BatchFeature(data=out)  # [B, T2, hidden_size]
