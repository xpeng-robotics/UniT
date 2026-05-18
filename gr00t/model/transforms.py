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

import random
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tree
from einops import rearrange
from PIL import Image
from pydantic import Field, PrivateAttr
from transformers import AutoProcessor, ProcessorMixin
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import InvertibleModalityTransform

from .backbone.eagle_backbone import DEFAULT_EAGLE_PATH

def _make_bridge_tokens(num_tokens: int) -> list:
    """Pure helper: build the list ``["<|bridge_0|>", ..., "<|bridge_{N-1}|>"]``.

    No global state, no side effects. ``num_tokens`` must be a positive int;
    the authoritative value is the model's ``unit_cfg.num_bridge_tokens``.
    """
    if not isinstance(num_tokens, int) or num_tokens <= 0:
        raise ValueError(
            f"num_bridge_tokens must be a positive int, got {num_tokens!r}. "
            "The authoritative value comes from the model config "
            "(``unit_cfg.num_bridge_tokens``); upstream code must pass it explicitly."
        )
    return [f"<|bridge_{i}|>" for i in range(num_tokens)]


def get_bridge_str(num_bridge_tokens: int) -> str:
    """Return the prompt-suffix string holding ``num_bridge_tokens`` bridge specials.

    ``num_bridge_tokens`` is required (no default). Callers must read it from
    the model config and pass it explicitly so that the prompt suffix and the
    backbone slice stay in lockstep.
    """
    return "".join(_make_bridge_tokens(num_bridge_tokens))

def formalize_language(language: str) -> str:
    """
    1. Force lowercase
    2. Remove all punctuations
    """
    language = language.lower()
    language = re.sub(r"[^\w\s]", "", language)
    return language


def build_eagle_processor(eagle_path: str, num_bridge_tokens: Optional[int] = None) -> ProcessorMixin:
    """Build eagle processor and (optionally) register bridge special tokens.

    Args:
        eagle_path: Path to eagle model.
        num_bridge_tokens: If a positive int, register that many ``<|bridge_i|>``
            special tokens on the tokenizer. If ``None``, no bridge tokens are
            registered (vanilla processor); use this only when bridge prompting
            is not in play (e.g. tokenizer-only training, or generic collators
            that never collate ``eagle_content``).
    """
    eagle_processor = AutoProcessor.from_pretrained(
        eagle_path, trust_remote_code=True, use_fast=True
    )
    eagle_processor.tokenizer.padding_side = "left"

    # Qwen2_5_VLProcessor: attach process_vision_info from qwen_vl_utils.
    if eagle_processor.__class__.__name__ == "Qwen2_5_VLProcessor":
        from qwen_vl_utils import process_vision_info
        eagle_processor.process_vision_info = process_vision_info

    if num_bridge_tokens is not None:
        tokens_to_add = _make_bridge_tokens(num_bridge_tokens)
        eagle_processor.tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    return eagle_processor


def collate(features: List[dict], eagle_processor) -> dict:
    batch = {}
    keys = features[0].keys()
    # print(keys)
    # print(eagle_processor.__class__)
    # print(features)

    for key in keys:
        values = [elem[key] for elem in features]

        if key == "eagle_content":
            text_list = []
            image_inputs = []
            for v in values:
                curr_text_list = v["text_list"]
                curr_image_inputs = v["image_inputs"]
                text_list += curr_text_list
                image_inputs += curr_image_inputs
            # print(text_list)
            # print(image_inputs)
            eagle_inputs = eagle_processor(
                text=text_list, images=image_inputs, return_tensors="pt", padding=True
            )
            for k, v in eagle_inputs.items():
                k = "eagle_" + k
                batch[k] = v
                # print(k, v.shape)

        elif key == "goal_images":
            image_inputs = eagle_processor.image_processor(
                images=values,
                videos=None,
                return_tensors="pt"
            )
            for k, v in image_inputs.items():
                k = "goal_image_" + k
                batch[k] = v
                # print(k, v.shape)
            
        elif key in ("imagenet_obs_images", "imagenet_goal_images"):
            # Stack ImageNet preprocessed images
            # values is a list of tensors [V*T, C, H, W]
            batch[key] = torch.stack(values, dim=0)  # [B, V*T, C, H, W]

        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        elif key == "orig_state":
            batch[key] = values
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))
    return batch


class DefaultDataCollator(DataCollatorMixin):
    def __init__(
        self,
        eagle_path: str = DEFAULT_EAGLE_PATH,
        num_bridge_tokens: Optional[int] = None,
        eagle_processor: Optional[ProcessorMixin] = None,
    ):
        """
        Args:
            eagle_path: Path used to build a processor when ``eagle_processor``
                is not provided.
            num_bridge_tokens: Bridge specials to register when building a
                processor locally. Must match the value used by the upstream
                ``EagleProcessTransform`` to keep training and inference
                tokenization aligned.
            eagle_processor: Already-built processor instance. When set, the
                collator reuses it as-is (preferred path: pass the transform's
                processor here so train/eval share one tokenizer state).
        """
        super().__init__()
        if eagle_processor is not None:
            self.eagle_processor = eagle_processor
        else:
            self.eagle_processor = build_eagle_processor(eagle_path, num_bridge_tokens)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate(features, self.eagle_processor)


class GR00TTransform(InvertibleModalityTransform):

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    formalize_language: bool = Field(default=False, description="Formalize language if True.")
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    eagle_path: str = DEFAULT_EAGLE_PATH

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[list[str]] = PrivateAttr(default=None)

    # eagle_processor: ProcessorMixin = Field(default=build_eagle_processor(eagle_path))
    eagle_processor: ProcessorMixin | None = None  # lazy init when building the processor stack
    fix_language: str | None = None
    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    max_state_dim: int
    max_action_dim: int
    state_horizon: int
    action_horizon: int

    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None

    use_bridge: bool = False
    ignore_lang_prefix: bool = False

    # Number of bridge tokens to register on the eagle tokenizer / append to
    # the prompt. The authoritative value lives in the model config
    # (``unit_cfg.num_bridge_tokens``); upstream code reads it from there
    # and passes it in. ``None`` is only valid when ``use_bridge=False``.
    num_bridge_tokens: int | None = None

    def model_post_init(self, __context):
        """Called by Pydantic after model init."""
        print(f"transform.eagle_path: {self.eagle_path}")
        print(f"transform.use_bridge: {self.use_bridge}")
        print(f"transform.ignore_lang_prefix: {self.ignore_lang_prefix}")
        if self.use_bridge and self.num_bridge_tokens is None:
            raise ValueError(
                "use_bridge=True requires num_bridge_tokens to be set. "
                "Read it from the model config (unit_cfg.num_bridge_tokens) "
                "and forward it via load_data_config(num_bridge_tokens=...)."
            )
        if self.num_bridge_tokens is not None:
            print(f"transform.num_bridge_tokens: {self.num_bridge_tokens}")
        if self.eagle_processor is None:
            self.eagle_processor = build_eagle_processor(self.eagle_path, self.num_bridge_tokens)

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            if "annotation" in key:
                modality = "language"
            else:
                try:
                    modality, _ = key.split(".")
                except:  # noqa: E722
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)
        # Use video key to determine batch size.
        video_ndim = data["video"].ndim
        if video_ndim == 5:  # Interpret as [T, V, H, W, C]
            is_batched = False
            batch_size = 1
        elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
            is_batched = True
            batch_size = data["video"].shape[0]
        else:
            raise ValueError(f"Unsupported video number of dimensions: {video_ndim}")

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            assert len(language_keys) == 1, f"{language_keys=}"
            self._language_key = language_keys[0]
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch:
                video: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """
        # TODO(YL, FH): check if this is correct
        images = batch["images"]  # [V, T, C, H, W]
        images.shape[0]

        np_images = rearrange(images, "v t c h w -> (t v) c h w")
        text_content = []

        # handle language
        lang = batch["language"]
        if isinstance(lang, list):
            lang = lang[0]
        text_content.append({"type": "text", "text": lang})

        eagle_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
        eagle_image = [{"type": "image", "image": img} for img in eagle_images]
        eagle_conversation = [
            {
                "role": "user",
                "content": eagle_image + text_content,
            }
        ]

        text_list = [
            self.eagle_processor.apply_chat_template(
                eagle_conversation, tokenize=False, add_generation_prompt=True
            )
        ]

        if self.use_bridge:
            # model_post_init enforces num_bridge_tokens is set when use_bridge=True;
            # this assert narrows the type and acts as a runtime sanity check.
            assert self.num_bridge_tokens is not None
            bridge_str = get_bridge_str(self.num_bridge_tokens)
            text_list = [text + bridge_str for text in text_list]

        image_inputs, video_inputs = self.eagle_processor.process_vision_info(eagle_conversation)
        eagle_content = {
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "text_list": text_list,
        }
        inputs = {}
        inputs["eagle_content"] = eagle_content
        return inputs

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        ## TODO(YL, FH): check if this is correct
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        # print(data["video"].shape, images.shape)
        return images

    def _prepare_language(self, data: dict):
        if self.fix_language is not None:
            # print(self.fix_language)
            return self.fix_language

        """Tokenize data['language'] (or default_instruction if missing)."""
        # print(f"self._language_key: {self._language_key}")

        if self._language_key is not None:
            raw_language = data[self._language_key]
            # print(f"type(raw_language): {type(raw_language)} raw_language: {raw_language}")
            if isinstance(raw_language, list) or isinstance(raw_language, np.ndarray ):
                raw_language = str(raw_language[0])

            # Language dropout
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction
        else:
            raw_language = self.default_instruction

        if self.ignore_lang_prefix:
            # print(f"raw_language: {raw_language}")
            raw_language = raw_language.split(":")[-1].strip()
            # print(raw_language)

        return raw_language

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        assert state.shape[0] == self.state_horizon, f"{state.shape=}, {self.state_horizon=}"

        n_state_dims = state.shape[-1]

        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")

        # Create mask for real state dims
        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros((self.action_horizon, self.max_action_dim), dtype=bool)
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        assert actions.shape[0] == self.action_horizon, f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens

    def apply_single(self, data: dict) -> dict:
        transformed_data = {}

        # 1) Prepare video and language with vlm processing.
        images = self._prepare_video(data)
        images = images.astype(np.uint8)
        language = self._prepare_language(data)
        batch_data = {"images": images, "language": language}
        vlm_outputs = self._apply_vlm_processing(batch_data)

        # 2) Prepare state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            # 3) Prepare actions
            transformed_data["segmentation_target"] = np.zeros((2,))
            transformed_data["segmentation_target_mask"] = np.zeros((1,))
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.training:
            action_and_mask_keys = ["action", "action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        data_split = [tree.map_structure(lambda x: x[i], data) for i in range(batch_size)]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return collate(data_split_processed, self.eagle_processor)

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            processed_data = self.apply_batch(data, batch_size)
        else:
            processed_data = self.apply_single(data)
        # Only pass orig_state during inference (training collate cannot handle dict)
        if not self.training and "orig_state" in data:
            processed_data["orig_state"] = data["orig_state"]

        return processed_data
        
    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)


class GR00TTransformWithGoalImage(GR00TTransform):

    enable_imagenet_preprocessing: bool = Field(
        default=False,
        description="Whether to enable ImageNet preprocessing for obs and goal images."
    )

    vision_model_type: str = Field(
        default="dinov2",
        description="The type of vision model to use."
    )

    tokenizer_only: bool = Field(
        default=False,
        description=(
            "If True, skip the eagle backbone-side processing entirely: "
            "no eagle processor instantiation, no VLM tokenisation, no "
            "<|bridge_i|> append, no goal-image eagle preprocessing. "
            "Only ImageNet-preprocessed images + state/action are produced. "
            "Use this for tokenizer-only training/evaluation where the "
            "eagle backbone is never invoked."
        ),
    )

    def model_post_init(self, __context):
        """Pydantic post-init hook (overrides parent).

        When ``tokenizer_only`` is True, skip ``build_eagle_processor`` and
        leave ``self.eagle_processor`` as None; ``apply_single`` short-circuits
        the eagle paths anyway.
        """
        print(f"transform.eagle_path: {self.eagle_path}")
        print(f"transform.use_bridge: {self.use_bridge}")
        print(f"transform.ignore_lang_prefix: {self.ignore_lang_prefix}")
        print(f"transform.tokenizer_only: {self.tokenizer_only}")
        if self.num_bridge_tokens is not None:
            print(f"transform.num_bridge_tokens: {self.num_bridge_tokens}")
        if self.tokenizer_only:
            assert not self.use_bridge, (
                "tokenizer_only=True is incompatible with use_bridge=True: "
                "bridge tokens are appended via the eagle path which is disabled."
            )
            return
        if self.eagle_processor is None:
            self.eagle_processor = build_eagle_processor(self.eagle_path, self.num_bridge_tokens)

    def _apply_imagenet_preprocessing(self, images):
        """
        Apply ImageNet standard preprocessing to images.
        Args:
            images: numpy array [V, T, C, H, W], uint8, range [0, 255]
        Returns:
            torch.Tensor [V*T, C, H, W], float32, ImageNet normalized
        """
        import torchvision.transforms as T
        
        # ImageNet normalization statistics
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
        imagenet_std = torch.tensor([0.229, 0.224, 0.225])
        
        # Rearrange to [V*T, C, H, W]
        np_images = rearrange(images, "v t c h w -> (v t) c h w")
        
        # Convert to tensor and normalize to [0, 1]
        images_tensor = torch.from_numpy(np_images).float() / 255.0
        
        # Apply ImageNet normalization
        # images_tensor shape: [V*T, C, H, W]
        imagenet_mean = imagenet_mean.view(1, 3, 1, 1)
        imagenet_std = imagenet_std.view(1, 3, 1, 1)
        images_tensor = (images_tensor - imagenet_mean) / imagenet_std
        
        return images_tensor

    def _apply_goal_image_processing(self, goal_images) -> BatchFeature:
        """
        Args:
            goal_images: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """
        np_images = rearrange(goal_images, "v t c h w -> (t v) c h w")
        text_content = []

        eagle_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
        eagle_image = [{"type": "image", "image": img} for img in eagle_images]
        eagle_conversation = [
            {
                "role": "user",
                "content": eagle_image,
            }
        ]
        # print(type(self.eagle_processor), type(self.eagle_processor.process_vision_info))
        image_inputs, video_inputs = self.eagle_processor.process_vision_info(eagle_conversation)
        return image_inputs

    def apply_single(self, data: dict) -> dict:
        # for k, v in data.items():
        #     if type(v) == torch.Tensor:
        #         print(k, v.shape)
        #     else:
        #         print(k, v)

        transformed_data = {}

        # 1) Prepare video and language with vlm processing.
        images = self._prepare_video(data)
        images = images.astype(np.uint8)
        language = self._prepare_language(data)

        # Split stacked video into observation frames vs goal frame (last). This assumes T>1
        # implies a goal frame; does not model "video-only observation" stacks yet.
        if images.shape[1] > 1:
            obs_images = images[:,:-1]
        else:
            obs_images = images

        batch_data = {"images": obs_images, "language": language}
        if self.tokenizer_only:
            # Skip eagle backbone path entirely; tokenizer training only
            # consumes ImageNet-preprocessed images + state/action.
            vlm_outputs = {}
        else:
            vlm_outputs = self._apply_vlm_processing(batch_data)

        # 2) Prepare state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        # if self.training:
        # 3) Prepare actions
        # if self.training:
        transformed_data["segmentation_target"] = np.zeros((2,))
        transformed_data["segmentation_target_mask"] = np.zeros((1,))
        transformed_data["has_real_action"] = np.ones((), dtype=bool)
        actions, actions_mask, _ = self._prepare_action(data)
        transformed_data["action"] = actions
        transformed_data["action_mask"] = actions_mask

        # if self.vision_model_type != "dinov2":
        for k, v in vlm_outputs.items():
            assert k not in transformed_data, f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        # Same split as above: last frame is goal when T>1.
        if images.shape[1] > 1:
            goal_images = images[:,-1:]
            
            # Apply ImageNet preprocessing if enabled (before VIT preprocessing)
            if self.enable_imagenet_preprocessing:
                imagenet_goal_images = self._apply_imagenet_preprocessing(goal_images)
                transformed_data["imagenet_goal_images"] = imagenet_goal_images
                
                imagenet_obs_images = self._apply_imagenet_preprocessing(obs_images)
                transformed_data["imagenet_obs_images"] = imagenet_obs_images
            
            if not self.tokenizer_only:
                goal_images = self._apply_goal_image_processing(goal_images)
                transformed_data["goal_images"] = goal_images

        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        # if self.training:
        action_and_mask_keys = ["action", "action_mask"]
        assert all(
            transformed_data[key].shape == transformed_data["action"].shape
            for key in action_and_mask_keys
        ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

        return transformed_data

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        ## TODO(YL, FH): check if this is correct
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        return images