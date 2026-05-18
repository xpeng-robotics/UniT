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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
    CoordinateTransform,
    StateActionSubtractPosition,
    StateActionRotateEuler,
    DropKeys,
    HierarchicalRelativeTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
    VideoOffsetCrop,
    VideoHorizontalFlip,
)
from gr00t.model.transforms import GR00TTransform, GR00TTransformWithGoalImage
from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH

@dataclass
class BaseDataConfig(ABC):
    eagle_path = DEFAULT_EAGLE_PATH
    use_bridge = False

    def __init__(
        self,
        eagle_path: str = DEFAULT_EAGLE_PATH,
        use_bridge: bool = False,
        ignore_lang_prefix: bool = False,
        enable_imagenet_preprocessing: bool = True,
        num_bridge_tokens: int = None,
        tokenizer_only: bool = False,
    ):
        self.eagle_path = eagle_path
        self.use_bridge = use_bridge
        self.ignore_lang_prefix = ignore_lang_prefix
        self.enable_imagenet_preprocessing = enable_imagenet_preprocessing
        self.tokenizer_only = tokenizer_only
        # num_bridge_tokens is purely a transport channel here. The authoritative
        # value lives in the model config (``unit_cfg.num_bridge_tokens``);
        # training scripts read it from there and pass it in. ``None`` is only
        # valid when ``use_bridge=False`` (the transform layer enforces this).
        self.num_bridge_tokens = num_bridge_tokens

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.video_delta_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


#####################################################################################
# helper functions
#####################################################################################


def import_external_data_config(data_config_str: str) -> Optional[BaseDataConfig]:
    """
    Import and instantiate an external data configuration class.

    Format: "module_path:ClassName" (e.g., "my_configs:RobotConfig")
    Supports nested modules like "package.submodule:ClassName"
    """
    if ":" not in data_config_str:
        return None

    import importlib
    import os
    import sys
    from pathlib import Path

    # Add current working directory to Python path
    current_dir = str(Path(os.getcwd()).absolute())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    try:
        module_path, class_name = data_config_str.split(":", 1)
        if not module_path or not class_name:
            raise ValueError(f"Invalid format: '{data_config_str}'. Use 'module:ClassName'")

        print(f"Loading external config: {module_path}.{class_name}")

        module = importlib.import_module(module_path)
        if not hasattr(module, class_name):
            available = [
                n
                for n in dir(module)
                if not n.startswith("_") and isinstance(getattr(module, n), type)
            ]
            raise AttributeError(
                f"Class '{class_name}' not found in '{module_path}'. Available: {available}"
            )

        # assert if the class has 'transform' and 'modality_config' methods
        if not hasattr(getattr(module, class_name), "transform"):
            raise AttributeError(f"Class '{class_name}' does not have a 'transform' method")
        if not hasattr(getattr(module, class_name), "modality_config"):
            raise AttributeError(f"Class '{class_name}' does not have a 'modality_config' method")

        return getattr(module, class_name)()

    except (ModuleNotFoundError, AttributeError, ValueError) as e:
        print(f"Config loading failed: {e}")
        print("Example: my_configs:MyConfig, package.submodule:ClassName")
        raise


def load_data_config(
    data_config_str: str,
    eagle_path: str = DEFAULT_EAGLE_PATH,
    use_bridge: bool = False,
    ignore_lang_prefix: bool = False,
    enable_imagenet_preprocessing: bool = False,
    num_bridge_tokens: int = None,
    tokenizer_only: bool = False,
) -> BaseDataConfig:
    """
    Get a data config class from a string.
    >>> load_data_config("so100")
    >>> get_data_config("dir.subdir.my_configs:RobotConfig")

    Args:
        num_bridge_tokens: Number of bridge tokens. Read from
            ``model_config.unit_cfg['num_bridge_tokens']`` by the training
            script and forwarded here. Required (non-None) when
            ``use_bridge=True``; otherwise the transform layer raises.
        tokenizer_only: If True, the data transform skips eagle backbone-side
            processing (VLM tokenisation, bridge-token append, goal-image eagle
            preprocessing). Use this for tokenizer-only training/evaluation.
    """
    if data_config_str in DATA_CONFIG_MAP:
        return DATA_CONFIG_MAP[data_config_str](
            eagle_path=eagle_path,
            use_bridge=use_bridge,
            ignore_lang_prefix=ignore_lang_prefix,
            enable_imagenet_preprocessing=enable_imagenet_preprocessing,
            num_bridge_tokens=num_bridge_tokens,
            tokenizer_only=tokenizer_only,
        )
    data_config_cls = import_external_data_config(data_config_str)
    if data_config_cls is not None:
        return data_config_cls
    # Yellow warning color
    yellow = "\033[93m"
    reset = "\033[0m"
    raise ValueError(
        f"{yellow}Invalid data_config '{data_config_str}'. "
        f"Available options: {list(DATA_CONFIG_MAP.keys())}, "
        f"or use 'module:ClassName' for external configs{reset}"
    )


###########################################################################################




class So100DataConfig(BaseDataConfig):
    video_keys = ["video.webcam"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    def transform(self) -> ModalityTransform:
        print(f"eagle_path: {self.eagle_path}")
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
                eagle_path=self.eagle_path,
                use_bridge=self.use_bridge, 
                ignore_lang_prefix=self.ignore_lang_prefix,
                enable_imagenet_preprocessing=self.enable_imagenet_preprocessing
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class So100DualCamDataConfig(So100DataConfig):
    video_keys = ["video.front", "video.wrist"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))


###########################################################################################


class UnitreeG1DataConfig(BaseDataConfig):
    video_keys = ["video.rs_view"]
    state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
                eagle_path=self.eagle_path,
                use_bridge=self.use_bridge, 
                ignore_lang_prefix=self.ignore_lang_prefix,
                enable_imagenet_preprocessing=self.enable_imagenet_preprocessing
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class UnitreeG1FullBodyDataConfig(UnitreeG1DataConfig):
    video_keys = ["video.rs_view"]
    state_keys = [
        "state.left_leg",
        "state.right_leg",
        "state.waist",
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))


###########################################################################################




class BimanualPandaGripperDataConfig(BaseDataConfig):
    video_keys = [
        "video.rightHand_view",
        "video.leftHand_view",
        "video.front_view",
    ]
    state_keys = [
        "state.right_arm_eef_pos",
        "state.right_arm_eef_quat",
        "state.right_gripper_qpos",
        "state.left_arm_eef_pos",
        "state.left_arm_eef_quat",
        "state.left_gripper_qpos",
    ]
    action_keys = [
        "action.right_arm_eef_pos",
        "action.right_arm_eef_rot",
        "action.right_gripper_close",
        "action.left_arm_eef_pos",
        "action.left_arm_eef_rot",
        "action.left_gripper_close",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.right_arm_eef_pos": "min_max",
        "state.right_gripper_qpos": "min_max",
        "state.left_arm_eef_pos": "min_max",
        "state.left_gripper_qpos": "min_max",
    }
    state_target_rotations = {
        "state.right_arm_eef_quat": "rotation_6d",
        "state.left_arm_eef_quat": "rotation_6d",
    }
    action_normalization_modes = {
        "action.right_gripper_close": "binary",
        "action.left_gripper_close": "binary",
    }

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
                target_rotations=self.state_target_rotations,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
                eagle_path=self.eagle_path,
                use_bridge=self.use_bridge, 
                ignore_lang_prefix=self.ignore_lang_prefix,
                enable_imagenet_preprocessing=self.enable_imagenet_preprocessing
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class BimanualPandaHandDataConfig(BimanualPandaGripperDataConfig):
    video_keys = [
        "video.rightHand_view",
        "video.leftHand_view",
        "video.ego_view",
    ]
    state_keys = [
        "state.right_arm_eef_pos",
        "state.right_arm_eef_quat",
        "state.right_hand",
        "state.left_arm_eef_pos",
        "state.left_arm_eef_quat",
        "state.left_hand",
    ]
    action_keys = [
        "action.right_arm_eef_pos",
        "action.right_arm_eef_rot",
        "action.right_hand",
        "action.left_arm_eef_pos",
        "action.left_arm_eef_rot",
        "action.left_hand",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.right_arm_eef_pos": "min_max",
        "state.right_hand": "min_max",
        "state.left_arm_eef_pos": "min_max",
        "state.left_hand": "min_max",
    }
    action_normalization_modes = {
        "action.right_hand": "min_max",
        "action.left_hand": "min_max",
    }
    state_target_rotations = {
        "state.right_arm_eef_quat": "rotation_6d",
        "state.left_arm_eef_quat": "rotation_6d",
    }


###########################################################################################


class SinglePandaGripperDataConfig(BimanualPandaGripperDataConfig):
    video_keys = [
        "video.left_view",
        "video.right_view",
        "video.wrist_view",
    ]
    state_keys = [
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    # Used in StateActionTransform for normalization and target rotations
    state_normalization_modes = {
        "state.end_effector_position_relative": "min_max",
        "state.end_effector_rotation_relative": "min_max",
        "state.gripper_qpos": "min_max",
        "state.base_position": "min_max",
        "state.base_rotation": "min_max",
    }
    state_target_rotations = {
        "state.end_effector_rotation_relative": "rotation_6d",
        "state.base_rotation": "rotation_6d",
    }
    action_normalization_modes = {
        "action.end_effector_position": "min_max",
        "action.end_effector_rotation": "min_max",
        "action.gripper_close": "binary",
        "action.base_motion": "min_max",
        "action.control_mode": "binary",
    }


###########################################################################################






class OxeDroidDataConfig(BaseDataConfig):
    video_keys = [
        "video.exterior_image_1",
        "video.exterior_image_2",
        "video.wrist_image",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_position",
    ]
    action_keys = [
        "action.eef_position_delta",
        "action.eef_rotation_delta",
        "action.gripper_position",
    ]
    language_keys = ["annotation.language.language_instruction"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.gripper_position": "min_max",
                },
                target_rotations={
                    "state.eef_rotation": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.gripper_position": "binary",
                },
                target_rotations={"action.eef_rotation_delta": "axis_angle"},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
                eagle_path=self.eagle_path,
                use_bridge=self.use_bridge, 
                ignore_lang_prefix=self.ignore_lang_prefix,
                enable_imagenet_preprocessing=self.enable_imagenet_preprocessing
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)
###########################################################################################
class OxeDroidSingleCamAbsDataConfig(BaseDataConfig):
    """DROID config with single camera and absolute action (state as action target).
    
    Changes from OxeDroidDataConfig:
    1. Only use exterior_image_1 (single camera)
    2. Action keys use absolute position (same as state), not delta
    """
    video_keys = [
        "video.exterior_image_1",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_position",
    ]
    # Action uses same keys as state (absolute position, not delta)
    action_keys = [
        "action.eef_position",
        "action.eef_rotation",
        "action.gripper_position",
    ]
    language_keys = ["annotation.language.language_instruction"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.gripper_position": "min_max",
                },
                target_rotations={
                    "state.eef_rotation": "rotation_6d",
                },
            ),
            # action transforms (same as state: absolute position)
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.eef_position": "min_max",
                    "action.gripper_position": "min_max",
                },
                target_rotations={
                    "action.eef_rotation": "rotation_6d",
                },
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=128,
                max_action_dim=128,
                eagle_path=self.eagle_path,
                use_bridge=self.use_bridge, 
                ignore_lang_prefix=self.ignore_lang_prefix,
                enable_imagenet_preprocessing=True,
                vision_model_type="dinov2"
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class OxeDroidSingleCamStateAsActionDataConfig(BaseDataConfig):
    """DROID config with single camera and absolute action (state as action target).
    
    Changes from OxeDroidDataConfig:
    1. Only use exterior_image_1 (single camera)
    2. Action keys use absolute position (same as state), not delta
    """
    video_keys = [
        "video.exterior_image_1",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_position",
    ]
    # Action uses same keys as state (absolute position, not delta)
    action_keys = [
        "action.eef_position",
        "action.eef_rotation",
        "action.gripper_position",
    ]
    language_keys = ["annotation.language.language_instruction"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "mean_std",
                    "state.gripper_position": "mean_std",
                },
                target_rotations={
                    "state.eef_rotation": "rotation_6d",
                },
            ),
            # action transforms (same as state: absolute position)
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.eef_position": "mean_std",
                    "action.gripper_position": "mean_std",
                },
                target_rotations={
                    "action.eef_rotation": "rotation_6d",
                },
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=128,
                max_action_dim=128,
                eagle_path=self.eagle_path,
                use_bridge=self.use_bridge, 
                ignore_lang_prefix=self.ignore_lang_prefix,
                enable_imagenet_preprocessing=True,
                vision_model_type="dinov2"
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)



class AgibotGenie1DataConfig(BaseDataConfig):
    video_keys = [
        "video.top_head",
        "video.hand_left",
        "video.hand_right",
    ]
    state_keys = [
        "state.left_arm_joint_position",
        "state.right_arm_joint_position",
        "state.left_effector_position",
        "state.right_effector_position",
        "state.head_position",
        "state.waist_position",
    ]
    action_keys = [
        "action.left_arm_joint_position",
        "action.right_arm_joint_position",
        "action.left_effector_position",
        "action.right_effector_position",
        "action.head_position",
        "action.waist_position",
        "action.robot_velocity",
    ]
    language_keys = ["annotation.language.action_text"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
                eagle_path=self.eagle_path,
                use_bridge=self.use_bridge, 
                ignore_lang_prefix=self.ignore_lang_prefix,
                enable_imagenet_preprocessing=self.enable_imagenet_preprocessing
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################





































































class EgoDexCameraHandGausNorm448CamEgoDataConfig(BaseDataConfig):
    """EgoDex camera + hand (wrist + finger tips) at 448 source resolution.

    Self-contained: depends only on BaseDataConfig. mean_std normalization on
    every state/action key; no CoordinateTransform.
    """

    video_keys = ["video.ego_view"]
    state_keys = [
        "state.camera_pos",
        "state.camera_rot",
        "state.rightHand_pos",
        "state.rightHand_rot",
        "state.rightThumbTip_pos",
        "state.rightIndexFingerTip_pos",
        "state.rightMiddleFingerTip_pos",
        "state.rightRingFingerTip_pos",
        "state.rightLittleFingerTip_pos",
        "state.leftHand_pos",
        "state.leftHand_rot",
        "state.leftThumbTip_pos",
        "state.leftIndexFingerTip_pos",
        "state.leftMiddleFingerTip_pos",
        "state.leftRingFingerTip_pos",
        "state.leftLittleFingerTip_pos",
    ]
    action_keys = [
        "action.camera_pos",
        "action.camera_rot",
        "action.rightHand_pos",
        "action.rightHand_rot",
        "action.rightThumbTip_pos",
        "action.rightIndexFingerTip_pos",
        "action.rightMiddleFingerTip_pos",
        "action.rightRingFingerTip_pos",
        "action.rightLittleFingerTip_pos",
        "action.leftHand_pos",
        "action.leftHand_rot",
        "action.leftThumbTip_pos",
        "action.leftIndexFingerTip_pos",
        "action.leftMiddleFingerTip_pos",
        "action.leftRingFingerTip_pos",
        "action.leftLittleFingerTip_pos",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    state_normalization_modes = {k: "mean_std" for k in state_keys}
    action_normalization_modes = {k: "mean_std" for k in action_keys}

    def transform(self) -> ModalityTransform:
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=128,
                max_action_dim=128,
                eagle_path=self.eagle_path,
                use_bridge=self.use_bridge,
                ignore_lang_prefix=self.ignore_lang_prefix,
                enable_imagenet_preprocessing=True,
                vision_model_type="dinov2",
                num_bridge_tokens=self.num_bridge_tokens,
                tokenizer_only=self.tokenizer_only,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class FourierGr1ArmsWaistGausNormCropCamEgoEefOnlyDataConfig(BaseDataConfig):
    """GR1 EEF-only (camera_egoview + wrist + finger pos), mean_std on every key."""

    video_keys = ["video.ego_view"]
    state_keys = [
        "state.camera_egoview_pos",
        "state.camera_egoview_rot6d",
        "state.wrist_r_pos",
        "state.wrist_r_rot6d",
        "state.thumb_r_pos",
        "state.index_r_pos",
        "state.middle_r_pos",
        "state.ring_r_pos",
        "state.pinky_r_pos",
        "state.wrist_l_pos",
        "state.wrist_l_rot6d",
        "state.thumb_l_pos",
        "state.index_l_pos",
        "state.middle_l_pos",
        "state.ring_l_pos",
        "state.pinky_l_pos",
    ]
    action_keys = [
        "action.camera_egoview_pos",
        "action.camera_egoview_rot6d",
        "action.wrist_r_pos",
        "action.wrist_r_rot6d",
        "action.thumb_r_pos",
        "action.index_r_pos",
        "action.middle_r_pos",
        "action.ring_r_pos",
        "action.pinky_r_pos",
        "action.wrist_l_pos",
        "action.wrist_l_rot6d",
        "action.thumb_l_pos",
        "action.index_l_pos",
        "action.middle_l_pos",
        "action.ring_l_pos",
        "action.pinky_l_pos",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    state_normalization_modes = {k: "mean_std" for k in state_keys}
    action_normalization_modes = {k: "mean_std" for k in action_keys}

    def transform(self) -> ModalityTransform:
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoOffsetCrop(
                apply_to=self.video_keys,
                top=int(256 * 0.17),
                left=0,
                height=int(256 * 0.66),
                width=256,
            ),
            VideoCrop(apply_to=self.video_keys, scale=0.95, height=168, width=256),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=128,
                max_action_dim=128,
                eagle_path=self.eagle_path,
                use_bridge=self.use_bridge,
                ignore_lang_prefix=self.ignore_lang_prefix,
                enable_imagenet_preprocessing=True,
                vision_model_type="dinov2",
                num_bridge_tokens=self.num_bridge_tokens,
                tokenizer_only=self.tokenizer_only,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class FourierGr1ArmsWaistGausNormCropCamEgoJointsOnlyDataConfig(BaseDataConfig):
    """GR1 joints-only (5 joint groups). SinCos on state, mean_std on action."""

    video_keys = ["video.ego_view"]
    state_keys = [
        "state.right_arm",
        "state.right_hand",
        "state.left_arm",
        "state.left_hand",
        "state.waist",
    ]
    action_keys = [
        "action.right_arm",
        "action.right_hand",
        "action.left_arm",
        "action.left_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    # After SinCos, state dims change; skip mean_std there. Actions use mean_std as usual.
    state_normalization_modes = {}
    action_normalization_modes = {k: "mean_std" for k in action_keys}

    def transform(self) -> ModalityTransform:
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoOffsetCrop(
                apply_to=self.video_keys,
                top=int(256 * 0.17),
                left=0,
                height=int(256 * 0.66),
                width=256,
            ),
            VideoCrop(apply_to=self.video_keys, scale=0.95, height=168, width=256),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=128,
                max_action_dim=128,
                eagle_path=self.eagle_path,
                use_bridge=self.use_bridge,
                ignore_lang_prefix=self.ignore_lang_prefix,
                enable_imagenet_preprocessing=True,
                vision_model_type="dinov2",
                num_bridge_tokens=self.num_bridge_tokens,
                tokenizer_only=self.tokenizer_only,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


DATA_CONFIG_MAP = {
    # Bimanual / single Panda
    "bimanual_panda_gripper": BimanualPandaGripperDataConfig,
    "bimanual_panda_hand": BimanualPandaHandDataConfig,
    "single_panda_gripper": SinglePandaGripperDataConfig,
    # SO100
    "so100": So100DataConfig,
    "so100_dualcam": So100DualCamDataConfig,
    # Unitree G1
    "unitree_g1": UnitreeG1DataConfig,
    "unitree_g1_full_body": UnitreeG1FullBodyDataConfig,
    # OXE DROID
    "oxe_droid": OxeDroidDataConfig,
    "oxe_droid_single_cam_abs": OxeDroidSingleCamAbsDataConfig,
    "oxe_droid_single_cam_state_as_action": OxeDroidSingleCamStateAsActionDataConfig,
    # Agibot Genie1
    "agibot_genie1": AgibotGenie1DataConfig,

    # ===== Mainline tokenizer configs =====
    # GR1 EEF-only / Joints-only (decoupled)
    "fourier_gr1_arms_waist_gausNorm_crop_cam_ego_eef_only": FourierGr1ArmsWaistGausNormCropCamEgoEefOnlyDataConfig,
    "fourier_gr1_arms_waist_gausNorm_crop_cam_ego_joints_only": FourierGr1ArmsWaistGausNormCropCamEgoJointsOnlyDataConfig,
    # EgoDex camera + hand at 448 source resolution
    "human_egodex_camera_hand_gausNorm_448_cam_ego": EgoDexCameraHandGausNorm448CamEgoDataConfig,
}