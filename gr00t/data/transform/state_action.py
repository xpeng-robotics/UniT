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

import functools
import random
from typing import Any, ClassVar, List, Dict

import numpy as np
import pytorch3d.transforms as pt
import torch
from pydantic import Field, PrivateAttr, field_validator, model_validator
import torch.nn.functional as F
from gr00t.data.schema import DatasetMetadata, RotationType, StateActionMetadata
from gr00t.data.transform.base import InvertibleModalityTransform, ModalityTransform


class RotationTransform:
    """Adapted from https://github.com/real-stanford/diffusion_policy/blob/548a52bbb105518058e27bf34dcf90bf6f73681a/diffusion_policy/model/common/rotation_transformer.py"""

    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

    def __init__(self, from_rep="axis_angle", to_rep="rotation_6d"):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        if from_rep.startswith("euler_angles"):
            from_convention = from_rep.split("_")[-1]
            from_rep = "euler_angles"
            from_convention = from_convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        else:
            from_convention = None
        if to_rep.startswith("euler_angles"):
            to_convention = to_rep.split("_")[-1]
            to_rep = "euler_angles"
            to_convention = to_convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        else:
            to_convention = None
        assert from_rep != to_rep, f"from_rep and to_rep cannot be the same: {from_rep}"
        assert from_rep in self.valid_reps, f"Invalid from_rep: {from_rep}"
        assert to_rep in self.valid_reps, f"Invalid to_rep: {to_rep}"

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != "matrix":
            funcs = [getattr(pt, f"{from_rep}_to_matrix"), getattr(pt, f"matrix_to_{from_rep}")]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [getattr(pt, f"matrix_to_{to_rep}"), getattr(pt, f"{to_rep}_to_matrix")]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: torch.Tensor, funcs: list) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        for func in funcs:
            x = func(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        return self._apply_funcs(x, self.inverse_funcs)


class Normalizer:
    valid_modes = ["q99", "mean_std", "min_max", "binary"]

    def __init__(self, mode: str, statistics: dict):
        self.mode = mode
        self.statistics = statistics
        for key, value in self.statistics.items():
            self.statistics[key] = torch.tensor(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"

        # Normalize the tensor
        if self.mode == "q99":
            # Range of q99 is [-1, 1]
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)

            # In the case of q01 == q99, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = q01 != q99
            normalized = torch.zeros_like(x)

            # Normalize the values where q01 != q99
            # Formula: 2 * (x - q01) / (q99 - q01) - 1
            normalized[..., mask] = (x[..., mask] - q01[..., mask]) / (
                q99[..., mask] - q01[..., mask]
            )
            normalized[..., mask] = 2 * normalized[..., mask] - 1

            # Set the normalized values to the original values where q01 == q99
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

            # Clip the normalized values to be between -1 and 1
            normalized = torch.clamp(normalized, -1, 1)

        elif self.mode == "mean_std":
            # Range of mean_std is not fixed, but can be positive or negative
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)

            # In the case of std == 0, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = std != 0
            normalized = torch.zeros_like(x)

            # Normalize the values where std != 0
            # Formula: (x - mean) / std
            normalized[..., mask] = (x[..., mask] - mean[..., mask]) / std[..., mask]

            # Set the normalized values to the original values where std == 0
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

        elif self.mode == "min_max":
            # Range of min_max is [-1, 1]
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)

            # In the case of min == max, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = min != max
            normalized = torch.zeros_like(x)

            # Normalize the values where min != max
            # Formula: 2 * (x - min) / (max - min) - 1
            normalized[..., mask] = (x[..., mask] - min[..., mask]) / (
                max[..., mask] - min[..., mask]
            )
            normalized[..., mask] = 2 * normalized[..., mask] - 1

            # Set the normalized values to the original values where min == max
            # normalized[..., ~mask] = x[..., ~mask].to(x.dtype)
            # Set the normalized values to 0 where min == max
            normalized[..., ~mask] = 0

        elif self.mode == "scale":
            # Range of scale is [0, 1]
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)
            abs_max = torch.max(torch.abs(min), torch.abs(max))
            mask = abs_max != 0
            normalized = torch.zeros_like(x)
            normalized[..., mask] = x[..., mask] / abs_max[..., mask]
            normalized[..., ~mask] = 0

        elif self.mode == "binary":
            # Range of binary is [0, 1]
            normalized = (x > 0.5).to(x.dtype)
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")

        return normalized

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        if self.mode == "q99":
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)
            return (x + 1) / 2 * (q99 - q01) + q01
        elif self.mode == "mean_std":
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)
            return x * std + mean
        elif self.mode == "min_max":
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)
            return (x + 1) / 2 * (max - min) + min
        elif self.mode == "binary":
            return (x > 0.5).to(x.dtype)
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")


class StateActionToTensor(InvertibleModalityTransform):
    """
    Transforms states and actions to tensors.
    """

    input_dtypes: dict[str, np.dtype] = Field(
        default_factory=dict, description="The input dtypes for each state key."
    )
    output_dtypes: dict[str, torch.dtype] = Field(
        default_factory=dict, description="The output dtypes for each state key."
    )

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to"}
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("input_dtypes", "output_dtypes", mode="before")
    def validate_dtypes(cls, v):
        for key, dtype in v.items():
            if isinstance(dtype, str):
                if dtype.startswith("torch."):
                    dtype_split = dtype.split(".")[-1]
                    v[key] = getattr(torch, dtype_split)
                elif dtype.startswith("np.") or dtype.startswith("numpy."):
                    dtype_split = dtype.split(".")[-1]
                    v[key] = np.dtype(dtype_split)
                else:
                    raise ValueError(f"Invalid dtype: {dtype}")
        return v

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(
                value, np.ndarray
            ), f"Unexpected input type: {type(value)}. Expected type: {np.ndarray}"
            data[key] = torch.from_numpy(value)
            if key in self.output_dtypes:
                data[key] = data[key].to(self.output_dtypes[key])
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(
                value, torch.Tensor
            ), f"Unexpected input type: {type(value)}. Expected type: {torch.Tensor}"
            data[key] = value.numpy()
            if key in self.input_dtypes:
                data[key] = data[key].astype(self.input_dtypes[key])
        return data



class HierarchicalRelativeTransform(InvertibleModalityTransform):
    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list, description="Not used in this transform, kept for compatibility."
    )
    
    # Bodies are 9D per node: {node}_pos + {node}_rot6d.
    camera_names: List[str] = ["camera", "camera_egoview"]
    wrist_names: List[str] = ["rightHand", "leftHand", "wrist_r", "wrist_l"]
    finger_to_wrist: Dict[str, str] = {
        # EgoDex
        "rightThumbIntermediateTip": "rightHand", "rightIndexFingerIntermediateBase": "rightHand",
        "rightMiddleFingerIntermediateBase": "rightHand", "rightRingFingerIntermediateBase": "rightHand",
        "rightLittleFingerIntermediateBase": "rightHand",
        "leftThumbIntermediateTip": "leftHand", "leftIndexFingerIntermediateBase": "leftHand",
        "leftMiddleFingerIntermediateBase": "leftHand", "leftRingFingerIntermediateBase": "leftHand",
        "leftLittleFingerIntermediateBase": "leftHand",
        # GR1
        "thumb_r": "wrist_r", "index_r": "wrist_r", "middle_r": "wrist_r",
        "ring_r": "wrist_r", "pinky_r": "wrist_r",
        "thumb_l": "wrist_l", "index_l": "wrist_l", "middle_l": "wrist_l",
        "ring_l": "wrist_l", "pinky_l": "wrist_l",
    }

    diff_keys: List[str] = Field(
        default_factory=list, 
        description="Keys for direct subtraction (action - state), e.g. joint angles like 'left_arm', 'right_arm'"
    )

    rot6d_only_keys: List[str] = Field(
        default_factory=list,
        description="Keys for rotation-only relative transform (6D), e.g. 'head_6d', 'waist_6d'"
    )

    def _rot6d_to_mat(self, d6):
        x = F.normalize(d6[..., 0:3], dim=-1)
        z = F.normalize(torch.cross(x, d6[..., 3:6], dim=-1), dim=-1)
        y = torch.cross(z, x, dim=-1)
        return torch.stack([x, y, z], dim=-1)

    def _mat_to_rot6d(self, mat):
        return mat[..., :3, :2].transpose(-1, -2).reshape(*mat.shape[:-2], 6)

    def _combine_to_4x4(self, pos, rot6d):
        T = torch.eye(4, device=pos.device, dtype=pos.dtype).repeat(*pos.shape[:-1], 1, 1)
        T[..., :3, :3] = self._rot6d_to_mat(rot6d)
        T[..., :3, 3] = pos
        return T

    def _get_keys(self, node, data, prefix="state"):
        pk = f"{prefix}.{node}_pos"
        pk = pk if pk in data else None
        rk = next((f"{prefix}.{node}_{s}" for s in ["rot", "rot6d"] if f"{prefix}.{node}_{s}" in data), None)
        return pk, rk

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # print(f"data.keys: {data.keys()}")
        """
        State: camera/wrist in world; fingers in wrist frame.
        Action: deltas in the same frames (world for camera/wrist, wrist for fingers).

        Also supports ``diff_keys`` (elementwise action - state) and ``rot6d_only_keys``
        (R_rel = R_state^{-1} R_action).
        """
        # 9D bodies: gather world-frame 4x4 for state and action per node.
        all_nodes = self.camera_names + self.wrist_names + list(self.finger_to_wrist.keys())
        wld_s_mats = {} # State at t
        wld_a_mats = {} # Action at t+k
        
        for node in all_nodes:
            # State 4x4
            spk, srk = self._get_keys(node, data, "state")
            if spk and srk: wld_s_mats[node] = self._combine_to_4x4(data[spk], data[srk])

            # Action 4x4 (often time-major, e.g. [T, 9])
            apk, ark = self._get_keys(node, data, "action")
            if apk and ark: wld_a_mats[node] = self._combine_to_4x4(data[apk], data[ark])

        if wld_s_mats:
            # State: express fingers in wrist frame; leave camera/wrist world poses as-is.
            for finger, wrist in self.finger_to_wrist.items():
                if finger in wld_s_mats and wrist in wld_s_mats:
                    T_f_in_w = torch.inverse(wld_s_mats[wrist]) @ wld_s_mats[finger]
                    pk, rk = self._get_keys(finger, data, "state")
                    data[pk], data[rk] = self._split_4x4(T_f_in_w)

            # Action: world deltas for camera/wrist; finger deltas in wrist frame.
            for node in all_nodes:
                apk, ark = self._get_keys(node, data, "action")
                if node not in wld_s_mats or node not in wld_a_mats: continue

                if node in self.camera_names or node in self.wrist_names:
                    # T_rel = T_wld(t)^{-1} T_wld_act(t+k)
                    delta_T = torch.inverse(wld_s_mats[node]) @ wld_a_mats[node]
                else:
                    # Finger delta in wrist frame: inv(T_f_now) T_f_future.
                    wrist = self.finger_to_wrist[node]
                    if wrist not in wld_s_mats or wrist not in wld_a_mats: continue
                    
                    T_f_in_w_now = torch.inverse(wld_s_mats[wrist]) @ wld_s_mats[node]
                    T_f_in_w_future = torch.inverse(wld_a_mats[wrist]) @ wld_a_mats[node]
                    delta_T = torch.inverse(T_f_in_w_now) @ T_f_in_w_future
                
                data[apk], data[ark] = self._split_4x4(delta_T)

        # diff_keys: elementwise relative action (e.g. joint angles).
        for key in self.diff_keys:
            state_key = f"state.{key}"
            action_key = f"action.{key}"
            if state_key in data and action_key in data:
                state_val = data[state_key]  # (dim,) or (1, dim)
                action_val = data[action_key]  # (T, dim)
                # Broadcast state along the action time axis.
                if state_val.dim() < action_val.dim():
                    state_val = state_val.unsqueeze(0)  # (1, dim)
                data[action_key] = action_val - state_val

        # rot6d_only_keys: rotation-only bodies (no translation).
        for key in self.rot6d_only_keys:
            state_key = f"state.{key}"
            action_key = f"action.{key}"
            if state_key in data and action_key in data:
                state_rot6d = data[state_key]  # (6,) or (1, 6)
                action_rot6d = data[action_key]  # (T, 6)

                if state_rot6d.dim() == 1:
                    state_rot6d = state_rot6d.unsqueeze(0)  # (1, 6)

                R_state = self._rot6d_to_mat(state_rot6d)  # (1, 3, 3)
                R_action = self._rot6d_to_mat(action_rot6d)  # (T, 3, 3)

                # R_rel = R_state^T R_action
                R_rel = R_state.transpose(-1, -2) @ R_action  # (T, 3, 3)

                data[action_key] = self._mat_to_rot6d(R_rel)

        return data

    def unapply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map relative quantities back to absolute space.

        - 9D bodies: ``T_abs = T_state @ T_rel``.
        - ``diff_keys``: ``action_abs = action_rel + state``.
        - ``rot6d_only_keys``: ``R_abs = R_state @ R_rel``.

        ``data`` may set ``_state_is_world_coord`` to True when ``state`` is still in
        world/original form rather than the finger-relative form produced by ``apply``.
        Default is False (state is assumed to match the post-``apply`` convention).
        """
        mats_s_abs = {}
        for node in self.camera_names + self.wrist_names:
            pk, rk = self._get_keys(node, data, "state")
            if pk and rk: mats_s_abs[node] = self._combine_to_4x4(data[pk], data[rk])
        
        for finger, wrist in self.finger_to_wrist.items():
            pk, rk = self._get_keys(finger, data, "state")
            if pk and rk and wrist in mats_s_abs:
                mats_s_abs[finger] = mats_s_abs[wrist] @ self._combine_to_4x4(data[pk], data[rk])
                data[pk], data[rk] = self._split_4x4(mats_s_abs[finger])

        mats_a_wld = {}
        for node in self.camera_names + self.wrist_names + list(self.finger_to_wrist.keys()):
            apk, ark = self._get_keys(node, data, "action")
            if not apk and not ark: 
                continue
            if node not in mats_s_abs:
                continue  # no state reference for this action chunk

            ref_action = data[apk] if apk else data[ark]
            batch_shape = ref_action.shape[:-1]
            device, dtype = ref_action.device, ref_action.dtype
            p_delta = data[apk] if apk else torch.zeros((*batch_shape, 3), device=device, dtype=dtype)
            r_delta = data[ark] if ark else torch.tensor([1., 0, 0, 0, 1, 0], device=device, dtype=dtype).repeat(*batch_shape, 1)
            T_delta = self._combine_to_4x4(p_delta, r_delta)
            
            if node in self.camera_names or node in self.wrist_names:
                T_act_wld = mats_s_abs[node] @ T_delta
            else:
                wrist = self.finger_to_wrist[node]
                if wrist not in mats_a_wld: continue
                T_f_in_w_now = torch.inverse(mats_s_abs[wrist]) @ mats_s_abs[node]
                T_act_wld = mats_a_wld[wrist] @ (T_f_in_w_now @ T_delta)
            
            mats_a_wld[node] = T_act_wld
            res_p, res_r = self._split_4x4(T_act_wld)
            if apk: data[apk] = res_p
            if ark: data[ark] = res_r

        # diff_keys: add state back.
        for key in self.diff_keys:
            state_key = f"state.{key}"
            action_key = f"action.{key}"
            if state_key in data and action_key in data:
                state_val = data[state_key]  # (dim,) or (1, dim)
                action_rel = data[action_key]  # (T, dim)
                if state_val.dim() < action_rel.dim():
                    state_val = state_val.unsqueeze(0)  # (1, dim)
                data[action_key] = action_rel + state_val

        # rot6d_only_keys: compose rotations.
        for key in self.rot6d_only_keys:
            state_key = f"state.{key}"
            action_key = f"action.{key}"
            if state_key in data and action_key in data:
                state_rot6d = data[state_key]  # (6,) or (1, 6)
                action_rel_rot6d = data[action_key]  # (T, 6)

                if state_rot6d.dim() == 1:
                    state_rot6d = state_rot6d.unsqueeze(0)  # (1, 6)

                R_state = self._rot6d_to_mat(state_rot6d)  # (1, 3, 3)
                R_rel = self._rot6d_to_mat(action_rel_rot6d)  # (T, 3, 3)

                R_abs = R_state @ R_rel  # (T, 3, 3)

                data[action_key] = self._mat_to_rot6d(R_abs)

        return data


    def _split_4x4(self, T):
        pos = T[..., :3, 3]
        rot6d = T[..., :3, :2].transpose(-1, -2).reshape(*T.shape[:-2], 6)
        return pos, rot6d



class CoordinateTransform(InvertibleModalityTransform):
    """
    Transforms coordinates (XYZ) or Rotation Vectors (6D) using a transformation matrix.
    Supports both Forward and Inverse transformations.
    """
    
    mode: str = Field(
        ..., 
        description="Transformation mode. Options: 'xyz' for (..., 3) points, 'rot6d' for (..., 6) rotation vectors."
    )
    matrix: List[List[float]] = Field(
        ..., 
        description="The 3x3 transformation matrix (Forward direction) as a list of lists."
    )

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "mode", "matrix"}
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("mode")
    def validate_mode(cls, v):
        if v not in ["xyz", "rot6d"]:
            raise ValueError(f"Invalid mode: {v}. Must be 'xyz' or 'rot6d'.")
        return v

    @field_validator("matrix")
    def validate_matrix(cls, v):
        arr = np.array(v)
        if arr.shape != (3, 3):
            raise ValueError(f"Matrix must be 3x3. Got shape {arr.shape}")
        return v

    def _get_matrix_tensor(self, device, dtype, inverse: bool = False) -> torch.Tensor:
        """Helper to get the transformation matrix as a Tensor on the correct device."""
        mat = torch.tensor(self.matrix, device=device, dtype=dtype)
        if inverse:
            # Calculate inverse. 
            # Note: If it's a pure rotation matrix, .T is sufficient, 
            # but linalg.inv is safer for general affine transforms.
            try:
                mat = torch.linalg.inv(mat)
            except RuntimeError:
                # Fallback for pseudo-inverse if not invertible (rare in coord transforms)
                mat = torch.linalg.pinv(mat)
        return mat

    def _process_batch(self, tensor: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """Core transformation logic."""
        if self.mode == "xyz":
            # Input: (..., 3)
            # Math: P_new = P_old @ Matrix.T
            return tensor @ matrix.T

        elif self.mode == "rot6d":
            # Input: (..., 6)
            original_shape = tensor.shape
            
            # 1. Reshape (..., 6) -> (..., 2, 3)
            # The 6D vector represents two 3D column vectors (r1, r2)
            tensor_reshaped = tensor.view(*original_shape[:-1], 2, 3)
            
            # 2. Apply Matrix to the last dimension (the 3D coords)
            # (..., 2, 3) @ (3, 3).T -> (..., 2, 3)
            transformed_reshaped = tensor_reshaped @ matrix.T
            
            # 3. Flatten back to (..., 6)
            return transformed_reshaped.view(*original_shape)
        
        return tensor

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Forward transformation (e.g., Egodex -> GR1).
        """
        for key in self.apply_to:
            if key not in data:
                continue
            
            value = data[key]
            assert isinstance(
                value, torch.Tensor
            ), f"Unexpected input type: {type(value)}. Expected type: {torch.Tensor}"

            # Get the matrix matching input's device and dtype
            mat = self._get_matrix_tensor(value.device, value.dtype, inverse=False)
            
            data[key] = self._process_batch(value, mat)
            
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Inverse transformation (e.g., GR1 -> Egodex).
        """
        for key in self.apply_to:
            if key not in data:
                continue
            
            value = data[key]
            assert isinstance(
                value, torch.Tensor
            ), f"Unexpected input type: {type(value)}. Expected type: {torch.Tensor}"

            # Get the INVERSE matrix
            mat_inv = self._get_matrix_tensor(value.device, value.dtype, inverse=True)
            
            data[key] = self._process_batch(value, mat_inv)
            
        return data


# class StateActionTransform(InvertibleModalityTransform):
#     """
#     Class for state or action transform.

#     Args:
#         apply_to (list[str]): The keys in the modality to load and transform.
#         normalization_modes (dict[str, str]): The normalization modes for each state key.
#             If a state key in apply_to is not present in the dictionary, it will not be normalized.
#         target_rotations (dict[str, str]): The target representations for each state key.
#             If a state key in apply_to is not present in the dictionary, it will not be rotated.
#     """

#     # Configurable attributes
#     apply_to: list[str] = Field(..., description="The keys in the modality to load and transform.")
#     normalization_modes: dict[str, str] = Field(
#         default_factory=dict, description="The normalization modes for each state key."
#     )
#     target_rotations: dict[str, str] = Field(
#         default_factory=dict, description="The target representations for each state key."
#     )
#     normalization_statistics: dict[str, dict] = Field(
#         default_factory=dict, description="The statistics for each state key."
#     )
#     modality_metadata: dict[str, StateActionMetadata] = Field(
#         default_factory=dict, description="The modality metadata for each state key."
#     )

#     # Model variables
#     _rotation_transformers: dict[str, RotationTransform] = PrivateAttr(default_factory=dict)
#     _normalizers: dict[str, Normalizer] = PrivateAttr(default_factory=dict)
#     _input_dtypes: dict[str, np.dtype | torch.dtype] = PrivateAttr(default_factory=dict)

#     # Model constants
#     _DEFAULT_MIN_MAX_STATISTICS: ClassVar[dict] = {
#         "rotation_6d": {
#             "min": [-1, -1, -1, -1, -1, -1],
#             "max": [1, 1, 1, 1, 1, 1],
#         },
#         "euler_angles": {
#             "min": [-np.pi, -np.pi, -np.pi],
#             "max": [np.pi, np.pi, np.pi],
#         },
#         "quaternion": {
#             "min": [-1, -1, -1, -1],
#             "max": [1, 1, 1, 1],
#         },
#         "axis_angle": {
#             "min": [-np.pi, -np.pi, -np.pi],
#             "max": [np.pi, np.pi, np.pi],
#         },
#     }

#     def model_dump(self, *args, **kwargs):
#         if kwargs.get("mode", "python") == "json":
#             include = {"apply_to", "normalization_modes", "target_rotations"}
#         else:
#             include = kwargs.pop("include", None)

#         return super().model_dump(*args, include=include, **kwargs)

#     @field_validator("modality_metadata", mode="before")
#     def validate_modality_metadata(cls, v):
#         for modality_key, config in v.items():
#             if isinstance(config, dict):
#                 config = StateActionMetadata.model_validate(config)
#             else:
#                 assert isinstance(
#                     config, StateActionMetadata
#                 ), f"Invalid source rotation config: {config}"
#             v[modality_key] = config
#         return v

#     @model_validator(mode="after")
#     def validate_normalization_statistics(self):
#         for modality_key, normalization_statistics in self.normalization_statistics.items():
#             if modality_key in self.normalization_modes:
#                 normalization_mode = self.normalization_modes[modality_key]
#                 if normalization_mode == "min_max":
#                     assert (
#                         "min" in normalization_statistics and "max" in normalization_statistics
#                     ), f"Min and max statistics are required for min_max normalization, but got {normalization_statistics}"
#                     assert len(normalization_statistics["min"]) == len(
#                         normalization_statistics["max"]
#                     ), f"Min and max statistics must have the same length, but got {normalization_statistics['min']} and {normalization_statistics['max']}"
#                 elif normalization_mode == "mean_std":
#                     assert (
#                         "mean" in normalization_statistics and "std" in normalization_statistics
#                     ), f"Mean and std statistics are required for mean_std normalization, but got {normalization_statistics}"
#                     assert len(normalization_statistics["mean"]) == len(
#                         normalization_statistics["std"]
#                     ), f"Mean and std statistics must have the same length, but got {normalization_statistics['mean']} and {normalization_statistics['std']}"
#                 elif normalization_mode == "q99":
#                     assert (
#                         "q01" in normalization_statistics and "q99" in normalization_statistics
#                     ), f"q01 and q99 statistics are required for q99 normalization, but got {normalization_statistics}"
#                     assert len(normalization_statistics["q01"]) == len(
#                         normalization_statistics["q99"]
#                     ), f"q01 and q99 statistics must have the same length, but got {normalization_statistics['q01']} and {normalization_statistics['q99']}"
#                 elif normalization_mode == "binary":
#                     assert (
#                         len(normalization_statistics) == 1
#                     ), f"Binary normalization should only have one value, but got {normalization_statistics}"
#                     assert normalization_statistics[0] in [
#                         0,
#                         1,
#                     ], f"Binary normalization should only have 0 or 1, but got {normalization_statistics[0]}"
#                 else:
#                     raise ValueError(f"Invalid normalization mode: {normalization_mode}")
#         return self

#     def set_metadata(self, dataset_metadata: DatasetMetadata):
#         dataset_statistics = dataset_metadata.statistics
#         modality_metadata = dataset_metadata.modalities

#         # Check that all state keys specified in apply_to have their modality_metadata
#         for key in self.apply_to:
#             split_key = key.split(".")
#             assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
#             if key not in self.modality_metadata:
#                 modality, state_key = split_key
#                 assert hasattr(modality_metadata, modality), f"{modality} config not found"
#                 assert state_key in getattr(
#                     modality_metadata, modality
#                 ), f"{state_key} config not found in {modality}"
#                 self.modality_metadata[key] = getattr(modality_metadata, modality)[state_key]

#         # Check that all state keys specified in normalization_modes have their statistics in state_statistics
#         for key in self.normalization_modes:
#             split_key = key.split(".")
#             assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
#             modality, state_key = split_key
#             assert hasattr(dataset_statistics, modality), f"{modality} statistics not found"
#             assert state_key in getattr(
#                 dataset_statistics, modality
#             ), f"{state_key} statistics not found"
#             assert (
#                 len(getattr(modality_metadata, modality)[state_key].shape) == 1
#             ), f"{getattr(modality_metadata, modality)[state_key].shape=}"
#             self.normalization_statistics[key] = getattr(dataset_statistics, modality)[
#                 state_key
#             ].model_dump()

#         # Initialize the rotation transformers
#         for key in self.target_rotations:
#             # Get the original representation of the state
#             from_rep = self.modality_metadata[key].rotation_type
#             assert from_rep is not None, f"Source rotation type not found for {key}"

#             # Get the target representation of the state, will raise an error if the target representation is not valid
#             to_rep = RotationType(self.target_rotations[key])

#             # If the original representation is not the same as the target representation, initialize the rotation transformer
#             if from_rep != to_rep:
#                 self._rotation_transformers[key] = RotationTransform(
#                     from_rep=from_rep.value, to_rep=to_rep.value
#                 )

#         # Initialize the normalizers
#         for key in self.normalization_modes:
#             modality, state_key = key.split(".")
#             # If the state has a nontrivial rotation, we need to handle it more carefully
#             # For absolute rotations, we need to convert them to the target representation and normalize them using min_max mode,
#             # since we can infer the bounds by the representation
#             # For relative rotations, we cannot normalize them as we don't know the bounds
#             if key in self._rotation_transformers:
#                 # Case 1: Absolute rotation
#                 if self.modality_metadata[key].absolute:
#                     # Check that the normalization mode is valid
#                     assert (
#                         self.normalization_modes[key] == "min_max"
#                     ), "Absolute rotations that are converted to other formats must be normalized using `min_max` mode"
#                     rotation_type = RotationType(self.target_rotations[key]).value
#                     # If the target representation is euler angles, we need to parse the convention
#                     if rotation_type.startswith("euler_angles"):
#                         rotation_type = "euler_angles"
#                     # Get the statistics for the target representation
#                     statistics = self._DEFAULT_MIN_MAX_STATISTICS[rotation_type]
#                 # Case 2: Relative rotation
#                 else:
#                     raise ValueError(
#                         f"Cannot normalize relative rotations: {key} that's converted to {self.target_rotations[key]}"
#                     )
#             # If the state is not continuous, we should not use normalization modes other than binary
#             elif (
#                 not self.modality_metadata[key].continuous
#                 and self.normalization_modes[key] != "binary"
#             ):
#                 raise ValueError(
#                     f"{key} is not continuous, so it should be normalized using `binary` mode"
#                 )
#             # Initialize the normalizer
#             else:
#                 statistics = self.normalization_statistics[key]
#             self._normalizers[key] = Normalizer(
#                 mode=self.normalization_modes[key], statistics=statistics
#             )

#     def apply(self, data: dict[str, Any]) -> dict[str, Any]:
#         for key in self.apply_to:
#             if key not in data:
#                 # We allow some keys to be missing in the data, and only process the keys that are present
#                 continue
#             if key not in self._input_dtypes:
#                 input_dtype = data[key].dtype
#                 assert isinstance(
#                     input_dtype, torch.dtype
#                 ), f"Unexpected input dtype: {input_dtype}. Expected type: {torch.dtype}"
#                 self._input_dtypes[key] = input_dtype
#             else:
#                 assert (
#                     data[key].dtype == self._input_dtypes[key]
#                 ), f"All states corresponding to the same key must be of the same dtype, input dtype: {data[key].dtype}, expected dtype: {self._input_dtypes[key]}"
#             # Rotate the state
#             state = data[key]
#             if key in self._rotation_transformers:
#                 state = self._rotation_transformers[key].forward(state)
#             # Normalize the state
#             if key in self._normalizers:
#                 state = self._normalizers[key].forward(state)
#             data[key] = state
#         return data

#     def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
#         for key in self.apply_to:
#             if key not in data:
#                 continue
#             state = data[key]
#             assert isinstance(
#                 state, torch.Tensor
#             ), f"Unexpected state type: {type(state)}. Expected type: {torch.Tensor}"
#             # Unnormalize the state
#             if key in self._normalizers:
#                 state = self._normalizers[key].inverse(state)
#             # Change the state back to its original representation
#             if key in self._rotation_transformers:
#                 state = self._rotation_transformers[key].inverse(state)
#             assert isinstance(
#                 state, torch.Tensor
#             ), f"State should be tensor after unapplying transformations, but got {type(state)}"
#             # Only convert back to the original dtype if it's known, i.e. `apply` was called before
#             # If not, we don't know the original dtype, so we don't convert
#             if key in self._input_dtypes:
#                 original_dtype = self._input_dtypes[key]
#                 if isinstance(original_dtype, np.dtype):
#                     state = state.numpy().astype(original_dtype)
#                 elif isinstance(original_dtype, torch.dtype):
#                     state = state.to(original_dtype)
#                 else:
#                     raise ValueError(f"Invalid input dtype: {original_dtype}")
#             data[key] = state
#         return data


class StateActionTransform(InvertibleModalityTransform):
    """
    Class for state or action transform.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        normalization_modes (dict[str, str]): The normalization modes for each state key.
            If a state key in apply_to is not present in the dictionary, it will not be normalized.
        target_rotations (dict[str, str]): The target representations for each state key.
            If a state key in apply_to is not present in the dictionary, it will not be rotated.
    """

    # Configurable attributes
    apply_to: list[str] = Field(..., description="The keys in the modality to load and transform.")
    normalization_modes: dict[str, str] = Field(
        default_factory=dict, description="The normalization modes for each state key."
    )
    target_rotations: dict[str, str] = Field(
        default_factory=dict, description="The target representations for each state key."
    )
    normalization_statistics: dict[str, dict] = Field(
        default_factory=dict, description="The statistics for each state key."
    )
    modality_metadata: dict[str, StateActionMetadata] = Field(
        default_factory=dict, description="The modality metadata for each state key."
    )
    use_relative: bool = False

    # Model variables
    _rotation_transformers: dict[str, RotationTransform] = PrivateAttr(default_factory=dict)
    _normalizers: dict[str, Normalizer] = PrivateAttr(default_factory=dict)
    _input_dtypes: dict[str, np.dtype | torch.dtype] = PrivateAttr(default_factory=dict)

    # Model constants
    _DEFAULT_MIN_MAX_STATISTICS: ClassVar[dict] = {
        "rotation_6d": {
            "min": [-1, -1, -1, -1, -1, -1],
            "max": [1, 1, 1, 1, 1, 1],
        },
        "euler_angles": {
            "min": [-np.pi, -np.pi, -np.pi],
            "max": [np.pi, np.pi, np.pi],
        },
        "quaternion": {
            "min": [-1, -1, -1, -1],
            "max": [1, 1, 1, 1],
        },
        "axis_angle": {
            "min": [-np.pi, -np.pi, -np.pi],
            "max": [np.pi, np.pi, np.pi],
        },
    }

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "normalization_modes", "target_rotations"}
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("modality_metadata", mode="before")
    def validate_modality_metadata(cls, v):
        for modality_key, config in v.items():
            if isinstance(config, dict):
                config = StateActionMetadata.model_validate(config)
            else:
                assert isinstance(
                    config, StateActionMetadata
                ), f"Invalid source rotation config: {config}"
            v[modality_key] = config
        return v

    @model_validator(mode="after")
    def validate_normalization_statistics(self):
        for modality_key, normalization_statistics in self.normalization_statistics.items():
            if modality_key in self.normalization_modes:
                normalization_mode = self.normalization_modes[modality_key]
                if normalization_mode == "min_max":
                    assert (
                        "min" in normalization_statistics and "max" in normalization_statistics
                    ), f"Min and max statistics are required for min_max normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["min"]) == len(
                        normalization_statistics["max"]
                    ), f"Min and max statistics must have the same length, but got {normalization_statistics['min']} and {normalization_statistics['max']}"
                elif normalization_mode == "mean_std":
                    assert (
                        "mean" in normalization_statistics and "std" in normalization_statistics
                    ), f"Mean and std statistics are required for mean_std normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["mean"]) == len(
                        normalization_statistics["std"]
                    ), f"Mean and std statistics must have the same length, but got {normalization_statistics['mean']} and {normalization_statistics['std']}"
                elif normalization_mode == "q99":
                    assert (
                        "q01" in normalization_statistics and "q99" in normalization_statistics
                    ), f"q01 and q99 statistics are required for q99 normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["q01"]) == len(
                        normalization_statistics["q99"]
                    ), f"q01 and q99 statistics must have the same length, but got {normalization_statistics['q01']} and {normalization_statistics['q99']}"
                elif normalization_mode == "binary":
                    assert (
                        len(normalization_statistics) == 1
                    ), f"Binary normalization should only have one value, but got {normalization_statistics}"
                    assert normalization_statistics[0] in [
                        0,
                        1,
                    ], f"Binary normalization should only have 0 or 1, but got {normalization_statistics[0]}"
                else:
                    raise ValueError(f"Invalid normalization mode: {normalization_mode}")
        return self

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        dataset_statistics = dataset_metadata.statistics
        modality_metadata = dataset_metadata.modalities

        # Check that all state keys specified in apply_to have their modality_metadata
        for key in self.apply_to:
            split_key = key.split(".")
            assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
            if key not in self.modality_metadata:
                modality, state_key = split_key
                assert hasattr(modality_metadata, modality), f"{modality} config not found"
                assert state_key in getattr(
                    modality_metadata, modality
                ), f"{state_key} config not found in {modality}"
                self.modality_metadata[key] = getattr(modality_metadata, modality)[state_key]

        # Check that all state keys specified in normalization_modes have their statistics in state_statistics
        for key in self.normalization_modes:
            split_key = key.split(".")
            assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
            modality, state_key = split_key
            assert hasattr(dataset_statistics, modality), f"{modality} statistics not found"
            assert (
                len(getattr(modality_metadata, modality)[state_key].shape) == 1
            ), f"{getattr(modality_metadata, modality)[state_key].shape=}"
            if self.use_relative:
                state_key = f"relative.{state_key}"
            assert state_key in getattr(
                dataset_statistics, modality
            ), f"{state_key} statistics not found"
            # print(f"{modality}.{state_key}")
            self.normalization_statistics[key] = getattr(dataset_statistics, modality)[
                state_key
            ].model_dump()

        # Initialize the rotation transformers
        for key in self.target_rotations:
            # Get the original representation of the state
            from_rep = self.modality_metadata[key].rotation_type
            assert from_rep is not None, f"Source rotation type not found for {key}"

            # Get the target representation of the state, will raise an error if the target representation is not valid
            to_rep = RotationType(self.target_rotations[key])

            # If the original representation is not the same as the target representation, initialize the rotation transformer
            if from_rep != to_rep:
                self._rotation_transformers[key] = RotationTransform(
                    from_rep=from_rep.value, to_rep=to_rep.value
                )

        # Initialize the normalizers
        # print(self.normalization_modes)
        for key in self.normalization_modes:
            modality, state_key = key.split(".")
            # If the state has a nontrivial rotation, we need to handle it more carefully
            # For absolute rotations, we need to convert them to the target representation and normalize them using min_max mode,
            # since we can infer the bounds by the representation
            # For relative rotations, we cannot normalize them as we don't know the bounds
            if key in self._rotation_transformers:
                # Case 1: Absolute rotation
                if self.modality_metadata[key].absolute:
                    # Check that the normalization mode is valid
                    assert (
                        self.normalization_modes[key] == "min_max"
                    ), "Absolute rotations that are converted to other formats must be normalized using `min_max` mode"
                    rotation_type = RotationType(self.target_rotations[key]).value
                    # If the target representation is euler angles, we need to parse the convention
                    if rotation_type.startswith("euler_angles"):
                        rotation_type = "euler_angles"
                    # Get the statistics for the target representation
                    statistics = self._DEFAULT_MIN_MAX_STATISTICS[rotation_type]
                # Case 2: Relative rotation
                else:
                    raise ValueError(
                        f"Cannot normalize relative rotations: {key} that's converted to {self.target_rotations[key]}"
                    )
            # If the state is not continuous, we should not use normalization modes other than binary
            elif (
                not self.modality_metadata[key].continuous
                and self.normalization_modes[key] != "binary"
            ):
                raise ValueError(
                    f"{key} is not continuous, so it should be normalized using `binary` mode"
                )
            # Initialize the normalizer
            else:
                statistics = self.normalization_statistics[key]
                # for k, v in self.normalization_statistics.items():
                #     if v['min'].shape[0] == 14:
                #         print(k, v)
            self._normalizers[key] = Normalizer(
                mode=self.normalization_modes[key], statistics=statistics
            )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                # We allow some keys to be missing in the data, and only process the keys that are present
                continue
            if key not in self._input_dtypes:
                input_dtype = data[key].dtype
                assert isinstance(
                    input_dtype, torch.dtype
                ), f"Unexpected input dtype: {input_dtype}. Expected type: {torch.dtype}"
                self._input_dtypes[key] = input_dtype
            else:
                assert (
                    data[key].dtype == self._input_dtypes[key]
                ), f"All states corresponding to the same key must be of the same dtype, input dtype: {data[key].dtype}, expected dtype: {self._input_dtypes[key]}"
            # Rotate the state
            state = data[key]
            # if state.shape[-1] == 14:
            #     print(f"{key} !!! {state}")
            if key in self._rotation_transformers:
                state = self._rotation_transformers[key].forward(state)
            # Normalize the state
            if key in self._normalizers:
                state = self._normalizers[key].forward(state)
            data[key] = state
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            state = data[key]
            assert isinstance(
                state, torch.Tensor
            ), f"Unexpected state type: {type(state)}. Expected type: {torch.Tensor}"
            # Unnormalize the state
            if key in self._normalizers:
                state = self._normalizers[key].inverse(state)
            # Change the state back to its original representation
            if key in self._rotation_transformers:
                state = self._rotation_transformers[key].inverse(state)
            assert isinstance(
                state, torch.Tensor
            ), f"State should be tensor after unapplying transformations, but got {type(state)}"
            # Only convert back to the original dtype if it's known, i.e. `apply` was called before
            # If not, we don't know the original dtype, so we don't convert
            if key in self._input_dtypes:
                original_dtype = self._input_dtypes[key]
                if isinstance(original_dtype, np.dtype):
                    state = state.numpy().astype(original_dtype)
                elif isinstance(original_dtype, torch.dtype):
                    state = state.to(original_dtype)
                else:
                    raise ValueError(f"Invalid input dtype: {original_dtype}")
            data[key] = state
        return data


class StateActionPerturbation(ModalityTransform):
    """
    Class for state or action perturbation.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        std (float): Standard deviation of the noise to be added to the state or action.
    """

    # Configurable attributes
    std: float = Field(
        ..., description="Standard deviation of the noise to be added to the state or action."
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training:
            # Don't perturb the data in eval mode
            return data
        if self.std < 0:
            # If the std is negative, we don't add any noise
            return data
        for key in self.apply_to:
            state = data[key]
            assert isinstance(state, torch.Tensor)
            transformed_data_min = torch.min(state)
            transformed_data_max = torch.max(state)
            noise = torch.randn_like(state) * self.std
            state += noise
            # Clip to the original range
            state = torch.clamp(state, transformed_data_min, transformed_data_max)
            data[key] = state
        return data


class StateActionDropout(ModalityTransform):
    """
    Class for state or action dropout.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        dropout_prob (float): Probability of dropping out a state or action.
    """

    # Configurable attributes
    dropout_prob: float = Field(..., description="Probability of dropping out a state or action.")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training:
            # Don't drop out the data in eval mode
            return data
        if self.dropout_prob < 0:
            # If the dropout probability is negative, we don't drop out any states
            return data
        if self.dropout_prob > 1e-9 and random.random() < self.dropout_prob:
            for key in self.apply_to:
                state = data[key]
                assert isinstance(state, torch.Tensor)
                state = torch.zeros_like(state)
                data[key] = state
        return data


class StateActionSinCosTransform(ModalityTransform):
    """
    Class for state or action sin-cos transform.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
    """

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            state = data[key]
            assert isinstance(state, torch.Tensor)
            sin_state = torch.sin(state)
            cos_state = torch.cos(state)
            data[key] = torch.cat([sin_state, cos_state], dim=-1)
        return data


class StateActionSubtractPosition(InvertibleModalityTransform):
    """
    Subtracts a reference position from specified keys.
    Useful for making position data relative to a reference point (e.g., hips).

    Args:
        apply_to (list[str]): The keys to transform.
        reference_key (str): The key of the reference position to subtract.
        position_indices (list[int]): Indices of the position coordinates (default: [0, 1, 2]).
    """
    reference_key: str = Field(..., description="The key of the reference position to subtract.")
    position_indices: List[int] = Field(default=[0, 1, 2], description="Indices of the position coordinates.")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.reference_key not in data:
            return data
        
        ref_pos = data[self.reference_key]
        assert isinstance(ref_pos, torch.Tensor)
        
        for key in self.apply_to:
            if key not in data:
                continue
            val = data[key]
            assert isinstance(val, torch.Tensor)
            val[..., self.position_indices] = val[..., self.position_indices] - ref_pos
            data[key] = val
        
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.reference_key not in data:
            return data
        
        ref_pos = data[self.reference_key]
        assert isinstance(ref_pos, torch.Tensor)
        
        for key in self.apply_to:
            if key not in data:
                continue
            val = data[key]
            assert isinstance(val, torch.Tensor)
            val[..., self.position_indices] = val[..., self.position_indices] + ref_pos
            data[key] = val
        
        return data


class StateActionRotateEuler(InvertibleModalityTransform):
    """
    Applies a rotation matrix to Euler angles.
    Converts Euler -> Matrix -> Multiply -> Euler.

    Args:
        apply_to (list[str]): The keys to transform.
        rotation_matrix (list[list[float]]): The 3x3 rotation matrix to right-multiply.
        rotation_indices (list[int]): Indices of the rotation coordinates (default: [0, 1, 2]).
        convention (str): Euler angle convention (default: "XYZ").
    """
    rotation_matrix: List[List[float]] = Field(..., description="The 3x3 rotation matrix to right-multiply.")
    rotation_indices: List[int] = Field(default=[0, 1, 2], description="Indices of the rotation coordinates.")
    convention: str = Field(default="XYZ", description="Euler angle convention.")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        transform_mat = torch.tensor(self.rotation_matrix, dtype=torch.float32)
        
        for key in self.apply_to:
            if key not in data:
                continue
            val = data[key]
            assert isinstance(val, torch.Tensor)
            
            eulers = val[..., self.rotation_indices]
            rot_mat = pt.euler_angles_to_matrix(eulers, convention=self.convention)
            transform_mat = transform_mat.to(rot_mat.device).to(rot_mat.dtype)
            new_rot_mat = torch.matmul(rot_mat, transform_mat)
            new_eulers = pt.matrix_to_euler_angles(new_rot_mat, convention=self.convention)
            
            val[..., self.rotation_indices] = new_eulers
            data[key] = val
        
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        transform_mat = torch.tensor(self.rotation_matrix, dtype=torch.float32)
        
        for key in self.apply_to:
            if key not in data:
                continue
            val = data[key]
            assert isinstance(val, torch.Tensor)
            
            eulers = val[..., self.rotation_indices]
            rot_mat = pt.euler_angles_to_matrix(eulers, convention=self.convention)
            transform_mat = transform_mat.to(rot_mat.device).to(rot_mat.dtype)
            new_rot_mat = torch.matmul(rot_mat, transform_mat.T)
            new_eulers = pt.matrix_to_euler_angles(new_rot_mat, convention=self.convention)
            
            val[..., self.rotation_indices] = new_eulers
            data[key] = val
        
        return data


class DropKeys(ModalityTransform):
    """
    Removes specified keys from the data dictionary.

    Args:
        apply_to (list[str]): The keys to remove from data.
    """

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key in data:
                del data[key]
        return data
