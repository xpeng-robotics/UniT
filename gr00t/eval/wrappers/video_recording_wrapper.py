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
import uuid
from pathlib import Path

import av
import gymnasium as gym
import numpy as np

def get_accumulate_timestamp_idxs(
    timestamps: list[float],
    start_time: float,
    dt: float,
    eps: float = 1e-5,
    next_global_idx: int | None = 0,
    allow_negative: bool = False,
) -> tuple[list[int], list[int], int]:
    """
    For each dt window, choose the first timestamp in the window.
    Assumes timestamps sorted. One timestamp might be chosen multiple times due to dropped frames.
    next_global_idx should start at 0 normally, and then use the returned next_global_idx.
    However, when overwiting previous values are desired, set last_global_idx to None.

    Returns:
    local_idxs: which index in the given timestamps array to chose from
    global_idxs: the global index of each chosen timestamp
    next_global_idx: used for next call.
    """
    local_idxs = list()
    global_idxs = list()
    for local_idx, ts in enumerate(timestamps):
        # add eps * dt to timestamps so that when ts == start_time + k * dt
        # is always recorded as kth element (avoiding floating point errors)
        global_idx = np.floor((ts - start_time) / dt + eps)
        if (not allow_negative) and (global_idx < 0):
            continue
        if next_global_idx is None:
            next_global_idx = global_idx

        n_repeats = max(0, global_idx - next_global_idx + 1)
        for i in range(n_repeats):
            local_idxs.append(local_idx)
            global_idxs.append(next_global_idx + i)
        next_global_idx += n_repeats
    return local_idxs, global_idxs, next_global_idx


class VideoRecorder:
    def __init__(
        self,
        fps,
        codec,
        input_pix_fmt,
        # options for codec
        **kwargs,
    ):
        """
        input_pix_fmt: rgb24, bgr24 see https://github.com/PyAV-Org/PyAV/blob/bc4eedd5fc474e0f25b22102b2771fe5a42bb1c7/av/video/frame.pyx#L352
        """

        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.kwargs = kwargs
        # runtime set
        self._reset_state()

    def _reset_state(self):
        self.container = None
        self.stream = None
        self.shape = None
        self.dtype = None
        self.start_time = None
        self.next_global_idx = 0

    @classmethod
    def create_h264(
        cls,
        fps,
        codec="h264",
        input_pix_fmt="rgb24",
        output_pix_fmt="yuv420p",
        crf=18,
        profile="high",
        **kwargs,
    ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={"crf": str(crf), "profile:v": "high"},
            **kwargs,
        )
        return obj

    def __del__(self):
        self.stop()

    def is_ready(self):
        return self.stream is not None

    def start(self, file_path, start_time=None):
        if self.is_ready():
            # if still recording, stop first and start anew.
            self.stop()

        self.container = av.open(file_path, mode="w")
        self.stream = self.container.add_stream(self.codec, rate=self.fps)
        codec_context = self.stream.codec_context
        for k, v in self.kwargs.items():
            setattr(codec_context, k, v)
        self.start_time = start_time

    def write_frame(self, img: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError("Must run start() before writing!")

        n_repeats = 1
        if self.start_time is not None:
            local_idxs, global_idxs, self.next_global_idx = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=self.start_time,
                dt=1 / self.fps,
                next_global_idx=self.next_global_idx,
            )
            # number of appearance means repeats
            n_repeats = len(local_idxs)

        if self.shape is None:
            self.shape = img.shape
            self.dtype = img.dtype
            h, w, c = img.shape
            self.stream.width = w
            self.stream.height = h
        assert img.shape == self.shape
        assert img.dtype == self.dtype

        frame = av.VideoFrame.from_ndarray(img, format=self.input_pix_fmt)
        for i in range(n_repeats):
            for packet in self.stream.encode(frame):
                self.container.mux(packet)

    def stop(self):
        if not self.is_ready():
            return

        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)

        # Close the file
        self.container.close()

        # reset runtime parameters
        self._reset_state()


# Extra state.* keys for get_additional_obs (camera vs left/right hand chains).
JOINTS_TO_TRACK = {
    # Camera (camera API; type prefix "camera_")
    "state.camera_egoview_pos": ("camera_pos3d", "egoview"),  # 3D camera position
    "state.camera_egoview_rot6d": ("camera_rot6d", "egoview"),  # 6D camera orientation
    # Left hand (body API)
    "state.wrist_l_pos": ("pos3d", "gripper0_left_l_palm"),
    "state.wrist_l_rot6d": ("rot6d", "gripper0_left_l_palm"),
    "state.thumb_l_pos": ("pos3d", "gripper0_left_L_thumb_distal_link"), 
    "state.thumb_l_rot6d": ("rot6d", "gripper0_left_L_thumb_distal_link"),
    "state.index_l_pos": ("pos3d", "gripper0_left_L_index_intermediate_link"),
    "state.index_l_rot6d": ("rot6d", "gripper0_left_L_index_intermediate_link"),
    "state.middle_l_pos": ("pos3d", "gripper0_left_L_middle_intermediate_link"),
    "state.middle_l_rot6d": ("rot6d", "gripper0_left_L_middle_intermediate_link"),
    "state.ring_l_pos": ("pos3d", "gripper0_left_L_ring_intermediate_link"),
    "state.ring_l_rot6d": ("rot6d", "gripper0_left_L_ring_intermediate_link"),
    "state.pinky_l_pos": ("pos3d", "gripper0_left_L_pinky_intermediate_link"),
    "state.pinky_l_rot6d": ("rot6d", "gripper0_left_L_pinky_intermediate_link"),
    # Right hand (body API)
    "state.wrist_r_pos": ("pos3d", "gripper0_right_r_palm"),
    "state.wrist_r_rot6d": ("rot6d", "gripper0_right_r_palm"),
    "state.thumb_r_pos": ("pos3d", "gripper0_right_R_thumb_distal_link"),
    "state.thumb_r_rot6d": ("rot6d", "gripper0_right_R_thumb_distal_link"),
    "state.index_r_pos": ("pos3d", "gripper0_right_R_index_intermediate_link"),
    "state.index_r_rot6d": ("rot6d", "gripper0_right_R_index_intermediate_link"),
    "state.middle_r_pos": ("pos3d", "gripper0_right_R_middle_intermediate_link"),
    "state.middle_r_rot6d": ("rot6d", "gripper0_right_R_middle_intermediate_link"),
    "state.ring_r_pos": ("pos3d", "gripper0_right_R_ring_intermediate_link"),
    "state.ring_r_rot6d": ("rot6d", "gripper0_right_R_ring_intermediate_link"),
    "state.pinky_r_pos": ("pos3d", "gripper0_right_R_pinky_intermediate_link"),
    "state.pinky_r_rot6d": ("rot6d", "gripper0_right_R_pinky_intermediate_link"),
}


class VideoRecordingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        video_recorder: VideoRecorder,
        mode="rgb_array",
        video_dir: Path | None = None,
        steps_per_render=1,
        state_modality_keys=None,
        **kwargs,
    ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)

        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)

        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.video_dir = video_dir
        self.video_recorder = video_recorder
        self.file_path = None
        self.state_modality_keys = state_modality_keys

        # Extend observation_space with extra state.* entries so VectorEnv can stack observations.
        # Only when state_modality_keys is not None (backward compatible with None).
        if state_modality_keys is not None:
            from gymnasium import spaces
            new_spaces = dict(self.observation_space.spaces) if hasattr(self.observation_space, 'spaces') else {}
            for key in state_modality_keys:
                if key not in new_spaces and key in JOINTS_TO_TRACK:
                    data_type, _ = JOINTS_TO_TRACK[key]
                    if data_type in ("pos3d", "camera_pos3d"):
                        new_spaces[key] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64)
                    elif data_type in ("rot6d", "camera_rot6d"):
                        new_spaces[key] = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
            self.observation_space = spaces.Dict(new_spaces)

        self.step_count = 0

        self.is_success = False

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        obs, info = result
        if self.state_modality_keys is not None and \
           {k for k in obs.keys() if k.startswith("state.")} != self.state_modality_keys:
            obs = self.get_additional_obs(obs)
            result = (obs, info)
        
        language = None
        obs = result[0]
        for k in obs:
            if "lang" in k or "annotation" in k:
                language = obs[k]
                if type(language) == tuple:
                    language = language[0]
                break

        self.frames = list()
        self.step_count = 1
        self.video_recorder.stop()

        if self.video_dir is not None and self.file_path is not None:
            # rename the file to indicate success or failure
            original_filestem = self.file_path.stem
            new_filestem = f"success{int(self.is_success)}_{original_filestem}"
            new_file_path = self.video_dir / f"{new_filestem}.mp4"
            os.rename(self.file_path, new_file_path)

        self.is_success = False
        if self.video_dir is not None:
            if language is None:
                self.file_path = self.video_dir / f"{uuid.uuid4()}.mp4"
            else:
                self.file_path = self.video_dir / f"{language}_{uuid.uuid4()}.mp4"
        return result

    def step(self, action):
        result = super().step(action)
        obs = result[0]
        # When configured, fill missing state.* keys via get_additional_obs.
        if self.state_modality_keys is not None and \
           {k for k in obs.keys() if k.startswith("state.")} != self.state_modality_keys:
            obs = self.get_additional_obs(obs)
            result = (obs,) + result[1:]

        self.step_count += 1
        if self.file_path is not None and ((self.step_count % self.steps_per_render) == 0):
            if not self.video_recorder.is_ready():
                self.video_recorder.start(self.file_path)

            frame = self.env.render()
            assert frame.dtype == np.uint8
            self.video_recorder.write_frame(frame)
            # self.is_success = result[-1]["success"]
            self.is_success = self.is_success | result[-1]["success"]
            # print(result)
        return result

    def _get_robosuite_env(self):
        """Walk the wrapper chain until we find an env that exposes `sim` (robosuite)."""
        env = self.env
        while hasattr(env, 'env'):
            if hasattr(env, 'sim'):
                return env
            env = env.env
        # Innermost env should be robosuite-based.
        if hasattr(env, 'sim'):
            return env
        raise RuntimeError(f"Cannot find robosuite environment with 'sim' attribute. Current env type: {type(env)}")

    def get_additional_obs(self, obs):
        robosuite_env = self._get_robosuite_env()
        
        for k in self.state_modality_keys:
            if k not in obs:
                if k not in JOINTS_TO_TRACK:
                    continue
                    
                data_type, obj_name = JOINTS_TO_TRACK[k]
                
                # Cameras: use Mujoco camera buffers.
                if data_type == "camera_pos3d":
                    cam_id = robosuite_env.sim.model.camera_name2id(obj_name)
                    cam_pos = robosuite_env.sim.data.cam_xpos[cam_id].copy()
                    obs[k] = cam_pos
                elif data_type == "camera_rot6d":
                    cam_id = robosuite_env.sim.model.camera_name2id(obj_name)
                    cam_mat = robosuite_env.sim.data.cam_xmat[cam_id].copy().reshape(3, 3)
                    rot_6d = cam_mat[:, :2].T.flatten()
                    obs[k] = rot_6d
                # Rigid body: use body_* APIs.
                elif data_type == "pos3d":
                    body_id = robosuite_env.sim.model.body_name2id(obj_name)
                    pos = robosuite_env.sim.data.body_xpos[body_id].copy()
                    obs[k] = pos
                elif data_type == "rot6d":
                    body_id = robosuite_env.sim.model.body_name2id(obj_name)
                    rot_mat = robosuite_env.sim.data.body_xmat[body_id].copy().reshape(3, 3)
                    rot_6d = rot_mat[:, :2].T.flatten()
                    obs[k] = rot_6d
                    
        return obs

    def render(self, mode="rgb_array", **kwargs):
        if self.video_recorder.is_ready():
            self.video_recorder.stop()
        return self.file_path
