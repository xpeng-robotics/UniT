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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import BasePolicy

# numpy print precision settings 3, dont use exponential notation
np.set_printoptions(precision=3, suppress=True)


def download_from_hg(repo_id: str, repo_type: str) -> str:
    """
    Download the model/dataset from the hugging face hub.
    return the path to the downloaded
    """
    from huggingface_hub import snapshot_download

    repo_path = snapshot_download(repo_id, repo_type=repo_type)
    return repo_path


def calc_mse_for_single_trajectory(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    steps=300,
    action_horizon=16,
    plot=False,
    plot_state=False,
    save_plot_path=None,
):
    state_joints_across_time = []
    gt_action_across_time = []
    pred_action_across_time = []
    bridge_loss_across_times = []
    action_dim_names = []
    
    # Cosine similarity metrics (for DINO mode)
    vision_cos_sim_across_times = []
    vision_cos_sim_baseline_across_times = []
    
    # Image lists for video generation (Pixel mode)
    reconstructed_images_list = []
    target_images_list = []
    
    # Collect normalized actions for optional plotting (concat model space).
    normalized_gt_action_across_time = []
    normalized_pred_action_across_time = []

    for step_count in range(steps):
        data_point = None
        if plot_state:
            data_point = dataset.get_step_data(traj_id, step_count)
            concat_state = np.concatenate(
                [data_point[f"state.{key}"][0] for key in modality_keys], axis=0
            )
            state_joints_across_time.append(concat_state)

        if step_count % action_horizon == 0:
            if data_point is None:
                data_point = dataset.get_step_data(traj_id, step_count)

            print("inferencing at step: ", step_count)
            action_chunk = policy.get_action_and_bridgeloss(data_point)
            bridge_loss = action_chunk.get('bridge_loss', None)
            if bridge_loss is not None:
                bridge_loss_across_times.append(bridge_loss)
            
            # Collect cosine similarity metrics (DINO mode)
            vision_cos_sim = action_chunk.get('vision_cos_sim', None)
            if vision_cos_sim is not None:
                vision_cos_sim_across_times.append(vision_cos_sim)
            vision_cos_sim_baseline = action_chunk.get('vision_cos_sim_baseline', None)
            if vision_cos_sim_baseline is not None:
                vision_cos_sim_baseline_across_times.append(vision_cos_sim_baseline)
            
            # Collect images for video generation (Pixel mode)
            reconstructed_images = action_chunk.get('reconstructed_images', None)
            target_images = action_chunk.get('target_images', None)
            if reconstructed_images is not None:
                reconstructed_images_list.append(reconstructed_images)  # tensor (B, C, H, W)
            if target_images is not None:
                target_images_list.append(target_images)  # tensor (B, C, H, W)
            
            # Concat-space normalized pred/gt (for optional plots).
            normalized_pred = action_chunk.get('normalized_pred_action', None)
            normalized_gt = action_chunk.get('normalized_gt_action', None)
            action_mask = action_chunk.get('action_mask', None)  # (T, action_dim) dim mask
            
            # Filter modality_keys to only include keys that exist in action_chunk
            # This handles cases where data_config.action_keys != actual model output keys
            available_keys = [key for key in modality_keys if f"action.{key}" in action_chunk]
            if step_count == 0:
                print(f"Available action keys in model output: {available_keys}")
                if len(available_keys) != len(modality_keys):
                    missing = set(modality_keys) - set(available_keys)
                    print(f"⚠️  Missing keys (not in model output): {missing}")
                if normalized_pred is not None:
                    print(f"Normalized pred action shape: {normalized_pred.shape}")
                if normalized_gt is not None:
                    print(f"Normalized gt action shape: {normalized_gt.shape}")
            
            for j in range(action_horizon):
                # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in available_keys],
                    axis=0,
                )
                pred_action_across_time.append(concat_pred_action)

                concat_gt_action = np.concatenate(
                    [data_point[f"action.{key}"][j] for key in available_keys], axis=0
                )
                gt_action_across_time.append(concat_gt_action)
                
                # DEBUG: first-step gt vs pred (first 3 dims).
                if step_count == 0 and j == 0:
                    print(f"[DEBUG eval_unit] gt action from data_point (first 3 dims): {concat_gt_action[:3]}")
                    print(f"[DEBUG eval_unit] pred action from action_chunk (first 3 dims): {concat_pred_action[:3]}")
                    state_pos = data_point.get("state.camera_egoview_pos", None)
                    if state_pos is not None:
                        print(f"[DEBUG eval_unit] state.camera_egoview_pos: {state_pos[0][:3]}")
                
                if normalized_pred is not None:
                    normalized_pred_action_across_time.append(normalized_pred[j])
                if normalized_gt is not None:
                    normalized_gt_action_across_time.append(normalized_gt[j])

                if len(action_dim_names) == 0:
                    for key in available_keys:
                        for k in range(len(action_chunk[f"action.{key}"][0])):
                            action_dim_names.append(f"{key}.{k}")

    # plot the joints
    state_joints_across_time = np.array(state_joints_across_time)[:steps]
    gt_action_across_time = np.array(gt_action_across_time)[:steps]
    pred_action_across_time = np.array(pred_action_across_time)[:steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape

    # calc MSE across time (mask out outliers to reduce impact)
    MAX_DIFF = 0.2  # zero out per-dim errors above this (outlier clamp)
    diff = gt_action_across_time - pred_action_across_time
    diff_masked = np.where(np.abs(diff) > MAX_DIFF, 0, diff)
    action_mse = np.mean(diff_masked ** 2)
    outlier_ratio = np.mean(np.abs(diff) > MAX_DIFF) * 100
    print(f"Unnormalized Action MSE (|diff|>{MAX_DIFF} masked): {action_mse:.6f}, outliers: {outlier_ratio:.1f}%")

    # Per-category MSE: pos, rot, rot6d, other
    category_mse = {}
    if len(action_dim_names) > 0:
        # Group dim indices by key prefix (strip component index).
        pos_dims, rot_dims, rot6d_dims, other_dims = [], [], [], []
        for i, name in enumerate(action_dim_names):
            key_name = name.rsplit('.', 1)[0]
            if key_name.endswith('_pos') or key_name.endswith('_pos'):
                pos_dims.append(i)
            elif key_name.endswith('_rot6d'):
                rot6d_dims.append(i)
            elif key_name.endswith('_rot'):
                rot_dims.append(i)
            else:
                other_dims.append(i)
        
        print(f"\n=== Unnormalized MSE by category (|diff|>{MAX_DIFF} masked) ===")
        for cat_name, cat_dims in [("pos", pos_dims), ("rot", rot_dims), ("rot6d", rot6d_dims), ("other", other_dims)]:
            if len(cat_dims) > 0:
                cat_diff = diff_masked[:, cat_dims]
                cat_mse = np.mean(cat_diff ** 2)
                cat_outlier = np.mean(np.abs(diff[:, cat_dims]) > MAX_DIFF) * 100
                print(f"  {cat_name:25s}: MSE={cat_mse:.6f}, dims={len(cat_dims)}, outliers={cat_outlier:.1f}%")
                category_mse[cat_name] = cat_mse
        print()

    print("state_joints vs time", state_joints_across_time.shape)
    print("gt_action_joints vs time", gt_action_across_time.shape)
    print("pred_action_joints vs time", pred_action_across_time.shape)
    
    # Stack normalized trajectories when present.
    normalized_gt_action_across_time = np.array(normalized_gt_action_across_time)[:steps] if normalized_gt_action_across_time else None
    normalized_pred_action_across_time = np.array(normalized_pred_action_across_time)[:steps] if normalized_pred_action_across_time else None
    MAX_DIFF = 1.0
    normalized_action_mse = None
    if normalized_gt_action_across_time is not None and normalized_pred_action_across_time is not None:
        norm_diff = normalized_gt_action_across_time - normalized_pred_action_across_time
        norm_diff_masked = np.where(np.abs(norm_diff) > MAX_DIFF, 0, norm_diff)
        norm_outlier_ratio = np.mean(np.abs(norm_diff) > MAX_DIFF) * 100
        if action_mask is not None:
            dim_mask = action_mask[0]  # same mask over time
            diff_sq = (norm_diff_masked ** 2) * dim_mask
            normalized_action_mse = diff_sq.sum() / (dim_mask.sum() * len(normalized_gt_action_across_time))
        else:
            normalized_action_mse = np.mean(norm_diff_masked ** 2)
        print(f"Normalized Action MSE (|diff|>{MAX_DIFF} masked): {normalized_action_mse:.6f}, outliers: {norm_outlier_ratio:.1f}%")
        print("normalized_gt_action vs time", normalized_gt_action_across_time.shape)
        print("normalized_pred_action vs time", normalized_pred_action_across_time.shape)

    if len(bridge_loss_across_times) > 0:
        bridge_loss_across_times = np.array(bridge_loss_across_times)
        print("bridge_loss_across_times vs time", bridge_loss_across_times.shape)
        bridge_loss = np.mean(bridge_loss_across_times)
    else:
        bridge_loss = None
    
    # Compute average cosine similarity metrics (DINO mode)
    vision_cos_sim = None
    vision_cos_sim_baseline = None
    if len(vision_cos_sim_across_times) > 0:
        vision_cos_sim = np.mean(vision_cos_sim_across_times)
        print(f"Vision Cos Sim (recon vs goal): {vision_cos_sim:.4f}")
    if len(vision_cos_sim_baseline_across_times) > 0:
        vision_cos_sim_baseline = np.mean(vision_cos_sim_baseline_across_times)
        print(f"Vision Cos Sim Baseline (obs vs goal): {vision_cos_sim_baseline:.4f}")
        

    # raise error when pred action has NaN
    if np.isnan(pred_action_across_time).any():
        raise ValueError("Pred action has NaN")

    # num_of_joints = state_joints_across_time.shape[1]
    action_dim = gt_action_across_time.shape[1]

    if plot or save_plot_path is not None:
        info = {
            "state_joints_across_time": state_joints_across_time,
            "gt_action_across_time": gt_action_across_time,
            "pred_action_across_time": pred_action_across_time,
            "bridge_loss_across_times": bridge_loss_across_times,
            "modality_keys": modality_keys,
            "traj_id": traj_id,
            "action_mse": action_mse,
            "bridge_loss": bridge_loss,
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "steps": steps,
            "action_dim_names": action_dim_names,
            "normalized_gt_action_across_time": normalized_gt_action_across_time,
            "normalized_pred_action_across_time": normalized_pred_action_across_time,
            "normalized_action_mse": normalized_action_mse,
            "action_mask": action_mask,  # (T, action_dim) dim mask
        }
        plot_trajectory(info, save_plot_path)

    # Additional metrics dict (for extended info without breaking existing callers)
    extra_metrics = {
        'vision_cos_sim': vision_cos_sim,
        'vision_cos_sim_baseline': vision_cos_sim_baseline,
        'reconstructed_images_list': reconstructed_images_list if len(reconstructed_images_list) > 0 else None,
        'target_images_list': target_images_list if len(target_images_list) > 0 else None,
    }

    return action_mse, bridge_loss, category_mse, extra_metrics


def plot_trajectory(
    info,
    save_plot_path=None,
):
    """Simple plot of the trajectory with state, gt action, and pred action."""

    # Use non interactive backend for matplotlib if headless
    if save_plot_path is not None:
        matplotlib.use("Agg")

    action_dim = info["action_dim"]
    state_joints_across_time = info["state_joints_across_time"]
    gt_action_across_time = info["gt_action_across_time"]
    pred_action_across_time = info["pred_action_across_time"]
    bridge_loss_across_times = info["bridge_loss_across_times"]
    modality_keys = info["modality_keys"]
    traj_id = info["traj_id"]
    action_mse = info["action_mse"]
    bridge_loss = info["bridge_loss"]
    action_horizon = info["action_horizon"]
    steps = info["steps"]
    
    normalized_gt_action_across_time = info.get("normalized_gt_action_across_time", None)
    normalized_pred_action_across_time = info.get("normalized_pred_action_across_time", None)
    normalized_action_mse = info.get("normalized_action_mse", None)

    # Adjust figure size and spacing to accommodate titles
    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(10, 4 * action_dim + 2))

    # Leave plenty of space at the top for titles
    plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)

    print("Creating visualization...")

    # Combine all modality keys into a single string
    # add new line if total length is more than 60 chars
    modality_string = ""
    for key in modality_keys:
        modality_string += key + "\n " if len(modality_string) > 40 else key + ", "
    title_text = f"Trajectory Analysis - ID: {traj_id}\nModalities: {modality_string[:-2]}\nUnnormalized MSE: {action_mse:.6f}"
    if bridge_loss is not None:
        title_text += f"\nBridge Loss: {bridge_loss:.6f}"

    fig.suptitle(title_text, fontsize=14, fontweight="bold", color="#2E86AB", y=0.95)
    
    # Handle single subplot case
    if action_dim == 1:
        axes = [axes]

    # Loop through each action dim
    for i, ax in enumerate(axes):
        # The dimensions of state_joints and action are the same only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, i], label="state joints", alpha=0.7)
        ax.plot(gt_action_across_time[:, i], label="gt action", linewidth=2)
        ax.plot(pred_action_across_time[:, i], label="pred action", linewidth=2)

        # put a dot every ACTION_HORIZON
        for k, j in enumerate(range(0, steps, action_horizon)):
            if j == 0:
                ax.plot(j, gt_action_across_time[j, i], "ro", label="inference point", markersize=6)
            else:
                ax.plot(j, gt_action_across_time[j, i], "ro", markersize=4)

            if len(bridge_loss_across_times) > 0:
                if j == 0:
                    ax.plot(j, bridge_loss_across_times[k], "ko", label="bridge loss", markersize=6)
                else:
                    ax.plot(j, bridge_loss_across_times[k], "ko", markersize=4)

        ax.set_title(f"{info['action_dim_names'][i]}", fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Set better axis labels
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)

    if save_plot_path:
        print("saving unnormalized plot to", save_plot_path)
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    
    plt.close(fig)
    
    # Optional second figure: normalized concat action.
    if normalized_gt_action_across_time is not None and normalized_pred_action_across_time is not None:
        normalized_action_dim = normalized_gt_action_across_time.shape[1]
        action_mask = info.get("action_mask", None)  # (T, action_dim) dim mask
        
        fig2, axes2 = plt.subplots(nrows=normalized_action_dim, ncols=1, figsize=(10, 4 * normalized_action_dim + 2))
        plt.subplots_adjust(top=0.92, left=0.1, right=0.96, hspace=0.4)
        
        title_text2 = f"Normalized Action Analysis - ID: {traj_id}\n(Concat action into model)"
        if normalized_action_mse is not None:
            title_text2 += f"\nNormalized MSE: {normalized_action_mse:.6f}"
        
        fig2.suptitle(title_text2, fontsize=14, fontweight="bold", color="#8B0000", y=0.95)
        
        # Handle single subplot case
        if normalized_action_dim == 1:
            axes2 = [axes2]
        
        for i, ax in enumerate(axes2):
            ax.plot(normalized_gt_action_across_time[:, i], label="normalized gt action", linewidth=2, color="blue")
            ax.plot(normalized_pred_action_across_time[:, i], label="normalized pred action", linewidth=2, color="orange")
            
            # put a dot every ACTION_HORIZON
            for k, j in enumerate(range(0, steps, action_horizon)):
                if j == 0:
                    ax.plot(j, normalized_gt_action_across_time[j, i], "ro", label="inference point", markersize=6)
                else:
                    ax.plot(j, normalized_gt_action_across_time[j, i], "ro", markersize=4)
            
            dim_mask_str = ""
            if action_mask is not None and action_mask.shape[-1] > i:
                dim_masked = action_mask[0, i] > 0.5  # first timestep
                dim_mask_str = " [VALID]" if dim_masked else " [MASKED]"
            ax.set_title(f"Dim {i}{dim_mask_str}", fontsize=12, fontweight="bold", pad=10)
            ax.legend(loc="upper right", framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Time Step", fontsize=10)
            ax.set_ylabel("Normalized Value", fontsize=10)
        
        if save_plot_path:
            normalized_save_path = save_plot_path.replace(".png", "_normalized.png")
            print("saving normalized plot to", normalized_save_path)
            plt.savefig(normalized_save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
        
        plt.close(fig2)
