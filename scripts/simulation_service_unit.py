# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RoboCasa / bridge simulation client (ZMQ).

Launches rollout against a running RobotInferenceServer. Run the server separately, e.g.:

``python scripts/inference_service_unit.py --server --model_path ... --data_config ... --port ...``
"""

import argparse
import json
import os

import numpy as np
from transformers import AutoConfig

from gr00t.eval.simulation import (
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
    VideoConfig,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Checkpoint path or HF id (read for video_delta_indices only when not default).",
        default="<PATH_TO_YOUR_MODEL>",
    )
    parser.add_argument("--env_name", type=str, help="Environment name.", default="<ENV_NAME>")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--n_episodes", type=int, default=2)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=16,
        help="Action steps per environment step.",
    )
    parser.add_argument("--max_episode_steps", type=int, default=1440)
    parser.add_argument("--client", action="store_true", required=True)

    args = parser.parse_args()

    video_delta_indices = np.array([0])
    if args.model_path != "<PATH_TO_YOUR_MODEL>":
        model_config = AutoConfig.from_pretrained(args.model_path)
        if hasattr(model_config, "video_delta_indices"):
            video_delta_indices = np.array(model_config.video_delta_indices)

    print(f"video_delta_indices: {video_delta_indices}")

    env_name = args.env_name
    parent = os.path.split(args.video_dir.replace(f"videos/{env_name}", ""))[0]
    result_path = os.path.join(parent, "results.json")
    print(os.path.exists(result_path), result_path)
    if os.path.exists(result_path):
        with open(result_path) as f:
            results = json.load(f)
        print(env_name)
        print(results["tasks"])
        print("\n\n")
        if env_name in results["tasks"]:
            print(f"Results for {env_name}:\nSuccess rate: {results['tasks'][env_name]:.2f}")
            raise SystemExit(0)

    simulation_client = SimulationInferenceClient(host=args.host, port=args.port)

    print("Available modality configs:")
    modality_config = simulation_client.get_modality_config()
    print(modality_config.keys())

    sim_cfg = SimulationConfig(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        video=VideoConfig(video_dir=args.video_dir),
        multistep=MultiStepConfig(
            n_action_steps=args.n_action_steps,
            max_episode_steps=args.max_episode_steps,
            video_delta_indices=video_delta_indices,
        ),
    )

    print(f"Running simulation for {args.env_name}...")
    env_name, episode_successes = simulation_client.run_simulation(sim_cfg)

    print(f"Results for {env_name}:")
    print(f"Success rate: {np.mean(episode_successes):.2f}")
