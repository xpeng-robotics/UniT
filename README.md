# UniT: Toward a Unified Physical Language for Human-to-Humanoid Policy Learning and World Modeling

<p align="center">
  <a href="https://xpeng-robotics.github.io/unit/">
    <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-5b8dc9?style=flat-square&logo=github">
  </a>
  <a href="https://arxiv.org/abs/2604.19734">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2604.19734-b31b1b?style=flat-square&logo=arxiv">
  </a>
</p>

<p align="center">
  <b>Boyu Chen</b><sup>1,2,*</sup> &nbsp;·&nbsp;
  <b>Yi Chen</b><sup>1,3,*</sup> &nbsp;·&nbsp;
  <b>Lu Qiu</b><sup>1,3</sup> &nbsp;·&nbsp;
  <b>Jerry Bai</b><sup>1</sup> &nbsp;·&nbsp;
  <b>Yuying Ge</b><sup>1,&dagger;</sup> &nbsp;·&nbsp;
  <b>Yixiao Ge</b><sup>1</sup>
  <br>
  <sup>1</sup>XPENG Robotics &nbsp;·&nbsp;
  <sup>2</sup>Tsinghua University &nbsp;·&nbsp;
  <sup>3</sup>The University of Hong Kong
  <br>
  <sub><sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>&dagger;</sup>Corresponding author</sub>
  <br>
  <sub>Correspondence: <a href="mailto:yyge13@gmail.com">yyge13@gmail.com</a></sub>
</p>

<p align="center">
  <img src="assets/teaser.jpeg" alt="UniT teaser — from human demonstration to humanoid policy and world model" width="100%">
</p>

---

> **Project page:** <https://xpeng-robotics.github.io/unit/>

## Overview

Scaling humanoid foundation models is bottlenecked by the scarcity of robotic data.
While massive egocentric human data offers a scalable alternative, bridging the
cross-embodiment chasm remains a fundamental challenge due to kinematic mismatches.
We introduce **UniT** (**Uni**fied Latent Action **T**okenizer via Visual
Anchoring), a framework that learns a unified physical language for
human-to-humanoid manipulation transfer. Grounded in the philosophy that
heterogeneous kinematics share consistent visual consequences, UniT employs a
tri-branch cross-reconstruction mechanism: actions predict vision to anchor
kinematics to physical outcomes, while vision reconstructs actions to filter out
irrelevant visual confounders. Concurrently, a fusion branch integrates these
purified modalities into a shared discrete latent space of cross-embodiment
physical intents.

We validate UniT across two paradigms: (1) **Policy Learning (VLA-UniT):** By
predicting these unified tokens, VLA-UniT achieves state-of-the-art performance
with high data efficiency on the RoboCasa GR1 benchmark. Leveraging diverse human
data further improves out-of-distribution (OOD) generalization in simulation
and real-world deployment, and enables *zero-shot task transfer* in the real
world. (2) **World Modeling (WM-UniT):** By aligning cross-embodiment dynamics
via unified tokens as conditions, it supports direct human-to-humanoid
action-conditioned generation, translating human knowledge into enhanced action
controllability for humanoid video generation.

Ultimately, by inducing a more aligned cross-embodiment representation
(empirically supported by t-SNE visualizations revealing improved alignment of
human and humanoid features), UniT offers a scalable path to distill human
priors into humanoid manipulation capabilities.

## Table of contents

- [Installation](#installation)
  - [Training environment](#training-environment)
  - [Simulation evaluation environment](#simulation-evaluation-environment)
- [Data preparation](#data-preparation)
  - [RoboCasa GR1 simulation data](#robocasa-gr1-simulation-data)
  - [EgoDex human data](#egodex-human-data)
- [Training](#training)
- [Evaluation](#evaluation)
  - [Online evaluation](#online-evaluation)
  - [Offline evaluation](#offline-evaluation)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Status](#status)
- [Contact](#contact)

## Installation

**Requirements:** conda; NVIDIA GPU and a PyTorch/CUDA build that matches your driver.  
Set **`eagle_path`** in the model config JSON to point at the vision-language backbone (e.g. a **Qwen2.5-VL** checkpoint on **Hugging Face** if you use the default recipe).

Bootstrap the Python env (optional overrides `CONDA_ENV_NAME`, `PIP_INDEX_URL`): [`examples/environment_setup.sh`](examples/environment_setup.sh).

### Training environment

Developed with **CUDA 12.x**; default hyperparameters assume **120+ GB** GPU memory per device.

```bash
bash examples/environment_setup.sh
conda activate unit
```

### Simulation evaluation environment

Requires the same **`unit`** base installation as [Training environment](#training-environment), plus simulation dependencies for RoboCasa / MuJoCo eval. Rollout was tested on **24 GB** GPUs (NVIDIA RTX 4090).

```bash
# System dependencies
sudo apt-get install -y libegl1-mesa libegl1-mesa-dev libosmesa6-dev patchelf
```

**`third_party/`** (local editable installs, not committed — see [`.gitignore`](.gitignore)):

```bash
mkdir -p third_party

# Install robosuite v1.5.1
git clone https://github.com/ARISE-Initiative/robosuite.git third_party/robosuite
cd third_party/robosuite
git checkout v1.5.1
pip install -e .
cd ../..

# Install robocasa-gr1-tabletop-tasks
git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git third_party/robocasa-gr1-tabletop-tasks
pip install -e third_party/robocasa-gr1-tabletop-tasks
```

**Required patch**: Apply the following fix to `third_party/robocasa-gr1-tabletop-tasks/robocasa/models/objects/kitchen_object_utils.py`.
In the `sample_kitchen_object_helper()` function, after the line `reg_choices = reg_choices[split_th:]` under `elif split == "B":`, add:

```python
if "assets/objects/sketchfab/basket/" in reg_choices[0]:
    reg_choices = [c for c in reg_choices if not c.endswith('basket_4/model.xml')]
```

**Download tabletop assets**:

```bash
python third_party/robocasa-gr1-tabletop-tasks/robocasa/scripts/download_tabletop_assets.py -y
```

Pipelines and eval env vars: [`examples/README.md`](examples/README.md).

## Data preparation

### RoboCasa GR1 simulation data

1. **Source.** [nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim) on Hugging Face (HDF5 + LeRobot).

2. **Joints-only (default).** Main UniT recipes use **joint** actions only—**no EEF**. Download the **LeRobot** splits from that dataset; you do **not** need HDF5 or the augmentation steps below.

3. **EEF + joints (optional).** Download **both** HDF5 and LeRobot. Install the **IK** stack used by the EEF extraction scripts:

```bash
pip install pin==3.9.0
pip install mink==0.0.5
```

Raw LeRobot data only includes joint angles. To add end-effector poses (3D position + 6D rotation), run two steps per task:

```bash
# Step 1: Extract EEF poses by replaying trajectories in the simulator
#   Input:  HDF5 file (e.g., HDF5/TaskName.hdf5)
#   Output: Replay parquet directory (e.g., Replay-Correct/TaskName/)
python preprocessing/extract_and_visualize_3d-pos_6d-rot_from_gr1.py \
    --dataset /path/to/HDF5/TaskName.hdf5 \
    --output_dir /path/to/Replay-Correct/TaskName \
    --render_image_names egoview \
    --verbose \
    --render_height 800 \
    --render_width 1280 \
    --num_parallel_jobs 10

# Step 2: Augment the LeRobot dataset with extracted poses
#   Input:  LeRobot dataset + Replay parquet from Step 1
#   Output: Augmented LeRobot dataset (LeRobot-AugPosRot-Correct/)
python preprocessing/aug_lerobot_data.py \
    --lerobot_base_path /path/to/LeRobot/gr1_unified.TaskName \
    --replay_base_path /path/to/Replay-Correct/TaskName/parquet/ \
    --output_base_path /path/to/LeRobot-AugPosRot-Correct/gr1_unified.TaskName
```

Repeat for all 24 tasks. The final augmented datasets under `LeRobot-AugPosRot-Correct/` are used for training.

`GR1_DATASET_DIR`: see [`examples/README.md`](examples/README.md).

### EgoDex human data

1. **Download** raw EgoDex data from [apple/ml-egodex](https://github.com/apple/ml-egodex).

2. **Install LeRobot** (required for data format conversion only):

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout d602e8169cbad9e93a4a3b3ee1dd8b332af7ebf8
pip install -e .
pip install tyro h5py
cd ..
```

3. **Convert to LeRobot format** (two steps per subset):

```bash
# Step 1: Convert raw EgoDex HDF5 to LeRobot v2.1 format
python preprocessing/convert_egodex_data_to_lerobot.py \
    --raw_dir /path/to/egodex/basic_pick_place \
    --repo_id /path/to/egodex_lerobot/part2/basic_pick_place

# Step 2: Convert from LeRobot v2.1 to v2.0 (GR00T-compatible format)
python preprocessing/convert_dataset_v21_to_v20_gr00t.py \
    --repo-id=part2/basic_pick_place \
    --root=/path/to/egodex_lerobot/part2/basic_pick_place
```

`EGODEX_DATASET_DIR`: see [`examples/README.md`](examples/README.md).

## Training

UniT training is staged: **tokenizer** → **dual-system pretrain** on the chosen data mix, then **dual-system finetune** on GR1 only when using the few-shot recipe. Scripts under [`examples/`](examples/) run these steps in order; a failure aborts the pipeline and the next step always resumes from the previous checkpoint.

We ship two **GR1-Joints** entry points:

- **`examples/run_gr1_100_egodex.sh`** — **EgoDex + GR1-100**: tokenizer, mixed pretrain (EgoDex + GR1), then finetune on GR1 alone.
- **`examples/run_gr1_full.sh`** — **GR1-full**: tokenizer, then dual-system pretrain on all GR1-Joints data (no separate finetune stage in this script).

[`examples/README.md`](examples/README.md) lists stage details, defaults, environment overrides, and copy-paste commands.

## Evaluation

> **Before evaluating a released checkpoint:** set `unit_cfg.groot_tokenizer_path` in the checkpoint's `config.json` to the corresponding tokenizer.

### Online evaluation

Closed-loop rollout against a trained checkpoint in the RoboCasa GR1 simulation environment:

```bash
bash examples/run_eval.sh <model_path> <eval_type>
```

Replace the placeholders:

- `<model_path>` — absolute or repo-relative path to your trained checkpoint directory (the same one consumed by `scripts/inference_service_unit.py`).
- `<eval_type>` — one of the five entries below.

**Evaluation types:**

| Type | Description | Tasks |
|------|-------------|-------|
| `id` | In-distribution training tasks | 24 (6 PnPClose + 18 PosttrainPnPNovel SplitA) |
| `ood_object_appearance` | OOD unseen object appearance | 18 (EvalPnPNovel SplitB) |
| `ood_container_combination` | OOD unseen source-target containers | 14 (PretrainPnPNovel SplitA) |
| `ood_object_type` | OOD unseen object types | 32 (PretrainPnPBase SplitA) |
| `unseen_close` | OOD variants of PnP*Close | 9 |

**Example:**

```bash
# In-distribution evaluation
bash examples/run_eval.sh /path/to/checkpoint id

# OOD unseen object appearance
bash examples/run_eval.sh /path/to/checkpoint ood_object_appearance

# Customize port, env count, episodes per task
PORT=8891 N_ENVS=1 N_EPISODES=50 \
    bash examples/run_eval.sh /path/to/checkpoint ood_object_type
```

`run_eval.sh` launches the inference server, retries each task up to 5 times against `scripts/simulation/simulation_service_unit.py`, then writes per-task success rates to `<model_path>/evaluation_sim_<eval_type>_${N_ENVS}envs${EVAL_TAG}/results.json`.

Tunables (env vars): `DATA_CONFIG`, `N_ENVS`, `N_EPISODES`, `CUDA_VISIBLE_DEVICES`, `PORT`, `EVAL_TAG`.

### Offline evaluation

Open-loop action-reconstruction MSE on held-out GR1-Joints trajectories, no simulator required:

```bash
bash examples/run_eval_loss.sh <model_path>
```

`<model_path>` is the trained checkpoint directory (same meaning as in online evaluation). The script runs `scripts/eval_policy_unit.py` over the 24 GR1 training datasets, using the tail slice of each dataset (`DATA_SPLIT`) as held-out trajectories, and writes per-task action plots and metrics to `<model_path>/eval_action_dual_system_${DATA_SPLIT}/`.

**Examples:**

```bash
# Default settings (last 2 trajectories per task, GR1-Joints data config)
bash examples/run_eval_loss.sh /path/to/checkpoint

# Point at a local GR1 LeRobot root
GR1_DATASET_DIR=/abs/path/to/gr1 \
    bash examples/run_eval_loss.sh /path/to/checkpoint
```

Tunables (env vars): `GR1_DATASET_DIR`, `DATA_CONFIG`, `DATA_SPLIT` (default `[-2:]`), `TRAJS` (default `2`), `CUDA_VISIBLE_DEVICES`.

> **Note:** Default `GR1_DATASET_DIR` in `examples/run_eval_loss.sh` points at the EEF-augmented layout (`LeRobot-AugPosRot-Correct/`). For the joints-only main recipe, point it at your plain LeRobot root.

### Reference results

Overall in-distribution success rates from `examples/run_eval.sh ... id` (24 tasks, 50 episodes each, `N_ENVS=1`):

| Recipe | Script | Overall SR |
|---|---|---|
| GR1-full | `examples/run_gr1_full.sh` | **66.4%** |
| EgoDex + GR1-100 (few-shot) | `examples/run_gr1_100_egodex.sh` | **50.9%** |

Per-task numbers are in [`docs/evaluation_id_results.md`](docs/evaluation_id_results.md).

## Citation

If you find this work useful, please cite:

```bibtex
@article{chen2026unit,
  title={UniT: Toward a Unified Physical Language for Human-to-Humanoid Policy Learning and World Modeling},
  author={Chen, Boyu and Chen, Yi and Qiu, Lu and Bai, Jerry and Ge, Yuying and Ge, Yixiao},
  journal={arXiv preprint arXiv:2604.19734},
  year={2026}
}
```

## Acknowledgements

This codebase is built on top of [NVIDIA Isaac GR00T N1.5](https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release), an open foundation model for generalized humanoid robot policies. We thank NVIDIA for open-sourcing the GR00T N1.5 model, data pipeline, and training infrastructure, which served as the foundation for our work. We also thank the authors of [DIAL](https://github.com/xpeng-robotics/DIAL), [EgoDex](https://github.com/apple/ml-egodex), [RoboCasa](https://github.com/robocasa/robocasa-gr1-tabletop-tasks), [LeRobot](https://github.com/huggingface/lerobot), and [lucidrains/vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) (for the VQ-VAE building blocks) for their open-source contributions.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Status

> **The code release is in progress.** Data preparation, the UniT tokenizer, and
> VLA-UniT (training + RoboCasa GR1 evaluation) are available now. WM-UniT,
> pretrained checkpoints, and the real-world deployment stack will follow.

Planned release order:

- ✅ Data preparation scripts
- ✅ UniT tokenizer — training & inference
- ✅ VLA-UniT — training & evaluation on RoboCasa GR1
- ✅ Pretrained checkpoints
- ⏳ WM-UniT — training & sampling on RoboCasa GR1 and GR00T-Teleop mixtures
- ⏳ Real-world deployment stack

## Contact

For questions about the paper or the upcoming release, please open an issue in
this repository, or reach out to:

- Yuying Ge &mdash; [yyge13@gmail.com](mailto:yyge13@gmail.com) *(corresponding author)*
- Boyu Chen &mdash; [boyuc448@gmail.com](mailto:boyuc448@gmail.com)
