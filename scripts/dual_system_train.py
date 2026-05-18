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
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

# Add Isaac-GR00T root to sys.path FIRST (highest priority)
isaac_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if isaac_root not in sys.path:
    sys.path.insert(0, isaac_root)
    print(f"Added Isaac-GR00T root to sys.path (priority 0): {isaac_root}")


import torch
import tyro
from transformers import TrainingArguments, AutoConfig

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset, LeRobotSingleDatasetWithGoalImage, MultiEmbodimentLeRobotMixtureDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config_unit import load_data_config
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.gr00t_n1_unit import GR00T_N1_5_UniT
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model
from transformers import set_seed

@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""

    # ========== Required fields (no default value) - MUST be first ==========
    dataset_path: List[str]
    """List of dataset paths."""
    
    data_config: List[str]
    """List of data config names for each dataset."""
    
    embodiment_tag: List[Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())]]
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    base_model_path: str
    """Path to the base model checkpoint or its config JSON (required, must be provided via CLI).
    For Stage 1 pretrain, point this to the model config JSON
    (e.g. gr00t/model/configs/shared_dual_system/gr00t_n1.5_bridge_human_tokenizer.json).
    For Stage 2 finetune or any resume, point this to a previously saved checkpoint directory."""

    output_dir: str
    """Directory to save model checkpoints (required, must be provided via CLI)."""

    # ========== Optional fields (with default values) ==========

    # Training parameters
    batch_size: int = 32
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    tune_bridge_visual: bool = False
    """Whether to fine-tune the bridge vision model."""

    tune_bridge_embedding: bool = True
    """Whether to fine-tune the bridge token embeddings."""

    tune_all_llm_embedding: bool = False
    """Whether to fine-tune all the llm token embeddings."""

    bridge_type: Literal["vision_lang", "vision_lang_obs", "vision_lang_obs_e2e"] = "vision_lang_obs"
    """Type of bridge features"""

    detach_vl_for_action: bool = False
    """Whether to detach VL features before passing to action head. 
    When True, action loss won't update VLM through VL features, but CE loss still can.
    This helps prevent VLM from being polluted by embodiment-specific action patterns."""

    compute_bridge_loss: bool = True
    """Whether to compute loss for bridge features"""

    select_layer: int = 12
    """Selected llm backbone layer"""

    resume_pretrained_option: Literal["backbone+bridge_projector", "action_head", "all"] = "all"
    """Options for resuming pretrained ckpts"""

    bridge_loss_type: Literal["mse", "cosine", "mse_cosine", "ce"] = "ce"
    """Type of bridge loss"""

    action_loss_weight: float = 1.0
    """Weight for action loss (set to 0.0 to disable action loss)"""

    bridge_loss_weight: float = 0.1
    """Global scale applied to bridge_loss in the total loss:
    total = (action_loss_weight * action_loss + bridge_loss_weight * bridge_loss) / 2."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    dataloader_num_workers: int = 12
    """Number of workers for data loading per GPU."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps for training."""

    dataloader_prefetch_factor: int = 4
    """Prefetch factor for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard', 'azure_ml')."""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for training. [decord, torchvision_av]"""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, we will balance the dataset weights, by multiplying the total trajectory to each dataset"""

    # Mixture dataset parameters
    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories within a dataset weighted by their length; otherwise, equal weighting."""

    data_split: str = "[:-10]"
    """Data split to use for training."""

    data_splits: List[str] = None
    """List of data splits for each dataset. If None, all datasets use data_split."""
    
    dataset_weights: List[float] = None
    """List of weights for each dataset. If None, all datasets use weight 1.0."""

    unified_embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = None
    """If set, all samples use this embodiment ID for action encoder/decoder."""

    ignore_lang_prefix: bool = False

    use_vl_mask: bool = True

    use_correct_attn_mask: bool = True
    """If True, convert HF-style attention mask (1=valid, 0=pad) to SDPA-style (0=valid, -10000=masked). Default False for backward compatibility."""

    use_image_type_embedding: bool = False

    action_only_one_obs: bool = False

    noise_tau: float = 0

    tune_image_type_embedding: bool = True

    omit_image_type_embedding_for_goal: bool = False

    reweight_noise: bool = False

    enable_imagenet_preprocessing: bool = False
    """Whether to enable ImageNet preprocessing for obs and goal images."""

    groot_tokenizer_path: str = None
    """Path to GR00T tokenizer checkpoint (required)."""

    # NOTE: num_bridge_tokens is intentionally NOT a CLI knob. It is a model
    # architectural property and the single source of truth lives in
    # ``model_config.unit_cfg['num_bridge_tokens']``. The training pipeline
    # reads it from there and forwards it to load_data_config below.

#####################################################################################
# main training function
#####################################################################################


def main(config: ArgsConfig):
    """Main training function."""
    
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    model_config = AutoConfig.from_pretrained(config.base_model_path)
    
    # ------------ step 1: load dataset ------------
    embodiment_tag_list = []
    for embodiment_tag in config.embodiment_tag:
        embodiment_tag_list.append(EmbodimentTag(embodiment_tag))

    if config.unified_embodiment_tag is not None:
        config.unified_embodiment_id = EMBODIMENT_TAG_MAPPING[EmbodimentTag(config.unified_embodiment_tag).value]
    else:
        config.unified_embodiment_id = None

    if config.data_splits is None:
        data_splits = [config.data_split] * len(config.dataset_path)
    else:
        data_splits = config.data_splits
    print(f"data_splits: {data_splits}")

    # 1.1 modality configs and transforms
    use_bridge = model_config.unit_cfg['use_bridge']
    # Single source of truth for N: read from model config; no CLI override.
    num_bridge_tokens = model_config.unit_cfg['num_bridge_tokens'] if use_bridge else None
    print(f"Data transform use_bridge: {use_bridge}, num_bridge_tokens: {num_bridge_tokens}")

    modality_configs_list = []
    transforms_list = []
    for data_config in config.data_config:
        data_config_cls = load_data_config(
            data_config,
            eagle_path=model_config.backbone_cfg['eagle_path'],
            use_bridge=use_bridge,
            ignore_lang_prefix=config.ignore_lang_prefix,
            num_bridge_tokens=num_bridge_tokens,
        )
        modality_configs = data_config_cls.modality_config()
        transforms = data_config_cls.transform()
        tokenizer = transforms.transforms[-1].eagle_processor.tokenizer
        tokenizer_len = len(tokenizer)
        modality_configs_list.append(modality_configs)
        transforms_list.append(transforms)

    # 1.2 data loader: we will use either single dataset or mixture dataset
    if len(config.dataset_path) == 1:
        train_dataset = LeRobotSingleDatasetWithGoalImage(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs_list[0],
            transforms=transforms_list[0],
            embodiment_tag=embodiment_tag_list[0],  # This will override the dataset's embodiment tag to "new_embodiment"
            video_backend=config.video_backend,
            split=config.data_split,
        )
    else:
        single_datasets = []
        for i, p in enumerate(config.dataset_path):
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            ## We use the same transforms, modality configs, and embodiment tag for all datasets here,
            ## in reality, you can use dataset from different modalities and embodiment tags
            
            dataset = LeRobotSingleDatasetWithGoalImage(
                dataset_path=p,
                modality_configs=modality_configs_list[i],
                transforms=transforms_list[i],
                embodiment_tag=embodiment_tag_list[i],
                video_backend=config.video_backend,
                split=data_splits[i],
            )
            single_datasets.append(dataset)

        # Get dataset weights (default to 1.0 for all if not specified)
        if config.dataset_weights is None:
            dataset_weights = [1.0] * len(single_datasets)
        else:
            dataset_weights = config.dataset_weights
            assert len(dataset_weights) == len(single_datasets), \
                f"dataset_weights length ({len(dataset_weights)}) must match dataset_path length ({len(single_datasets)})"
        
        train_dataset = MultiEmbodimentLeRobotMixtureDataset(
            data_mixture=[
                (dataset, weight)
                for dataset, weight in zip(single_datasets, dataset_weights)
            ],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
        print(f"Loaded {len(single_datasets)} datasets with weights: {dataset_weights}")

    # ------------ step 2: load model ------------
    # First, get the data config to determine action horizon
    data_action_horizon = len(data_config_cls.action_indices)
    
    print(f"🤖 Loading Tokenizer model...")
    print(f"   Action horizon: {data_action_horizon}")
    # Load model
    model = GR00T_N1_5_UniT.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        resume_pretrained_option=config.resume_pretrained_option,

        tune_llm=config.tune_llm,  # backbone's LLM
        tune_visual=config.tune_visual,  # backbone's vision tower
        tune_projector=config.tune_projector,  # action head's projector
        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT

        tune_bridge_visual=config.tune_bridge_visual,
        tune_bridge_embedding=config.tune_bridge_embedding,
        tokenizer_len=tokenizer_len,
        bridge_type=config.bridge_type,
        detach_vl_for_action=config.detach_vl_for_action,
        compute_bridge_loss=config.compute_bridge_loss,
        select_layer=config.select_layer,

        bridge_loss_type=config.bridge_loss_type,
        action_loss_weight=config.action_loss_weight,
        bridge_loss_weight=config.bridge_loss_weight,
        tune_all_llm_embedding=config.tune_all_llm_embedding,

        use_vl_mask=config.use_vl_mask,
        use_correct_attn_mask=config.use_correct_attn_mask,
        use_image_type_embedding=config.use_image_type_embedding,
        action_only_one_obs=config.action_only_one_obs,

        noise_tau=config.noise_tau,
        reweight_noise=config.reweight_noise,

        groot_tokenizer_path=config.groot_tokenizer_path,
        unified_embodiment_id=config.unified_embodiment_id,
    )

    # Strict check: action_horizon must be consistent between data config and model config.
    # We deliberately do NOT auto-rebuild the action head here:
    #   - Rebuilding would silently swap FlowmatchingActionHeadUniT for FlowmatchingActionHead,
    #     dropping VL attention mask, unified_embodiment_id and per-sample loss.
    #   - A length mismatch is always a user-side configuration bug; failing fast is safer.
    if data_action_horizon != model.action_head.config.action_horizon:
        raise ValueError(
            f"action_horizon mismatch: data_config={data_action_horizon} "
            f"vs model.action_head.config={model.action_head.config.action_horizon}. "
            f"Align `len(data_config.action_indices)` with `action_head_cfg.action_horizon` "
            f"in the model JSON config (e.g. {config.base_model_path})."
        )

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"
    model.config.ignore_lang_prefix = config.ignore_lang_prefix
    model.config.video_delta_indices = data_config_cls.video_delta_indices
    model.config.output_dir = config.output_dir

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        # evaluation_strategy="no",
        save_total_limit=5,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
        eagle_path=model.config.backbone_cfg['eagle_path'],
        num_bridge_tokens=num_bridge_tokens,
    )

    # 2.3 run experiment
    experiment.train()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            script_path = Path(__file__).absolute()

            # Use subprocess.run instead of os.system
            raw_args_list = sys.argv[1:]
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
                *raw_args_list,
            ]

            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
