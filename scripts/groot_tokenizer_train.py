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

"""
GR00T Tokenizer Training Script

This script trains the GR00T Tokenizer for VQ-based reconstruction.
The tokenizer learns to reconstruct both visual observations and actions from quantized latent codes.
Configure the datasets directly in the DATASET_CONFIGS section below.
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Dict, Any

import torch
import tyro
from transformers import TrainingArguments, AutoConfig

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDatasetWithGoalImage, MultiEmbodimentLeRobotMixtureDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config_unit import load_data_config
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1_tokenizer_unit import GR00T_Tokenizer
from gr00t.utils.peft import get_lora_model
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from transformers import set_seed


@dataclass
class ArgsConfig:
    """Configuration for GR00T Tokenizer training."""

    # ========== Required fields (no default value) - MUST be first ==========
    dataset_path: List[str]
    """List of dataset paths."""
    
    data_config: List[str]
    """List of data config names for each dataset."""
    
    embodiment_tag: List[Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())]]
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    base_model_path: str
    """Path to the base tokenizer model config JSON (required, must be provided via CLI)."""

    output_dir: str
    """Directory to save model checkpoints (required, must be provided via CLI)."""

    # ========== Optional fields (with default values) ==========

    # Training parameters
    batch_size: int = 32
    """Batch size per GPU for training."""

    max_steps: int = 80000
    """Maximum number of training steps."""

    num_gpus: int = 8
    """Number of GPUs to use for training."""

    save_steps: int = 20000
    """Number of steps between saving checkpoints."""

    # Tokenizer-specific training parameters
    tune_vision_model: bool = False
    """Whether to fine-tune the shared vision model."""

    tune_vision_m_former: bool = True
    """Whether to fine-tune the vision branch M-Former."""

    tune_bridge_projector: bool = True
    """Whether to fine-tune the bridge projector (for action decoder input)."""

    tune_action_encoder: bool = True
    """Whether to fine-tune the action encoder."""

    tune_fusion: bool = True
    """Whether to fine-tune the visual-action fusion module."""

    tune_vq: bool = True
    """Whether to fine-tune the vector quantizer."""

    tune_vision_decoder: bool = True
    """Whether to fine-tune the vision reconstruction decoder."""

    tune_action_decoder_projector: bool = True
    """Whether to fine-tune the action decoder projector."""

    tune_action_decoder_diffusion: bool = True
    """Whether to fine-tune the action decoder diffusion model."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    unified_embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = None
    """If set, all samples use this embodiment ID for action encoder/decoder."""

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
    """Whether to use the full model for LORA."""

    dataloader_num_workers: int = 12
    """Number of workers for data loading per GPU."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps for training."""

    dataloader_prefetch_factor: int = 4
    """Prefetch factor for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "tensorboard"
    """Where to report training metrics."""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for training."""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """If True, balance dataset weights by multiplying the total trajectory length."""

    balance_trajectory_weights: bool = True
    """If True, sample trajectories weighted by their length."""

    data_split: str = "[:-10]"
    """Data split to use for training."""

    data_splits: List[str] = None
    
    dataset_weights: List[float] = None
    """List of weights for each dataset. If None, all datasets use weight 1.0."""

    episode_ids_json: str = None
    """Path to JSON file containing train/val/test episode splits. Format: {"train": [0,1,2,...], "val": [...], "test": [...]}"""
    
    episode_ids_key: str = "train"
    """Key to use from episode_ids_json file. One of: train, val, test."""

    ignore_lang_prefix: bool = False
    """Whether to ignore language prefix."""

    use_multi_scenario_training: bool = True
    """Whether to enable multi-scenario training (3x batch expansion for robust fusion learning)."""

    use_dino: bool = False
    """Whether to use DINO hidden state mode for vision reconstruction (cosine similarity loss instead of MSE+LPIPS)."""

#####################################################################################
# main training function
#####################################################################################


def main(config: ArgsConfig):
    """Main training function."""
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

    # Load episode_ids from JSON if provided
    episode_ids = None
    if config.episode_ids_json is not None:
        import json
        with open(config.episode_ids_json, 'r') as f:
            episode_splits = json.load(f)
        episode_ids = episode_splits.get(config.episode_ids_key, None)
        if episode_ids is None:
            raise ValueError(f"Key '{config.episode_ids_key}' not found in {config.episode_ids_json}. Available keys: {list(episode_splits.keys())}")
        print(f"Loaded {len(episode_ids)} episode_ids from {config.episode_ids_json}['{config.episode_ids_key}']")

    # 1.1 modality configs and transforms
    modality_configs_list = []
    transforms_list = []
    for data_config in config.data_config:
        # Tokenizer training does not invoke the eagle backbone:
        # - use_bridge=False  -> no <|bridge_i|> tokens appended to the prompt
        # - tokenizer_only=True -> skip eagle processor build, VLM tokenisation,
        #   and goal-image eagle preprocessing entirely (only ImageNet-preprocessed
        #   obs/goal images + state/action are produced).
        data_config_cls = load_data_config(
            data_config,
            eagle_path=model_config.backbone_cfg['eagle_path'],
            use_bridge=False,
            ignore_lang_prefix=config.ignore_lang_prefix,
            tokenizer_only=True,
        )
        modality_configs = data_config_cls.modality_config()
        transforms = data_config_cls.transform()
        
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
            split=config.data_split if episode_ids is None else None,
            episode_ids=episode_ids,
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
                split=data_splits[i] if episode_ids is None else None,
                episode_ids=episode_ids,
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
    
    # Enable multi-scenario training if specified
    if config.use_multi_scenario_training:
        model_config.use_multi_scenario_training = True
        print(f"✅ Multi-scenario training ENABLED (3x batch expansion)")
        print(f"   Scenarios: both (pv=1,pa=1), vision_only (pv=1,pa=0), action_only (pv=0,pa=1)")
        if config.batch_size > 16:
            print(f"   ⚠️  WARNING: Batch size {config.batch_size} may cause OOM with 3x expansion")
            print(f"   Consider reducing to ~{config.batch_size // 3} for similar memory usage")
    else:
        model_config.use_multi_scenario_training = False
        print(f"Multi-scenario training DISABLED (standard training)")
    
    # Enable DINO hidden state mode if specified
    if config.use_dino:
        # Set vision decoder to use DINO hidden states (input/output are features, not images)
        if hasattr(model_config, 'vision_decoder_cfg'):
            model_config.vision_decoder_cfg['is_io_hidden_states'] = True
        print(f"✅ DINO mode ENABLED:")
        print(f"   - vision_decoder_cfg.is_io_hidden_states = True")
        print(f"   - Vision reconstruction uses cosine similarity loss")

    # Load Tokenizer model
    model = GR00T_Tokenizer.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_vision_model=config.tune_vision_model,
        tune_vision_m_former=config.tune_vision_m_former,
        tune_bridge_projector=config.tune_bridge_projector,
        tune_action_encoder=config.tune_action_encoder,
        tune_fusion=config.tune_fusion,
        tune_vq=config.tune_vq,
        tune_vision_decoder=config.tune_vision_decoder,
        tune_action_decoder_projector=config.tune_action_decoder_projector,
        tune_action_decoder_diffusion=config.tune_action_decoder_diffusion,

        unified_embodiment_id=config.unified_embodiment_id,
    )
    
    # do not need to update action horizon if action_encoder and action_decoder are the same

    # # Update action horizon if needed (both action_encoder and action_decoder)
    # if data_action_horizon != model.action_decodxer.config.action_horizon:
    #     print(f"⚙️  Updating action horizon from {model.action_decoder.config.action_horizon} to {data_action_horizon}")
        
    #     # Update action encoder config
    #     model.action_decoder.config.action_horizon = data_action_horizon
        
    #     # Update action decoder (similar to Bridge model)
    #     from gr00t.model.action_head.flow_matching_action_head_unit import FlowmatchingActionHeadUniT
        
    #     new_action_decoder_config = model.action_decoder.config
    #     new_action_decoder_config.action_horizon = data_action_horizon
    #     new_action_decoder = FlowmatchingActionHeadUniT(new_action_decoder_config)
    #     new_action_decoder.load_state_dict(model.action_decoder.state_dict(), strict=False)
    #     model.action_decoder = new_action_decoder
        
    #     # Update config
    #     model.config.action_horizon = data_action_horizon
    #     model.config.action_en
    #     model.config.action_decoder_cfg["action_horizon"] = data_action_horizon
        
    #     # Reset trainable parameters for action decoder
    #     model.action_decoder.set_trainable_parameters(
    #         tune_projector=config.tune_action_decoder_projector,
    #         tune_diffusion_model=config.tune_action_decoder_diffusion
    #     )
    
    # Set model configuration
    model.config.use_multi_scenario_training = config.use_multi_scenario_training
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"
    model.config.ignore_lang_prefix = config.ignore_lang_prefix
    model.config.video_delta_indices = data_config_cls.video_delta_indices
    
    # DINO mode: set model instance attributes (config values are cached at __init__)
    if config.use_dino:
        # Set instance attributes directly (these are what forward() actually uses)
        model.is_dino_mode = True
        model.use_lpips_loss = False
        # Also update config for serialization
        model.config.vision_decoder_cfg['is_io_hidden_states'] = True
        model.config.use_lpips_loss = False
        print(f"✅ DINO mode applied to model:")
        print(f"   - model.is_dino_mode = True")
        print(f"   - model.use_lpips_loss = False")
        print(f"   - Vision reconstruction: cosine similarity loss (not MSE+LPIPS)")

    # Apply LoRA if specified
    if config.lora_rank > 0:
        print(f"🔧 Applying LoRA (rank={config.lora_rank}, alpha={config.lora_alpha})")
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )
    
    # Setup training arguments
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
        logging_steps=100.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=5,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )
    
    # Create trainer
    print(f"\n🚀 Starting training...")
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
        eagle_path=model.config.backbone_cfg['eagle_path'],
    )
    
    # Start training
    experiment.train()


if __name__ == "__main__":
    # Parse arguments
    config = tyro.cli(ArgsConfig)
    
    # Validate GPU configuration
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert config.num_gpus <= available_gpus, \
        f"Requested {config.num_gpus} GPUs but only {available_gpus} available"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    
    if config.num_gpus == 1:
        # Single GPU mode
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        main(config)
    else:
        # Multi-GPU mode - use torchrun
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            script_path = Path(__file__).absolute()
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            
            raw_args_list = sys.argv[1:]
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",
                str(script_path),
                *raw_args_list,
            ]
            
            print("Running torchrun command:", " ".join(cmd))
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)

