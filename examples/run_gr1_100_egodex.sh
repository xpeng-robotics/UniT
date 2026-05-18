#!/bin/bash
# UniT example pipeline: EgoDex basic pick-and-place + GR1-Joints (100 episodes).
# Stage 1: tokenizer training on EgoDex + GR1-Joints  (80k steps).
# Stage 2: dual-system pretrain on EgoDex + GR1-Joints (20k steps).
# Stage 3: dual-system finetune on GR1-Joints only     (20k steps, resumes Stage 2).
# Adjust GR1_DATASET_DIR / EGODEX_DATASET_DIR to your local LeRobot roots.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GR1_DATASET_DIR="${GR1_DATASET_DIR:-/path/to/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot-AugPosRot-Correct}"
EGODEX_DATASET_DIR="${EGODEX_DATASET_DIR:-/path/to/egodex_lerobot_gr00t_448}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/example_gr1_100_egodex}"
NUM_GPUS="${NUM_GPUS:-8}"

TOKENIZER_STEPS=80000
PRETRAIN_STEPS=20000
FINETUNE_STEPS=20000

GR1_JOINTS_CONFIG=fourier_gr1_arms_waist_gausNorm_crop_cam_ego_joints_only
EGODEX_CONFIG=human_egodex_camera_hand_gausNorm_448_cam_ego

EGODEX_DATASETS=(
    part2/basic_pick_place
)

GR1_DATASETS=(
    gr1_unified.PnPPotatoToMicrowaveClose
    gr1_unified.PnPMilkToMicrowaveClose
    gr1_unified.PnPCanToDrawerClose
    gr1_unified.PnPCupToDrawerClose
    gr1_unified.PnPBottleToCabinetClose
    gr1_unified.PnPWineToCabinetClose
    gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA
    gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA
    gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA
    gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA
    gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA
    gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA
    gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA
    gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA
    gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA
    gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA
    gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA
    gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA
    gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA
    gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA
    gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA
    gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA
    gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA
    gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA
)

# Mixed arrays (EgoDex first, then GR1) shared by tokenizer + pretrain.
MIX_DATASET_PATHS=()
MIX_DATA_CONFIGS=()
MIX_EMBODIMENT_TAGS=()
MIX_DATA_SPLITS=()
MIX_DATASET_WEIGHTS=()

for d in "${EGODEX_DATASETS[@]}"; do
    MIX_DATASET_PATHS+=("${EGODEX_DATASET_DIR}/${d}")
    MIX_DATA_CONFIGS+=("$EGODEX_CONFIG")
    MIX_EMBODIMENT_TAGS+=("human_egodex")
    MIX_DATA_SPLITS+=("[:-10]")
    MIX_DATASET_WEIGHTS+=(1.0)
done

for d in "${GR1_DATASETS[@]}"; do
    MIX_DATASET_PATHS+=("${GR1_DATASET_DIR}/${d}")
    MIX_DATA_CONFIGS+=("$GR1_JOINTS_CONFIG")
    MIX_EMBODIMENT_TAGS+=("gr1")
    MIX_DATA_SPLITS+=("[:100]")
    MIX_DATASET_WEIGHTS+=(5.0)
done

# GR1-only arrays (finetune stage).
GR1_DATASET_PATHS=()
GR1_DATA_CONFIGS=()
GR1_EMBODIMENT_TAGS=()
GR1_DATA_SPLITS=()
GR1_DATASET_WEIGHTS=()
for d in "${GR1_DATASETS[@]}"; do
    GR1_DATASET_PATHS+=("${GR1_DATASET_DIR}/${d}")
    GR1_DATA_CONFIGS+=("$GR1_JOINTS_CONFIG")
    GR1_EMBODIMENT_TAGS+=("gr1")
    GR1_DATA_SPLITS+=("[:100]")
    GR1_DATASET_WEIGHTS+=(1.0)
done

TOKENIZER_OUTPUT_DIR="${OUTPUT_ROOT}/tokenizer"
TOKENIZER_BASE_CONFIG="$PROJECT_ROOT/gr00t/model/configs/shared_tokenizer/gr00t_tokenizer_mix_unified_eef_dino.json"

echo "========== Stage 1: Tokenizer training (${TOKENIZER_STEPS} steps) =========="
python -u scripts/groot_tokenizer_train.py \
    --dataset-path "${MIX_DATASET_PATHS[@]}" \
    --data-config "${MIX_DATA_CONFIGS[@]}" \
    --embodiment_tag "${MIX_EMBODIMENT_TAGS[@]}" \
    --data_splits "${MIX_DATA_SPLITS[@]}" \
    --dataset_weights "${MIX_DATASET_WEIGHTS[@]}" \
    --num-gpus "$NUM_GPUS" \
    --batch-size 32 \
    --max-steps "$TOKENIZER_STEPS" \
    --save-steps "$TOKENIZER_STEPS" \
    --report_to tensorboard \
    --use_dino \
    --base_model_path "$TOKENIZER_BASE_CONFIG" \
    --output_dir "$TOKENIZER_OUTPUT_DIR"

TOKENIZER_CKPT="${TOKENIZER_OUTPUT_DIR}/checkpoint-${TOKENIZER_STEPS}"

PRETRAIN_OUTPUT_DIR="${OUTPUT_ROOT}/pretrain"
PRETRAIN_BASE_CONFIG="$PROJECT_ROOT/gr00t/model/configs/shared_dual_system/gr00t_n1.5_unit_tokenizer.json"

echo "========== Stage 2: Dual-system pretrain (${PRETRAIN_STEPS} steps) =========="
python -u scripts/dual_system_train.py \
    --dataset-path "${MIX_DATASET_PATHS[@]}" \
    --data-config "${MIX_DATA_CONFIGS[@]}" \
    --embodiment_tag "${MIX_EMBODIMENT_TAGS[@]}" \
    --data_splits "${MIX_DATA_SPLITS[@]}" \
    --dataset-weights "${MIX_DATASET_WEIGHTS[@]}" \
    --num-gpus "$NUM_GPUS" \
    --batch-size 32 \
    --max-steps "$PRETRAIN_STEPS" \
    --save-steps "$PRETRAIN_STEPS" \
    --report_to tensorboard \
    --base_model_path "$PRETRAIN_BASE_CONFIG" \
    --bridge_type "vision_lang_obs" \
    --detach_vl_for_action \
    --use_image_type_embedding \
    --ignore_lang_prefix \
    --compute-bridge-loss \
    --tune-llm \
    --no-tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --no-tune-bridge-visual \
    --groot_tokenizer_path "$TOKENIZER_CKPT" \
    --tune-bridge-embedding \
    --enable_imagenet_preprocessing \
    --select_layer 36 \
    --learning_rate 5e-5 \
    --output_dir "$PRETRAIN_OUTPUT_DIR"

PRETRAIN_CKPT="${PRETRAIN_OUTPUT_DIR}/checkpoint-${PRETRAIN_STEPS}"

FINETUNE_OUTPUT_DIR="${OUTPUT_ROOT}/finetune"

echo "========== Stage 3: Dual-system finetune on GR1-only (${FINETUNE_STEPS} steps) =========="
python -u scripts/dual_system_train.py \
    --dataset-path "${GR1_DATASET_PATHS[@]}" \
    --data-config "${GR1_DATA_CONFIGS[@]}" \
    --embodiment_tag "${GR1_EMBODIMENT_TAGS[@]}" \
    --data_splits "${GR1_DATA_SPLITS[@]}" \
    --dataset-weights "${GR1_DATASET_WEIGHTS[@]}" \
    --num-gpus "$NUM_GPUS" \
    --batch-size 32 \
    --max-steps "$FINETUNE_STEPS" \
    --save-steps "$FINETUNE_STEPS" \
    --report_to tensorboard \
    --base_model_path "$PRETRAIN_CKPT" \
    --bridge_type "vision_lang_obs" \
    --detach_vl_for_action \
    --use_image_type_embedding \
    --ignore_lang_prefix \
    --compute-bridge-loss \
    --tune-llm \
    --no-tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --no-tune-bridge-visual \
    --groot_tokenizer_path "$TOKENIZER_CKPT" \
    --tune-bridge-embedding \
    --enable_imagenet_preprocessing \
    --select_layer 36 \
    --learning_rate 5e-5 \
    --output_dir "$FINETUNE_OUTPUT_DIR"

echo "========== Pipeline complete =========="
echo "Tokenizer checkpoint: $TOKENIZER_CKPT"
echo "Pretrain  checkpoint: $PRETRAIN_CKPT"
echo "Finetune  checkpoint: ${FINETUNE_OUTPUT_DIR}/checkpoint-${FINETUNE_STEPS}"
