#!/bin/bash
# UniT example pipeline: GR1-Joints only (full episodes, 24 tasks).
# Stage 1: tokenizer training (80k steps).
# Stage 2: dual-system pretrain (single stage, 160k steps).
# Adjust GR1_DATASET_DIR to your local LeRobot root before running.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GR1_DATASET_DIR="${GR1_DATASET_DIR:-/path/to/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot-AugPosRot-Correct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/example_gr1_full}"
NUM_GPUS="${NUM_GPUS:-8}"

TOKENIZER_STEPS=80000
PRETRAIN_STEPS=160000

GR1_JOINTS_CONFIG=fourier_gr1_arms_waist_gausNorm_crop_cam_ego_joints_only

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

DATASET_PATHS=()
DATA_CONFIGS=()
EMBODIMENT_TAGS=()
DATA_SPLITS=()
DATASET_WEIGHTS=()
for d in "${GR1_DATASETS[@]}"; do
    DATASET_PATHS+=("${GR1_DATASET_DIR}/${d}")
    DATA_CONFIGS+=("$GR1_JOINTS_CONFIG")
    EMBODIMENT_TAGS+=("gr1")
    DATA_SPLITS+=("[:-10]")
    DATASET_WEIGHTS+=(1.0)
done

TOKENIZER_OUTPUT_DIR="${OUTPUT_ROOT}/tokenizer"
TOKENIZER_BASE_CONFIG="$PROJECT_ROOT/gr00t/model/configs/shared_tokenizer/gr00t_tokenizer_mix_unified_eef_dino.json"

echo "========== Stage 1: Tokenizer training (${TOKENIZER_STEPS} steps) =========="
python -u scripts/groot_tokenizer_train.py \
    --dataset-path "${DATASET_PATHS[@]}" \
    --data-config "${DATA_CONFIGS[@]}" \
    --embodiment_tag "${EMBODIMENT_TAGS[@]}" \
    --data_splits "${DATA_SPLITS[@]}" \
    --dataset_weights "${DATASET_WEIGHTS[@]}" \
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
    --dataset-path "${DATASET_PATHS[@]}" \
    --data-config "${DATA_CONFIGS[@]}" \
    --embodiment_tag "${EMBODIMENT_TAGS[@]}" \
    --data_splits "${DATA_SPLITS[@]}" \
    --dataset-weights "${DATASET_WEIGHTS[@]}" \
    --num-gpus "$NUM_GPUS" \
    --batch-size 32 \
    --max-steps "$PRETRAIN_STEPS" \
    --save-steps 80000 \
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

echo "========== Pipeline complete =========="
echo "Tokenizer checkpoint: $TOKENIZER_CKPT"
echo "Pretrain  checkpoint: ${PRETRAIN_OUTPUT_DIR}/checkpoint-${PRETRAIN_STEPS}"
