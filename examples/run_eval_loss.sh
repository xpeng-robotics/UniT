#!/bin/bash
# UniT open-loop action-loss evaluation on GR1-Joints validation trajectories.
# Runs `scripts/eval_policy_unit.py` over the 24 training tasks, using the held
# out tail of each LeRobot dataset to compute action-reconstruction MSE.
#
# Usage:
#   bash examples/run_eval_loss.sh <model_path>
#
# Environment variables:
#   GR1_DATASET_DIR   (default: /path/to/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot-AugPosRot-Correct)
#   DATA_CONFIG       (default: fourier_gr1_arms_waist_gausNorm_crop_cam_ego_joints_only)
#   DATA_SPLIT        (default: [-2:])
#   TRAJS             (default: 2)
#   CUDA_VISIBLE_DEVICES (default: 0)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH=${1:?"Usage: bash examples/run_eval_loss.sh <model_path>"}

GR1_DATASET_DIR=${GR1_DATASET_DIR:-/path/to/PhysicalAI-Robotics-GR00T-Teleop-Sim/LeRobot-AugPosRot-Correct}
DATA_CONFIG=${DATA_CONFIG:-fourier_gr1_arms_waist_gausNorm_crop_cam_ego_joints_only}
DATA_SPLIT=${DATA_SPLIT:-"[-2:]"}
TRAJS=${TRAJS:-2}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Weights already cached locally; skip HF online HEAD probes.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
for d in "${GR1_DATASETS[@]}"; do
    DATASET_PATHS+=("${GR1_DATASET_DIR}/${d}")
done

SAVE_DIR="${MODEL_PATH}/eval_action_dual_system_${DATA_SPLIT}"

echo "=============================================="
echo "UniT open-loop action-loss evaluation"
echo "  MODEL_PATH:  $MODEL_PATH"
echo "  DATA_CONFIG: $DATA_CONFIG"
echo "  DATA_SPLIT:  $DATA_SPLIT"
echo "  TRAJS:       $TRAJS"
echo "  TASKS:       ${#DATASET_PATHS[@]}"
echo "  SAVE_DIR:    $SAVE_DIR"
echo "=============================================="

python3 -u scripts/eval_policy_unit.py \
    --dataset-path "${DATASET_PATHS[@]}" \
    --model_path "$MODEL_PATH" \
    --data-config "$DATA_CONFIG" \
    --embodiment_tag gr1 \
    --trajs "$TRAJS" \
    --data_split "$DATA_SPLIT" \
    --save_results_path "$SAVE_DIR" \
    --plot_state

echo "Done: $SAVE_DIR"
