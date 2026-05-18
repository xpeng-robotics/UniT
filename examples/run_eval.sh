#!/bin/bash
# UniT RoboCasa GR1 simulation evaluation.
# Usage:
#   bash examples/run_eval.sh <model_path> <eval_type>
#
# EVAL_TYPE:
#   id                         - In-distribution training tasks
#                                (6 PnPClose + 18 PosttrainPnPNovel SplitA)
#   ood_object_appearance      - OOD unseen object appearance
#                                (18 EvalPnPNovel SplitB)
#   ood_container_combination  - OOD unseen source-target container combo
#                                (14 PretrainPnPNovel SplitA)
#   ood_object_type            - OOD unseen object type
#                                (32 PretrainPnPBase SplitA)
#   unseen_close               - Held-out PnPClose variants
#                                (9 PnP*Close tasks)
#
# Environment variables:
#   DATA_CONFIG            (default: fourier_gr1_arms_waist_gausNorm_crop_cam_ego_joints_only)
#   N_ENVS                 (default: 1)
#   N_EPISODES             (default: 50)
#   CUDA_VISIBLE_DEVICES   (default: 0)
#   PORT                   (default: 5800 + first GPU id)
#   EVAL_TAG               (default: _run1)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH=${1:?"Usage: bash examples/run_eval.sh <model_path> <eval_type>"}
EVAL_TYPE=${2:?"Eval type: id | ood_object_appearance | ood_container_combination | ood_object_type | unseen_close"}

DATA_CONFIG=${DATA_CONFIG:-fourier_gr1_arms_waist_gausNorm_crop_cam_ego_joints_only}
N_ENVS=${N_ENVS:-1}
N_EPISODES=${N_EPISODES:-50}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
_CVD_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
PORT="${PORT:-$((5800 + _CVD_GPU))}"
EVAL_TAG=${EVAL_TAG:-_run1}
export CUDA_VISIBLE_DEVICES

case "$EVAL_TYPE" in
  id)
    EVAL_SUBDIR="evaluation_sim_id_${N_ENVS}envs${EVAL_TAG}"
    task_names=(
        gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
    ) ;;
  ood_object_appearance)
    EVAL_SUBDIR="evaluation_sim_ood_object_appearance_${N_ENVS}envs${EVAL_TAG}"
    task_names=(
        gr1_unified/EvalPnPNovelFromCuttingboardToBasketSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromCuttingboardToCardboardboxSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromCuttingboardToPanSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromCuttingboardToPotSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromCuttingboardToTieredbasketSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromPlacematToBasketSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromPlacematToBowlSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromPlacematToPlateSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromPlacematToTieredshelfSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromPlateToBowlSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromPlateToCardboardboxSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromPlateToPanSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromPlateToPlateSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromTrayToCardboardboxSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromTrayToPlateSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromTrayToPotSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromTrayToTieredbasketSplitB_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/EvalPnPNovelFromTrayToTieredshelfSplitB_GR1ArmsAndWaistFourierHands_Env
    ) ;;
  ood_container_combination)
    EVAL_SUBDIR="evaluation_sim_ood_container_combination_${N_ENVS}envs${EVAL_TAG}"
    task_names=(
        gr1_unified/PretrainPnPNovelFromCuttingboardToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromCuttingboardToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromCuttingboardToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromTrayToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromTrayToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromTrayToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromPlateToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromPlateToPotSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromPlateToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromPlateToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromPlacematToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromPlacematToPotSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromPlacematToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPNovelFromPlacematToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
    ) ;;
  ood_object_type)
    EVAL_SUBDIR="evaluation_sim_ood_object_type_${N_ENVS}envs${EVAL_TAG}"
    task_names=(
        gr1_unified/PretrainPnPBaseFromCuttingboardToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromCuttingboardToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromCuttingboardToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromTrayToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromTrayToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromTrayToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlateToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlateToPotSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlateToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlateToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlacematToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlacematToPotSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlacematToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlacematToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PretrainPnPBaseFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
    ) ;;
  unseen_close)
    EVAL_SUBDIR="evaluation_sim_unseen_close_${N_ENVS}envs${EVAL_TAG}"
    task_names=(
        gr1_unified/PnPCanToCabinetClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPBottleToDrawerClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPWineToDrawerClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPCupToCabinetClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPCupToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPEggplantToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPAppleToDrawerClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPAppleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
        gr1_unified/PnPCornToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
    ) ;;
  *)
    echo "Error: Unknown EVAL_TYPE '$EVAL_TYPE'."
    echo "Valid: id | ood_object_appearance | ood_container_combination | ood_object_type | unseen_close"
    exit 1 ;;
esac

echo "=============================================="
echo "UniT RoboCasa GR1 Evaluation"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  EVAL_TYPE:  $EVAL_TYPE  ($EVAL_SUBDIR)"
echo "  DATA_CONFIG:$DATA_CONFIG"
echo "  N_ENVS:     $N_ENVS"
echo "  N_EPISODES: $N_EPISODES"
echo "  PORT:       $PORT  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  TASKS:      ${#task_names[@]}"
echo "=============================================="

EVAL_DIR="${MODEL_PATH}/${EVAL_SUBDIR}"
mkdir -p "$EVAL_DIR"

CLIENT_LOG="${EVAL_DIR}/test_robocasa_gr1_client.log"
SERVER_LOG="${EVAL_DIR}/test_robocasa_gr1_server.log"
RESULTS_JSON="${EVAL_DIR}/results.json"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

nohup python3 -u scripts/inference_service_unit.py --server \
    --model_path "$MODEL_PATH" \
    --port "$PORT" \
    --data_config "$DATA_CONFIG" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID (log: $SERVER_LOG); waiting 30s for warm-up..."
sleep 30s

cleanup() {
    kill -9 "$SERVER_PID" 2>/dev/null || true
    ps aux | grep "port $PORT" | grep -v grep | awk '{print $2}' | xargs -r kill -9 || true
}
trap cleanup EXIT

{
for task_name in "${task_names[@]}"; do
    echo "Executing command for: $task_name PORT: ${PORT} MODEL_PATH: ${MODEL_PATH}"
    retry=0; max_retries=5; ok=0
    while [ $retry -lt $max_retries ] && [ $ok -eq 0 ]; do
        if python3 -u scripts/simulation/simulation_service_unit.py --client \
            --env_name "$task_name" \
            --video_dir "${EVAL_DIR}/videos/$task_name" \
            --max_episode_steps 720 \
            --n_envs "$N_ENVS" \
            --n_episodes "$N_EPISODES" \
            --port "$PORT"; then
            echo "Successfully executed: $task_name"
            ok=1
        else
            retry=$((retry+1))
            echo "Retry $retry/$max_retries for $task_name"
            sleep 5
        fi
    done
    [ $ok -eq 0 ] && echo "FAILED: $task_name"
    echo -e "\n==================================================\n"
done
} 2>&1 | tee "$CLIENT_LOG"

sleep 60s
python3 scripts/compute_success_rate.py -i "$CLIENT_LOG" -o "$RESULTS_JSON"
echo "Done: $RESULTS_JSON"
