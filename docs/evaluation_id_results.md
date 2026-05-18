# In-distribution evaluation — per-task success rates

Closed-loop RoboCasa GR1 simulation results from `examples/run_eval.sh <ckpt> id`
(24 ID tasks × 50 episodes = 1200 rollouts, `N_ENVS=1`, `DATA_CONFIG=fourier_gr1_arms_waist_gausNorm_crop_cam_ego_joints_only`).

> Rollouts are stochastic, so a single run is **not** a point estimate—expect roughly ±1 percentage point of fluctuation on the overall success rate across reruns.

Aggregates reported by `scripts/compute_success_rate.py`:

| Recipe | Overall | PnPClose | PnPOnly (Posttrain SplitA) |
|---|---|---|---|
| GR1-full (`run_gr1_full.sh`, `checkpoint-160000`) | **66.4%** | 70.3% | 65.1% |
| EgoDex + GR1-100 (`run_gr1_100_egodex.sh`, `checkpoint-20000`) | **50.9%** | 53.0% | 50.2% |

## Per-task success rate

| Task | GR1-full | EgoDex + GR1-100 |
|---|---:|---:|
| PnPCupToDrawerClose | 0.64 | 0.50 |
| PnPPotatoToMicrowaveClose | 0.72 | 0.54 |
| PnPMilkToMicrowaveClose | 0.76 | 0.62 |
| PnPBottleToCabinetClose | 0.86 | 0.54 |
| PnPWineToCabinetClose | 0.48 | 0.34 |
| PnPCanToDrawerClose | 0.76 | 0.64 |
| PosttrainPnPNovelFromCuttingboardToBasketSplitA | 0.74 | 0.66 |
| PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA | 0.62 | 0.50 |
| PosttrainPnPNovelFromCuttingboardToPanSplitA | 0.90 | 0.68 |
| PosttrainPnPNovelFromCuttingboardToPotSplitA | 0.78 | 0.52 |
| PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA | 0.78 | 0.38 |
| PosttrainPnPNovelFromPlacematToBasketSplitA | 0.60 | 0.38 |
| PosttrainPnPNovelFromPlacematToBowlSplitA | 0.68 | 0.56 |
| PosttrainPnPNovelFromPlacematToPlateSplitA | 0.74 | 0.36 |
| PosttrainPnPNovelFromPlacematToTieredshelfSplitA | 0.44 | 0.22 |
| PosttrainPnPNovelFromPlateToBowlSplitA | 0.58 | 0.52 |
| PosttrainPnPNovelFromPlateToCardboardboxSplitA | 0.54 | 0.46 |
| PosttrainPnPNovelFromPlateToPanSplitA | 0.66 | 0.50 |
| PosttrainPnPNovelFromPlateToPlateSplitA | 0.78 | 0.50 |
| PosttrainPnPNovelFromTrayToCardboardboxSplitA | 0.60 | 0.54 |
| PosttrainPnPNovelFromTrayToPlateSplitA | 0.68 | 0.62 |
| PosttrainPnPNovelFromTrayToPotSplitA | 0.56 | 0.64 |
| PosttrainPnPNovelFromTrayToTieredbasketSplitA | 0.56 | 0.58 |
| PosttrainPnPNovelFromTrayToTieredshelfSplitA | 0.48 | 0.42 |

Task names share the suffix `_GR1ArmsAndWaistFourierHands_Env` in the raw
`results.json`; it is omitted above for readability.
