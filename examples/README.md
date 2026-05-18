# UniT example training pipelines

End-to-end pipeline scripts that reproduce the headline GR1-Joints recipes from the paper. For installation, data preparation, the full evaluation walkthrough, and `unit_cfg` notes, see the root [README](../README.md). The summary below only documents what is **not** already covered there: per-stage step counts and the environment-variable defaults consumed by `run_gr1_*.sh`.

## Pipeline stages

| Script | Stages (steps) |
| --- | --- |
| `run_gr1_full.sh` | tokenizer (80k) → dual-system pretrain (160k) |
| `run_gr1_100_egodex.sh` | tokenizer (80k) → dual-system pretrain (20k, EgoDex+GR1) → dual-system finetune (20k, GR1-only) |

Each pipeline runs sequentially under `set -euo pipefail`; a failing stage aborts the run, and the next stage resumes from the checkpoint produced by the previous one. Stage outputs land under `${OUTPUT_ROOT}/{tokenizer,pretrain,finetune}`.

## Environment overrides

| Variable | Default | Purpose |
| --- | --- | --- |
| `GR1_DATASET_DIR` | `/path/to/gr1_lerobot` | LeRobot root containing the 24 `gr1_unified.*` datasets |
| `EGODEX_DATASET_DIR` | `/path/to/egodex_lerobot` | EgoDex LeRobot root (only used by `run_gr1_100_egodex.sh`) |
| `OUTPUT_ROOT` | `outputs/example_<name>` | Where all stage outputs are written |
| `NUM_GPUS` | `8` | Forwarded to `--num-gpus` |

Other knobs (batch size, learning rate, tune-* flags, …) are inlined in the scripts to keep them self-contained and match the paper recipe.
