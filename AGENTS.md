# Repository Guidelines

## Project Structure & Module Organization
The training entry point is `main.py`, which wires models defined under `models/` through optimizer and loss helpers in `engine.py` and `util/`. Dataset adapters, COCO evaluation tools, and panoptic helpers live in `datasets/`, while `d2/` wraps Detectron2 integration, and `embedding/` hosts scripts such as `extract_embeddings.py` and `test_extraction.sh` for generating FACE/skin embeddings. Assets like `FACET annotations/` and experiment shell scripts (`run_all_*.sh`) keep dataset splits and benchmarking recipes. Place new experiments alongside these scripts so contributors can discover them quickly.

## Build, Test, and Development Commands
Use a CUDA-ready Conda/virtualenv, then install deps:
```
pip install -r requirements.txt
```
Typical workflows:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /data/coco
python main.py --batch_size 2 --no_aux_loss --eval --resume <checkpoint> --coco_path /data/coco
python run_with_submitit.py --timeout 3000 --coco_path /data/coco
python -m unittest test_all.py
```
Keep dataset paths consistent (e.g., `/data/coco/{train2017,val2017,annotations}`) and pin seeds when comparing runs.

## Coding Style & Naming Conventions
The repo follows standard Python 3.9+ style: 4-space indentation, snake_case functions, and PascalCase modules/classes (`HungarianMatcher`). Run `python -m flake8` (configured via `tox.ini`, 120-char limit, selected ignore list) before sending changes. Prefer `torch` type annotations (`Tensor`) and descriptive variable names that match DETR terminology (queries, matcher, postprocessors). Organize imports as stdlib, third-party, local.

## Testing Guidelines
Unit tests live in `test_all.py` (unittest) and should cover matcher logic, position encoders, TorchScript export, and ONNX paths. Extend this module or mimic its patterns when adding new utilities; target deterministic tensors and seed Torch. Run `python -m unittest test_all.py` locally and capture relevant logs or screenshots showing new metrics. For GPU-heavy scripts, consider adding lightweight input shapes so CI stays fast.

## Commit & Pull Request Guidelines
Recent history favors concise subject lines describing the changed area (e.g., “run_all scripts add per-skin eval”). Follow the same pattern: start with the subsystem, keep to ~72 chars, add optional Korean context after a hyphen if needed. PRs should explain the motivation, list new scripts/configs, link tracked issues, and include: dataset/experiment arguments, training duration or GPUs, evaluation tables, and any migration notes (new files in `FACET annotations/`, renamed checkpoints, etc.). Always mention if checkpoints or data formats change so downstream automation can be updated.
