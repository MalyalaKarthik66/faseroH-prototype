# Copilot Instructions for FASEROH Prototype

## Project Scope
- This repository is a research prototype for symbolic seq2seq learning.
- Current task is function-to-Taylor expansion translation; architecture is designed to later swap to histogram encoders.
- Keep changes minimal and modular; avoid broad refactors unless explicitly requested.

## Quick Start Commands
- Create/activate env (Windows):
  - `python -m venv .wenv`
  - `.\.wenv\Scripts\activate`
  - `pip install -r requirements.txt`
- Run tests:
  - `pytest`
  - `pytest tests/test_tokenizer.py`
  - `pytest tests/test_metrics.py`
- Run experiments:
  - `python experiments/run_experiment.py --config configs/transformer.yaml --results-dir results/transformer`
  - `python experiments/run_experiment.py --config configs/lstm.yaml --results-dir results/lstm`
- Demo:
  - `python demo.py`

## Architecture Boundaries
- `data/`: synthetic expression generation, dataset validation, and data loading.
- `tokenizer/`: infix/prefix conversion, vocabulary, encode/decode behavior.
- `models/`: seq2seq model implementations (`lstm_seq2seq.py`, `transformer_seq2seq.py`) and encoder/decoder submodules.
- `training/`: training loop/orchestration (`train.py`, `trainer.py`) including optimization, scheduling, and logging.
- `evaluation/`: syntactic, semantic, and coefficient metrics plus reporting.
- `experiments/`: end-to-end orchestration and comparison scripts.

## Project Conventions
- Expression representation is prefix-token based in model-facing flows.
- Keep config-driven behavior in `configs/*.yaml`; avoid hardcoding hyperparameters.
- Preserve result artifact layout under `results/<run>/` (metrics, predictions, checkpoints, vocab).
- Prefer deterministic behavior for experiments: honor existing seed plumbing.
- Maintain device-agnostic code paths (`auto`/CPU/CUDA handling).

## Common Pitfalls
- `run_experiment.py` regenerates data unless `--skip-data` is passed and dataset exists.
- SymPy simplification/semantic checks can be sensitive to expression validity; avoid loosening validation without tests.
- Changes to tokenization or vocabulary often require updating tests and re-generating run artifacts.

## High-Value Files
- Entry pipeline: [experiments/run_experiment.py](experiments/run_experiment.py)
- Training core: [training/trainer.py](training/trainer.py)
- Tokenization core: [tokenizer/tokenizer.py](tokenizer/tokenizer.py)
- Model implementations: [models/lstm_seq2seq.py](models/lstm_seq2seq.py), [models/transformer_seq2seq.py](models/transformer_seq2seq.py)
- Primary docs: [README.md](README.md), [PROJECT_REVIEW.md](PROJECT_REVIEW.md)

## Link, Don’t Duplicate
- For system overview and narrative context, link to [README.md](README.md) and [PROJECT_REVIEW.md](PROJECT_REVIEW.md) rather than copying long explanations.
- For run configuration details, reference specific YAML files in `configs/`.

## Change Expectations for Agents
- Fix root causes; do not patch around issues with one-off conditionals unless requested.
- Keep public behavior stable unless user asks for behavior changes.
- Validate changes with the most targeted tests first, then broader tests if needed.