# FASEROH Prototype

**Fast Accurate Symbolic Empirical Representation Of Histograms**

A symbolic sequence-to-sequence learning prototype built for the ML4SCI GSoC project. The system learns to map numerical representations to closed-form symbolic mathematical expressions — framed as a translation problem between sequences of tokens.

The current validated task maps a symbolic function to its truncated Taylor expansion around `x = 0` (orders 0–4). This serves as the surrogate step before the full histogram-to-symbolic pipeline is assembled.

---

## Overview

| Component | Description |
|---|---|
| Task | Symbolic function → Taylor expansion (surrogate) |
| Architecture | LSTM + attention / Transformer encoder-decoder |
| Tokenization | Prefix notation with rational coefficient support |
| Training | AdamW, LR scheduling, FP16, early stopping |
| Evaluation | Syntactic, semantic (SymPy), and coefficient metrics |
| Hardware | CUDA auto-selection (validated on RTX 3050) |

---

## Repository Structure

```
faseroH-prototype/
├── configs/               # YAML experiment configuration files
│   ├── transformer.yaml
│   ├── lstm.yaml
│   └── transformer_5epochs.yaml
├── data/                  # Dataset generation, validation, dataloaders
│   ├── generate_expression_trees.py
│   ├── generate_taylor_dataset.py
│   ├── histogram_sim.py
│   └── validate_dataset.py
├── tokenizer/             # Prefix conversion, vocabulary, tokenizer
│   ├── prefix_converter.py
│   ├── tokenizer.py
│   └── vocabulary.py
├── models/                # LSTM and Transformer seq2seq implementations
│   ├── encoders/
│   ├── decoders/
│   ├── lstm_seq2seq.py
│   └── transformer_seq2seq.py
├── training/              # Training entrypoint and trainer logic
│   ├── train.py
│   └── trainer.py
├── evaluation/            # Metrics and standalone evaluation scripts
│   ├── syntactic_metrics.py
│   ├── semantic_metrics.py
│   ├── coefficient_metrics.py
│   ├── run_evaluation.py
│   ├── error_analysis.py
│   └── error_report.py
├── experiments/           # Model comparison and qualitative analysis
│   ├── run_experiment.py
│   ├── compare_models.py
│   ├── inference.py
│   └── generate_examples.py
├── notebooks/             # Analysis and visualization notebooks
│   ├── dataset_analysis.ipynb
│   └── results_analysis.ipynb
├── tests/                 # Pytest coverage for core components
├── results/               # Checkpoints, metrics, predictions, reports
├── demo.py                # CLI inference tool
├── setup.py
└── requirements.txt
```

---

## Environment Setup

```powershell
python -m venv .wenv
.\.wenv\Scripts\activate
pip install -r requirements.txt
```

All commands below use `.wenv\Scripts\python.exe` explicitly for full reproducibility.

---

## Quickstart

The full pipeline runs in three steps: generate data, train, evaluate.

### Step 1 — Generate Dataset

```powershell
$proj = 'c:\Users\karth\ML4SCIPROTO2\faseroH-prototype'
Push-Location $proj
& "$proj\.wenv\Scripts\python.exe" data/generate_taylor_dataset.py `
    --size 20000 --seed 42 --output data/taylor_dataset_20k.csv
Pop-Location
```

Optionally validate the generated dataset:

```powershell
& "$proj\.wenv\Scripts\python.exe" data/validate_dataset.py `
    --input data/taylor_dataset_20k.csv
```

Expected output: `valid_rows=20000/20000`

---

### Step 2 — Train

```powershell
& "$proj\.wenv\Scripts\python.exe" training/train.py `
    --config configs/transformer.yaml `
    --results-dir results/transformer_20k
```

Key defaults in `configs/transformer.yaml`:

| Parameter | Value |
|---|---|
| Dataset size | 20,000 |
| Epochs | 30 |
| Beam width | 1 (greedy during training) |
| Device | auto (CUDA if available) |

If CUDA is detected, the log will show:
```
[DEVICE] Using device: cuda
```

Checkpoints are saved to `results/transformer_20k/checkpoints/best_model.pt`.

---

### Step 3 — Evaluate

Training and evaluation are intentionally decoupled. Evaluation loads the saved checkpoint and runs the full metric suite — including beam search and SymPy equivalence checks — without re-running any training.

```powershell
& "$proj\.wenv\Scripts\python.exe" evaluation/run_evaluation.py
```

Reads from:
- Config: `configs/transformer.yaml`
- Checkpoint: `results/transformer_20k/checkpoints/best_model.pt`

Writes to:
- `results/transformer_20k/metrics_full.json`

---

## Metrics

| Category | Metric | Description |
|---|---|---|
| Syntactic | `token_accuracy` | Per-token prediction accuracy |
| Syntactic | `sequence_accuracy` | Exact full-sequence match rate |
| Syntactic | `bleu` | BLEU score over token sequences |
| Syntactic | `edit_distance` | Average token-level edit distance |
| Semantic | `semantic_equivalence` | SymPy symbolic equality check |
| Numerical | `coefficient_mse` | MSE between Taylor coefficient vectors |

---

## Running Both Models

To run and compare both LSTM and Transformer end-to-end:

```powershell
# Transformer
& "$proj\.wenv\Scripts\python.exe" experiments/run_experiment.py `
    --config configs/transformer.yaml --results-dir results/transformer

# LSTM (skip data generation since dataset already exists)
& "$proj\.wenv\Scripts\python.exe" experiments/run_experiment.py `
    --config configs/lstm.yaml --results-dir results/lstm --skip-data
```

Generate a side-by-side comparison report:

```powershell
& "$proj\.wenv\Scripts\python.exe" experiments/compare_models.py `
    --lstm results/lstm/metrics.json `
    --transformer results/transformer/metrics.json `
    --output results/model_comparison.md
```

Generate qualitative prediction examples:

```powershell
& "$proj\.wenv\Scripts\python.exe" experiments/generate_examples.py `
    --input results/transformer/sample_predictions.json `
    --output results/qualitative_examples.md --limit 12
```

Generate error analysis report:

```powershell
& "$proj\.wenv\Scripts\python.exe" evaluation/error_report.py `
    --input results/transformer/sample_predictions.json `
    --output results/error_analysis.md
```

---

## Demo — Single Expression Inference

Run inference on a single symbolic expression from the command line:

```powershell
& "$proj\.wenv\Scripts\python.exe" demo.py
```

Loads the trained checkpoint and vocabulary, runs greedy or beam decoding (per config), and prints the predicted Taylor expansion.

---

## Results (20,000-sample Surrogate Task)

| Model | Token Acc. | Seq. Acc. | BLEU | Edit Dist. | Semantic Eq. | Coeff. MSE |
|---|---|---|---|---|---|---|
| LSTM + Attention | 0.8175 | 0.6865 | 0.720 | 1.543 | 0.773 | 12879.56 |
| Transformer | 0.4224 | 0.0515 | 0.186 | 4.912 | 0.055 | 46537.52 |

The LSTM model (trained for 40 epochs) achieves strong performance across all metrics. The Transformer model (trained for 30 epochs) shows lower performance at this dataset scale, which is expected, as Transformer architectures typically require larger datasets to effectively capture structured symbolic relationships.

---

## Design Philosophy

Training and evaluation are strictly separated. Training runs only forward and backward passes — no beam search or SymPy checks during epochs. This keeps training fast and clean. Evaluation runs once on the held-out test set after training completes.

The architecture is modular by design. The `TextEncoder` (current surrogate input) and `HistogramEncoder` (final FASEROH input) share the same decoder interface. Swapping them requires no changes to the tokenizer, vocabulary, decoder, or evaluation pipeline.

---

## License

This project is developed as part of the ML4SCI GSoC 2026 FASEROH project.