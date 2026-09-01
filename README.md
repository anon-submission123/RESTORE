# Residuals Are Not Uniform: Calibrating Collaborative Residuals for LLM-Enhanced Sequential Recommendation

This repository provides the code for training and evaluating **Base**,
**Symmetric**, and **RESTORE**, and for generating the main result table.

## Datasets

- `Games`
- `Sports`
- `Arts`
- `Baby`

Download the processed datasets from [Google Drive](https://drive.google.com/file/d/12CytdAvXAOjConlzYyJQfh6ZTJo_6iYm/view?usp=sharing).
Please unzip the dataset files under directory `./data`.

## Upstream Embeddings

The upstream item embeddings used in these experiments were produced with the
official implementations from the following repositories:

| Upstream | Source repository |
| --- | --- |
| `LLM2Rec` | [HappyPointer/LLM2Rec](https://github.com/HappyPointer/LLM2Rec) |
| `LLM2Vec` | [McGill-NLP/llm2vec](https://github.com/McGill-NLP/llm2vec) |
| `LLMEmb` | [Applied-Machine-Learning-Lab/LLMEmb](https://github.com/Applied-Machine-Learning-Lab/LLMEmb) |
| `EasyRec` | [HKUDS/EasyRec](https://github.com/HKUDS/EasyRec) |
| `BLAIR` | [hyp1231/AmazonReviews2023](https://github.com/hyp1231/AmazonReviews2023) |

Please place the upstream embedding files under directory `./item_info`.

## Requirements

```bash
python -m pip install -r requirements.txt
```

## Running Experiments

Each command runs seeds `2024`, `2025`, and `2026`.

### Run One Method in Full

Each program runs its method across all datasets and upstream representations:

```bash
python train_base.py --device cuda:0
python train_ours.py --device cuda:0
python train_symmetric.py --device cuda:0
```

Run all three programs in this section before summarizing the complete results.

### Run One Selected Combination

Specify both the dataset and upstream representation. The selected method runs
that combination for all three seeds:

```bash
python train_base.py --dataset Games_5core --upstream LLM2Rec --device cuda:0
python train_ours.py --dataset Games_5core --upstream LLM2Rec --device cuda:0
python train_symmetric.py --dataset Games_5core --upstream LLM2Rec --device cuda:0
```

This mode produces only the selected combination and does not by itself provide
all results required for the complete summary.

Each seed produces:

- `best_checkpoint.pt`
- `validation_metrics.json`
- `training_history.csv`
- `test_metrics.json`

Run artifacts are organized by upstream representation, dataset, seed, and
method under `./results/runs`. `training_history.csv` contains per-epoch records.

## Summarizing Results

```bash
python summarize_results.py
```

This command aggregates the three seeds for Base, Symmetric, and RESTORE and
produces `./results/summary_results.csv` and `./results/summary_results.md`.
