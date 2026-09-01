"""Shared training and full-catalog evaluation for the main-table methods."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from seqrec.qlmc import qlmc_scores
from seqrec.recdata import NormalRecData
from seqrec.role_density import GateConfig, RoleDensitySASRec
from seqrec.utils import get_config, init_seed


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_ROOT = REPOSITORY_ROOT / "results/runs"

DATASET_PATHS = {
    "Games_5core": REPOSITORY_ROOT / "data/Video_Games/5-core/downstream",
    "Sports_5core": REPOSITORY_ROOT / "data/Sports_and_Outdoors/5-core/downstream",
    "Arts_5core": REPOSITORY_ROOT / "data/Arts_Crafts_and_Sewing/5-core/downstream",
    "Baby_5core": REPOSITORY_ROOT / "data/Baby_Products/5-core/downstream",
}
EMBEDDING_FILENAMES = {
    "LLM2Rec": "LLM2Rec_Qwen2-0.5B_item_embs.npy",
    "LLM2Vec": "LLM2Vec_Qwen2-0.5B_item_embs.npy",
    "LLMEmb": "LLMEmb_Qwen2-0.5B_item_embs.npy",
    "EasyRec": "EasyRec_roberta-large_item_embs_scaled40.npy",
    "BLAIR": "BLAIR_roberta-base_item_embs_scaled40.npy",
}
SEEDS = (2024, 2025, 2026)
METHODS = ("Base", "Symmetric", "Ours")
METHOD_DIRECTORIES = {"Base": "base", "Symmetric": "symmetric", "Ours": "ours"}
METRICS = ("recall@10", "ndcg@10", "recall@20", "ndcg@20")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def embedding_path(dataset: str, upstream: str) -> Path:
    return REPOSITORY_ROOT / "item_info" / dataset / EMBEDDING_FILENAMES[upstream]


def run_root(
    artifacts_root: Path,
    dataset: str,
    upstream: str,
    seed: int,
    method: str,
) -> Path:
    return artifacts_root / upstream / dataset / str(seed) / METHOD_DIRECTORIES[method]


def read_sequences(dataset: str, split: str) -> list[list[int]]:
    filename = {
        "train": "train_data.txt",
        "validation": "val_data.txt",
        "test": "test_data.txt",
    }[split]
    with (DATASET_PATHS[dataset] / filename).open() as handle:
        return [list(map(int, line.split()))[-11:] for line in handle if line.strip()]


def support_tensors(dataset: str, item_num: int) -> tuple[torch.Tensor, torch.Tensor]:
    positive = torch.zeros(item_num + 1, dtype=torch.long)
    history = torch.zeros_like(positive)
    for row in read_sequences(dataset, "train"):
        positive[row[-1]] += 1
        ids = torch.tensor(row[:-1], dtype=torch.long)
        if ids.numel():
            history.scatter_add_(0, ids, torch.ones_like(ids))
    return positive, history


class CachedSequenceDataset(Dataset):
    """Tensor-backed version of the repository sequence dataset."""

    def __init__(self, source) -> None:
        maximum = int(source.config["max_seq_length"])
        count = len(source.sequences)
        self.item_seqs = torch.zeros((count, maximum), dtype=torch.long)
        self.labels = torch.empty(count, dtype=torch.long)
        self.lengths = torch.empty(count, dtype=torch.long)
        for index, sequence in enumerate(source.sequences):
            history = sequence[:-1]
            self.item_seqs[index, :len(history)] = torch.tensor(history, dtype=torch.long)
            self.labels[index] = sequence[-1]
            self.lengths[index] = len(history)
        self.seq_type = source.seq_type

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, index: int) -> dict:
        return {
            "item_seqs": self.item_seqs[index],
            "labels": self.labels[index],
            "seq_lengths": self.lengths[index],
            "seq_ids": index,
            "seq_type": self.seq_type,
        }


def model_config(dataset: str, upstream: str, seed: int) -> tuple[dict, dict]:
    config = get_config(
        "SASRec",
        None,
        {
            "dataset": dataset,
            "embedding": str(embedding_path(dataset, upstream)),
            "rand_seed": seed,
            "model": "RoleDensitySASRec",
            "run_id": "main_table",
        },
    )
    init_seed(seed, config["reproducibility"])
    train, validation, test, select_pool, item_num = NormalRecData(config).load_data()
    config["select_pool"] = select_pool
    config["item_num"] = item_num
    config["eos_token"] = item_num + 1
    datasets = {
        "train": CachedSequenceDataset(train),
        "validation": CachedSequenceDataset(validation),
        "test": CachedSequenceDataset(test),
    }
    return config, datasets


def load_embedding(dataset: str, upstream: str, item_num: int, device: str) -> torch.Tensor:
    array = np.load(embedding_path(dataset, upstream))
    if array.ndim != 2 or array.shape[0] != item_num + 1:
        raise ValueError("embedding rows must match padding plus the complete item catalog")
    if not np.isfinite(array).all():
        raise ValueError("embedding contains a non-finite value")
    if not np.array_equal(array[0], np.zeros_like(array[0])):
        raise ValueError("embedding row zero must be the padding vector")
    return torch.from_numpy(array).float().to(device)


def make_model(
    dataset: str,
    upstream: str,
    seed: int,
    method: str,
    device: str,
) -> tuple[RoleDensitySASRec, dict, dict]:
    config, datasets = model_config(dataset, upstream, seed)
    positive, history = support_tensors(dataset, config["item_num"])
    nonzero = positive[positive > 0].float()
    if not nonzero.numel():
        raise ValueError("training positive support is empty")
    tau = float(nonzero.median())

    if method == "Base":
        history_config = GateConfig("off")
        candidate_config = GateConfig("off")
    elif method == "Symmetric":
        history_config = GateConfig("identity")
        candidate_config = GateConfig("identity")
    elif method == "Ours":
        history_config = GateConfig("identity")
        candidate_config = GateConfig("density", tau=tau, gamma=1.0)
    else:
        raise ValueError(method)

    model = RoleDensitySASRec(
        config,
        load_embedding(dataset, upstream, config["item_num"], device),
        positive.to(device),
        history.to(device),
        history_config,
        candidate_config,
    ).to(device)
    return model, config, datasets


@torch.no_grad()
def scores_for_batch(
    model: RoleDensitySASRec,
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = model.get_representation(batch).view(-1, model.config["hidden_size"])
    table = model.candidate_vectors()
    scores = query @ table.T
    scores[:, 0] = -torch.inf
    histories = batch["item_seqs"]
    targets = batch["labels"].view(-1)
    rows = torch.arange(targets.numel(), device=scores.device).unsqueeze(1).expand_as(histories)
    valid = histories != 0
    scores[rows[valid], histories[valid]] = -torch.inf
    scores[torch.arange(targets.numel(), device=scores.device), targets] = (
        query * table[targets]
    ).sum(1)
    return query, table, scores


def metric_arrays(ranking: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    matches = ranking.eq(targets.unsqueeze(1))
    result = {}
    for cutoff in (10, 20):
        hit = matches[:, :cutoff].any(1)
        position = matches[:, :cutoff].float().argmax(1)
        result[f"recall@{cutoff}"] = hit.float()
        result[f"ndcg@{cutoff}"] = torch.where(
            hit,
            1 / torch.log2(position.float() + 2),
            torch.zeros_like(position, dtype=torch.float),
        )
    return result


@torch.no_grad()
def validation_metrics(model: RoleDensitySASRec, dataset, device: str) -> dict:
    model.eval()
    values = {name: [] for name in METRICS}
    for batch in DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0):
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        _, _, scores = scores_for_batch(model, batch)
        metrics = metric_arrays(scores.topk(20, dim=1).indices, batch["labels"].view(-1))
        for name in METRICS:
            values[name].append(metrics[name].cpu())
    return {name: float(torch.cat(parts).mean()) for name, parts in values.items()}


def load_checkpoint(model: RoleDensitySASRec, path: Path) -> dict:
    value = torch.load(path, map_location=model.history_gate.device, weights_only=False)
    model.load_state_dict(value["model_state"], strict=True)
    return value


def train_task(
    dataset: str,
    upstream: str,
    seed: int,
    method: str,
    device: str,
    artifacts_root: Path,
) -> None:
    root = run_root(artifacts_root, dataset, upstream, seed, method)
    checkpoint = root / "best_checkpoint.pt"
    completed = root / "validation_metrics.json"
    if checkpoint.is_file() and completed.is_file():
        print(f"REUSE={checkpoint}", flush=True)
        return

    root.mkdir(parents=True, exist_ok=True)
    model, config, datasets = make_model(dataset, upstream, seed, method, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    generator = torch.Generator().manual_seed(seed + 104729)
    loader = DataLoader(
        datasets["train"],
        batch_size=config["train_batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    best_score = -math.inf
    best_epoch = 0
    start_epoch = 1
    epoch_rows = []
    if checkpoint.is_file():
        saved = load_checkpoint(model, checkpoint)
        optimizer.load_state_dict(saved["optimizer_state"])
        generator.set_state(saved["loader_generator_state"].cpu())
        best_epoch = int(saved["epoch"])
        best_score = float(saved["validation"][config["val_metric"]])
        start_epoch = best_epoch + 1
        print(f"RESUME={checkpoint};START_EPOCH={start_epoch}", flush=True)

    started = time.time()
    for epoch in range(start_epoch, config["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            query = model.get_representation(batch).view(-1, config["hidden_size"])
            logits = query @ model.candidate_vectors()[1:].T
            loss = model.loss_func(logits, batch["labels"].view(-1) - 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())

        row = {"epoch": epoch, "loss": total_loss / len(loader)}
        if epoch % config["eval_interval"] == 0:
            metrics = validation_metrics(model, datasets["validation"], device)
            row.update(metrics)
            print(
                json.dumps(
                    {
                        "dataset": dataset,
                        "upstream": upstream,
                        "seed": seed,
                        "method": method,
                        **row,
                    }
                ),
                flush=True,
            )
            score = metrics[config["val_metric"]]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "loader_generator_state": generator.get_state(),
                        "epoch": epoch,
                        "validation": metrics,
                    },
                    checkpoint,
                )
            elif epoch - best_epoch >= config["patience"]:
                epoch_rows.append(row)
                break
        epoch_rows.append(row)

    pd.DataFrame(epoch_rows).to_csv(root / "training_history.csv", index=False)
    best = load_checkpoint(model, checkpoint)
    write_json(
        completed,
        {
            "best_epoch": int(best["epoch"]),
            "elapsed_seconds": time.time() - started,
            **best["validation"],
        },
    )
    print(f"CHECKPOINT={checkpoint}", flush=True)


@torch.no_grad()
def full_catalog_metrics(
    model: RoleDensitySASRec,
    dataset,
    device: str,
    apply_qlmc: bool,
) -> dict:
    model.eval()
    seen = model.positive_support > 0
    seen[0] = False
    values = {name: [] for name in METRICS}
    query_count = 0

    for batch in DataLoader(dataset, batch_size=384, shuffle=False, num_workers=0):
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        query, _, scores = scores_for_batch(model, batch)
        targets = batch["labels"].view(-1)
        histories = batch["item_seqs"]

        if apply_qlmc:
            base_table = model.base_candidate_table()
            base_scores = query @ base_table.T
            base_scores[:, 0] = -torch.inf
            rows = torch.arange(targets.numel(), device=device).unsqueeze(1).expand_as(histories)
            valid = histories != 0
            base_scores[rows[valid], histories[valid]] = -torch.inf
            base_scores[torch.arange(targets.numel(), device=device), targets] = (
                query * base_table[targets]
            ).sum(1)
            scores, _ = qlmc_scores(
                base_scores,
                scores,
                seen,
                local_top_l=100,
                lambda_q=1.0,
            )

        metrics = metric_arrays(scores.topk(20, dim=1).indices, targets)
        for name in METRICS:
            values[name].append(metrics[name].cpu())
        query_count += targets.numel()

    return {
        "query_count": int(query_count),
        **{name: float(torch.cat(parts).mean()) for name, parts in values.items()},
    }


def evaluate_task(
    dataset: str,
    upstream: str,
    seed: int,
    method: str,
    device: str,
    artifacts_root: Path,
) -> None:
    root = run_root(artifacts_root, dataset, upstream, seed, method)
    checkpoint = root / "best_checkpoint.pt"
    output = root / "test_metrics.json"
    if output.is_file():
        print(f"REUSE={output}", flush=True)
        return
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model, _, datasets = make_model(dataset, upstream, seed, method, device)
    load_checkpoint(model, checkpoint)
    metrics = full_catalog_metrics(
        model,
        datasets["test"],
        device,
        apply_qlmc=method == "Ours",
    )
    result = {
        "dataset": dataset,
        "upstream": upstream,
        "seed": seed,
        "method": method,
        **metrics,
    }
    write_json(output, result)
    print(json.dumps(result), flush=True)


def method_cli(method: str) -> None:
    parser = argparse.ArgumentParser(
        description=f"Train and evaluate {method} for the main table."
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_PATHS))
    parser.add_argument("--upstream", choices=tuple(EMBEDDING_FILENAMES))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    args = parser.parse_args()

    datasets = (args.dataset,) if args.dataset else tuple(DATASET_PATHS)
    upstreams = (args.upstream,) if args.upstream else tuple(EMBEDDING_FILENAMES)
    seeds = (args.seed,) if args.seed else SEEDS

    for dataset in datasets:
        for upstream in upstreams:
            for seed in seeds:
                print(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "upstream": upstream,
                            "seed": seed,
                            "method": method,
                            "stage": args.stage,
                        }
                    ),
                    flush=True,
                )
                if args.stage in {"train", "all"}:
                    train_task(
                        dataset,
                        upstream,
                        seed,
                        method,
                        args.device,
                        args.artifacts_root,
                    )
                if args.stage in {"evaluate", "all"}:
                    evaluate_task(
                        dataset,
                        upstream,
                        seed,
                        method,
                        args.device,
                        args.artifacts_root,
                    )
