import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from method.model import GazeSearchGlobalLocalModel
from method.dataset_gaze import (
    DEFAULT_TEXT_MODEL,
    GazeSearchDataset,
    GazeSearchImageDataset,
)
from method.utils.loss import compute_loss
from method.utils.metrics import evaluate_scanpaths
from method.test import generate_predictions
from method.utils.misc import (
    load_checkpoint,
    move_batch,
    save_checkpoint,
    set_seed,
    write_json,
)
from Data.dataset import ground_truth_scanpaths

DEFAULT_DATA_ROOT = PROJECT_ROOT / "Data"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output"


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch,
    termination_pos_weight,
    attention_map_weight=0.1,
    lambda_geo=0.1,
    clinical_duration_sigma=1.0 / 7.0,
    duration_loss_weight=1.0,
    max_grad_norm=1.0,
    sed_grid_loss_weight=0.2,
    consistency_loss_weight=0.1,
):
    model.train()
    totals = defaultdict(float)
    batches = 0
    progress = tqdm(data_loader, desc=f"train {epoch:03d}")
    for batch_index, batch in enumerate(progress):
        batch = move_batch(batch, device)
        outputs = model(
            batch["image"],
            batch["task_id"],
            batch["history"],
            batch["padding_mask"],
            batch["global_text_feature"],
            batch["local_text_features"],
        )
        losses = compute_loss(
            outputs,
            batch,
            termination_pos_weight=termination_pos_weight,
            attention_map_weight=attention_map_weight,
            lambda_geo=lambda_geo,
            clinical_duration_sigma=clinical_duration_sigma,
            duration_loss_weight=duration_loss_weight,
            sed_grid_loss_weight=sed_grid_loss_weight,
            consistency_loss_weight=consistency_loss_weight,
        )
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError(
                f"Non-finite loss at batch {batch_index}: {losses}"
            )
        if any((float(value.detach()) < -1e-06 for value in losses.values())):
            raise FloatingPointError(
                f"A loss component became negative at batch {batch_index}: {losses}"
            )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        batches += 1
        for name, value in losses.items():
            totals[name] += float(value.detach())
        progress.set_postfix(loss=f"{float(losses['loss'].detach()):.4f}")
    if not batches:
        raise RuntimeError("Training DataLoader produced no batches")
    return {name: value / batches for name, value in totals.items()}


@torch.no_grad()
def evaluate_loss(
    model,
    data_loader,
    device,
    termination_pos_weight,
    attention_map_weight=0.1,
    lambda_geo=0.1,
    clinical_duration_sigma=1.0 / 7.0,
    duration_loss_weight=1.0,
    sed_grid_loss_weight=0.2,
    consistency_loss_weight=0.1,
):
    model.eval()
    totals = defaultdict(float)
    batches = 0
    for batch in tqdm(data_loader, desc="valid"):
        batch = move_batch(batch, device)
        outputs = model(
            batch["image"],
            batch["task_id"],
            batch["history"],
            batch["padding_mask"],
            batch["global_text_feature"],
            batch["local_text_features"],
        )
        losses = compute_loss(
            outputs,
            batch,
            termination_pos_weight=termination_pos_weight,
            attention_map_weight=attention_map_weight,
            lambda_geo=lambda_geo,
            clinical_duration_sigma=clinical_duration_sigma,
            duration_loss_weight=duration_loss_weight,
            sed_grid_loss_weight=sed_grid_loss_weight,
            consistency_loss_weight=consistency_loss_weight,
        )
        batches += 1
        for name, value in losses.items():
            totals[name] += float(value)
    if not batches:
        raise RuntimeError("Validation DataLoader produced no batches")
    return {name: value / batches for name, value in totals.items()}


def get_args_parser():
    parser = argparse.ArgumentParser(description="Train HiSTAR on GazeSearch")
    parser.add_argument(
        "--pretrained-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize ResNet-50 with ImageNet weights.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--train-backbone", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--task-condition-scale", type=float, default=0.5)
    parser.add_argument("--attention-map-weight", type=float, default=0.1)
    parser.add_argument("--attention-lambda1", type=float, default=1.0)
    parser.add_argument("--attention-lambda2", type=float, default=-1.0)
    parser.add_argument("--history-gaussian-sigma", type=float, default=0.75)
    parser.add_argument("--lambda-geo", type=float, default=0.1)
    parser.add_argument(
        "--clinical-duration-sigma",
        type=float,
        default=1.0 / 7.0,
        help="Gaussian duration-field sigma in normalized 7x7 region coordinates.",
    )
    parser.add_argument("--duration-loss-weight", type=float, default=1.0)
    parser.add_argument("--sed-grid-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--consistency-loss-weight",
        type=float,
        default=0.1,
        help="Weight of CSCC clinical-significance and saccade-geometry consistency.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument(
        "--selection-metric",
        choices=("scanmatch", "validation_loss"),
        default="scanmatch",
        help="Choose checkpoint_best by spatial ScanMatch or teacher-forced validation loss.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=8)
    return parser


def _experiment_directory(args):
    if args.resume:
        return args.resume.expanduser().resolve().parent
    timestamp = datetime.now().strftime("experiment_%Y%m%d_%H%M%S")
    return args.output_dir.expanduser().resolve() / timestamp


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested but is not available")
    train_dataset = GazeSearchDataset(
        args.data_root, split="train", text_model=args.text_model
    )
    valid_dataset = GazeSearchDataset(
        args.data_root, split="valid", text_model=args.text_model
    )
    valid_image_dataset = GazeSearchImageDataset(
        args.data_root, split="valid", text_model=args.text_model
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_options)
    valid_image_loader = DataLoader(
        valid_image_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_ground_truth = ground_truth_scanpaths(valid_image_dataset)
    terminal_count = len(train_dataset.records)
    termination_pos_weight = (len(train_dataset) - terminal_count) / max(
        terminal_count, 1
    )
    model = GazeSearchGlobalLocalModel(
        weights=args.pretrained_backbone,
        freeze_backbone=not args.train_backbone,
        dropout=args.dropout,
        attention_lambda1=args.attention_lambda1,
        attention_lambda2=args.attention_lambda2,
        history_gaussian_sigma=args.history_gaussian_sigma,
        task_condition_scale=args.task_condition_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-06
    )
    experiment_dir = _experiment_directory(args)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    serializable_args.update(
        {
            "model_class": "GazeSearchGlobalLocalModel",
            "text_encoder": args.text_model,
            "text_embedding_dim": 128,
            "local_grid": [7, 7],
        }
    )
    write_json(experiment_dir / "args.json", serializable_args)
    start_epoch, best_validation_loss = (0, float("inf"))
    best_selection_value = (
        -float("inf") if args.selection_metric == "scanmatch" else float("inf")
    )
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(
            checkpoint.get("best_validation_loss", float("inf"))
        )
        best_selection_value = float(
            checkpoint.get("best_selection_value", best_selection_value)
        )
    log_path = experiment_dir / "train_log.txt"
    epochs_without_improvement = 0
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            termination_pos_weight,
            args.attention_map_weight,
            args.lambda_geo,
            args.clinical_duration_sigma,
            args.duration_loss_weight,
            args.max_grad_norm,
            args.sed_grid_loss_weight,
            args.consistency_loss_weight,
        )
        valid_metrics = evaluate_loss(
            model,
            valid_loader,
            device,
            termination_pos_weight,
            args.attention_map_weight,
            args.lambda_geo,
            args.clinical_duration_sigma,
            args.duration_loss_weight,
            args.sed_grid_loss_weight,
            args.consistency_loss_weight,
        )
        valid_scanpath_metrics = None
        if args.selection_metric == "scanmatch":
            valid_predictions = generate_predictions(
                model, valid_image_loader, device, sample=False, max_batches=0
            )
            valid_scanpath_metrics, _ = evaluate_scanpaths(
                valid_predictions, valid_ground_truth
            )
        scheduler.step(valid_metrics["loss"])
        result = {
            "epoch": epoch,
            "train": train_metrics,
            "valid": valid_metrics,
            "valid_scanpath": valid_scanpath_metrics,
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(json.dumps(result, indent=2))
        selection_value = (
            valid_scanpath_metrics["ScanMatch w/o Dur."]
            if args.selection_metric == "scanmatch"
            else valid_metrics["loss"]
        )
        is_best = (
            selection_value > best_selection_value
            if args.selection_metric == "scanmatch"
            else selection_value < best_selection_value
        )
        if is_best:
            best_selection_value = selection_value
        best_validation_loss = min(best_validation_loss, valid_metrics["loss"])
        epochs_without_improvement = 0 if is_best else epochs_without_improvement + 1
        save_checkpoint(
            experiment_dir / "checkpoint_last.pth",
            model,
            optimizer,
            epoch,
            serializable_args,
            best_validation_loss=best_validation_loss,
            best_selection_value=best_selection_value,
            selection_metric=args.selection_metric,
        )
        save_checkpoint(
            experiment_dir / f"checkpoint_epoch_{int(epoch)}.pth",
            model,
            optimizer,
            epoch,
            serializable_args,
            best_validation_loss=best_validation_loss,
            best_selection_value=best_selection_value,
            selection_metric=args.selection_metric,
        )
        if is_best:
            save_checkpoint(
                experiment_dir / "checkpoint_best.pth",
                model,
                optimizer,
                epoch,
                serializable_args,
                best_validation_loss=best_validation_loss,
                best_selection_value=best_selection_value,
                selection_metric=args.selection_metric,
            )
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Early stopping after {epochs_without_improvement} epochs without selection-metric improvement"
            )
            break
    print(f"Training output: {experiment_dir}")
    return experiment_dir


if __name__ == "__main__":
    main(get_args_parser().parse_args())
