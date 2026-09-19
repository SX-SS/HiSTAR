import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from method.model import GazeSearchGlobalLocalModel
from method.utils.metrics import (
    METRIC_NAMES,
    evaluate_scanpaths,
    metric_standard_deviations,
)
from method.utils.misc import (
    load_checkpoint,
    save_scanpath_visualization,
    set_seed,
    write_json,
)
from Data.dataset import DEFAULT_DATA_ROOT, ground_truth_scanpaths
from method.dataset_gaze import DEFAULT_TEXT_MODEL, GazeSearchImageDataset

OUTPUT_ROOT = Path(__file__).resolve().parent / "output"


def get_args_parser():
    parser = argparse.ArgumentParser(description="Evaluate HiSTAR on GazeSearch")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument(
        "--edge-margin",
        type=int,
        default=2,
        help="Exclude this many outer pixels during generation.",
    )
    parser.add_argument(
        "--diagnostic-pairs",
        type=int,
        default=0,
        help="Optional extra same-image/different-task forward passes (default: 0, disabled).",
    )
    parser.add_argument(
        "--task-condition-scale",
        type=float,
        default=None,
        help="Optional inference override; 0 reproduces checkpoints trained before direct task conditioning.",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one scanpath figure per predicted image/finding pair.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Limit evaluation batches; 0 evaluates all images.",
    )
    return parser


def duration_diagnostics(predictions, ground_truth):
    predicted = np.asarray(
        [value for path in predictions for value in path.get("T", [])[1:]], dtype=float
    )
    target = np.asarray(
        [value for path in ground_truth for value in path.get("T", [])[1:]], dtype=float
    )
    references = defaultdict(list)
    for path in ground_truth:
        references[path["name"], path["task"], path.get("condition", "present")].append(
            path
        )
    absolute_errors = []
    for path in predictions:
        key = (path["name"], path["task"], path.get("condition", "present"))
        pred_values = np.asarray(path.get("T", [])[1:], dtype=float)
        for reference in references.get(key, []):
            ref_values = np.asarray(reference.get("T", [])[1:], dtype=float)
            length = min(len(pred_values), len(ref_values))
            if length:
                absolute_errors.extend(
                    np.abs(pred_values[:length] - ref_values[:length])
                )

    def stats(values):
        if not len(values):
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "median": None,
                "min": None,
                "max": None,
            }
        return {
            "count": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    predicted_stats, target_stats = (stats(predicted), stats(target))
    target_std = target_stats["std"] or 0.0
    collapse_ratio = (
        (predicted_stats["std"] or 0.0) / target_std if target_std > 0 else None
    )
    return {
        "note": "The synthetic 0.3s center fixation is excluded.",
        "predicted": predicted_stats,
        "target": target_stats,
        "matched_step_mae_seconds": float(np.mean(absolute_errors))
        if absolute_errors
        else None,
        "predicted_to_target_std_ratio": collapse_ratio,
        "constant_prediction_warning": bool(
            collapse_ratio is not None and collapse_ratio < 0.2
        ),
    }


def conditioning_diagnostics(predictions):
    by_image = defaultdict(list)
    for path in predictions:
        by_image[path["name"]].append(path)
    pairs = []
    for image_name, paths in by_image.items():
        for first_index in range(len(paths)):
            for second_index in range(first_index + 1, len(paths)):
                first, second = (paths[first_index], paths[second_index])
                if first["task"] == second["task"]:
                    continue
                first_xy = np.column_stack((first["X"], first["Y"])).astype(float)
                second_xy = np.column_stack((second["X"], second["Y"])).astype(float)
                length = min(len(first_xy), len(second_xy))
                distances = np.linalg.norm(
                    first_xy[:length] - second_xy[:length], axis=1
                )
                exactly_identical = len(first_xy) == len(second_xy) and np.array_equal(
                    first_xy, second_xy
                )
                pairs.append(
                    {
                        "name": image_name,
                        "task_a": first["task"],
                        "task_b": second["task"],
                        "exactly_identical": bool(exactly_identical),
                        "mean_position_difference_px": float(distances.mean())
                        if length
                        else None,
                    }
                )
    identical = sum((pair["exactly_identical"] for pair in pairs))
    return {
        "different_finding_pairs_on_same_image": len(pairs),
        "exactly_identical_pairs": identical,
        "identical_pair_fraction": identical / len(pairs) if pairs else None,
        "pairs": pairs,
    }


@torch.no_grad()
def task_query_diagnostics(model, dataset, device, max_pairs):
    if max_pairs <= 0:
        return {"evaluated_pairs": 0, "pairs": []}
    indices_by_image = defaultdict(list)
    for index, record in enumerate(dataset.items):
        indices_by_image[record["name"]].append(index)
    selected = []
    for indices in indices_by_image.values():
        for first_position in range(len(indices)):
            for second_position in range(first_position + 1, len(indices)):
                first_index, second_index = (
                    indices[first_position],
                    indices[second_position],
                )
                if (
                    dataset.items[first_index]["task"]
                    != dataset.items[second_index]["task"]
                ):
                    selected.append((first_index, second_index))
                    if len(selected) >= max_pairs:
                        break
            if len(selected) >= max_pairs:
                break
        if len(selected) >= max_pairs:
            break
    rows = []
    for first_index, second_index in selected:
        first, second = (dataset[first_index], dataset[second_index])
        images = torch.stack((first["image"], second["image"])).to(device)
        task_ids = torch.stack((first["task_id"], second["task_id"])).to(device)
        history = torch.zeros(2, model.max_history, 2, device=device)
        history[:, 0] = 0.5
        padding = torch.ones(2, model.max_history, dtype=torch.bool, device=device)
        padding[:, 0] = False
        outputs = model(
            images,
            task_ids,
            history,
            padding,
            torch.stack(
                (first["global_text_feature"], second["global_text_feature"])
            ).to(device),
            torch.stack(
                (first["local_text_features"], second["local_text_features"])
            ).to(device),
        )
        query = outputs["task_query"]
        raw_query = outputs["raw_task_query"]
        logits = outputs["coordinate_logits"].flatten(1)
        query_difference = (query[0] - query[1]).abs()
        logit_difference = (logits[0] - logits[1]).abs()
        rows.append(
            {
                "name": first["image_name"],
                "task_a": first["task"],
                "task_b": second["task"],
                "query_mean_abs_difference": float(query_difference.mean()),
                "query_max_abs_difference": float(query_difference.max()),
                "query_cosine_similarity": float(
                    F.cosine_similarity(query[0:1], query[1:2])
                ),
                "query_numerically_equal": bool(
                    torch.allclose(query[0], query[1], atol=1e-06, rtol=1e-05)
                ),
                "raw_query_mean_abs_difference": float(
                    (raw_query[0] - raw_query[1]).abs().mean()
                ),
                "raw_query_cosine_similarity": float(
                    F.cosine_similarity(raw_query[0:1], raw_query[1:2])
                ),
                "heatmap_mean_abs_difference": float(logit_difference.mean()),
                "first_step_same_argmax": bool(
                    logits[0].argmax() == logits[1].argmax()
                ),
            }
        )
    return {
        "evaluated_pairs": len(rows),
        "numerically_equal_query_pairs": sum(
            (row["query_numerically_equal"] for row in rows)
        ),
        "same_first_step_argmax_pairs": sum(
            (row["first_step_same_argmax"] for row in rows)
        ),
        "pairs": rows,
    }


def _latest_checkpoint():
    candidates = list(OUTPUT_ROOT.glob("experiment_*/checkpoint_best.pth"))
    if not candidates:
        candidates = list(OUTPUT_ROOT.glob("experiment_*/checkpoint_last.pth"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found under {OUTPUT_ROOT}. Run train.py first or pass --checkpoint."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


@torch.no_grad()
def generate_predictions(
    model,
    data_loader,
    device,
    sample,
    max_batches,
    edge_margin=2,
    return_attention_maps=False,
):
    predictions = []
    attention_maps = []
    for batch_index, batch in enumerate(tqdm(data_loader, desc="inference")):
        if max_batches and batch_index >= max_batches:
            break
        generated = model.generate(
            batch["image"].to(device),
            batch["task_id"].to(device),
            batch["global_text_feature"].to(device),
            batch["local_text_features"].to(device),
            sample=sample,
            edge_margin=edge_margin,
        )
        for index, scanpath in enumerate(generated):
            coordinates = scanpath["coordinates"].numpy()
            if return_attention_maps:
                attention_maps.append(scanpath["attention_map"].numpy())
            predictions.append(
                {
                    "name": batch["image_name"][index],
                    "task": batch["task"][index],
                    "condition": batch["condition"][index],
                    "X": coordinates[:, 0].tolist(),
                    "Y": coordinates[:, 1].tolist(),
                    "T": scanpath["durations"].numpy().tolist(),
                }
            )
    if not predictions:
        raise RuntimeError("Inference produced no predictions")
    if return_attention_maps:
        return (predictions, attention_maps)
    return predictions


def _attention_map_filename(index, prediction):
    safe_task = re.sub("[^a-zA-Z0-9_-]+", "_", prediction["task"])
    return f"{index:04d}_{Path(prediction['name']).stem}_{safe_task}.png"


def _write_attention_map(values, output_path):
    values = np.asarray(values, dtype=np.float32).squeeze()
    if values.ndim != 2 or values.shape != (224, 224):
        raise ValueError(f"Expected attention map [224,224], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Attention map contains NaN or infinity")
    pixels = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="L").save(output_path)


def save_attention_maps(
    predictions, predicted_attention_maps, dataset, output_dir, suffix
):
    if len(predictions) != len(predicted_attention_maps):
        raise ValueError("Prediction and attention-map counts must agree")
    predicted_dir = output_dir / f"pred_attentionmap{suffix}"
    ground_truth_dir = output_dir / f"gt_attentionmap{suffix}"
    for index, (prediction, predicted_map) in enumerate(
        zip(predictions, predicted_attention_maps)
    ):
        filename = _attention_map_filename(index, prediction)
        _write_attention_map(predicted_map, predicted_dir / filename)
        gt_values = dataset._load_attention_map(prediction["name"], prediction["task"])[
            "attention_map"
        ].numpy()
        _write_attention_map(gt_values, ground_truth_dir / filename)


def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    checkpoint_path = (args.checkpoint or _latest_checkpoint()).expanduser().resolve()
    output_dir = (args.output_dir or checkpoint_path.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = GazeSearchImageDataset(
        args.data_root, split=args.split, text_model=args.text_model
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = checkpoint.get("args", {})
    model = GazeSearchGlobalLocalModel(
        weights=False,
        freeze_backbone=not model_args.get("train_backbone", True),
        dropout=float(model_args.get("dropout", 0.0)),
        attention_lambda1=float(model_args.get("attention_lambda1", 1.0)),
        attention_lambda2=float(model_args.get("attention_lambda2", -1.0)),
        history_gaussian_sigma=float(model_args.get("history_gaussian_sigma", 0.75)),
        task_condition_scale=args.task_condition_scale
        if args.task_condition_scale is not None
        else float(model_args.get("task_condition_scale", 0.0)),
    ).to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    print(f"Checkpoint: {checkpoint_path}")
    predictions, predicted_attention_maps = generate_predictions(
        model,
        data_loader,
        device,
        args.sample,
        args.max_batches,
        args.edge_margin,
        return_attention_maps=True,
    )
    selected_keys = {
        (path["name"], path["task"], path["condition"]) for path in predictions
    }
    ground_truth = [
        path
        for path in ground_truth_scanpaths(dataset)
        if (path["name"], path["task"], path["condition"]) in selected_keys
    ]
    metrics, per_sample = evaluate_scanpaths(predictions, ground_truth)
    metric_std = metric_standard_deviations(per_sample)
    print("\nGazeSearch scanpath metrics")
    for name in METRIC_NAMES:
        print(f"{name:28s}: {metrics[name]:.6f}, std: {metric_std[name]:.6f}")
    suffix = "" if args.split == "test" else f"_{args.split}"
    save_attention_maps(
        predictions, predicted_attention_maps, dataset, output_dir, suffix
    )
    metrics_output = {
        **metrics,
        "standard_deviation": metric_std,
        "standard_deviation_ddof": 0,
        "standard_deviation_scope": "across evaluated per-sample rows",
        "evaluated_sample_count": len(per_sample),
    }
    write_json(output_dir / f"metrics{suffix}.json", metrics_output)
    write_json(output_dir / f"metrics_per_sample{suffix}.json", per_sample)
    write_json(output_dir / f"ground_truth{suffix}.json", ground_truth)
    write_json(output_dir / f"predictions{suffix}.json", predictions)
    write_json(
        output_dir / f"duration_diagnostics{suffix}.json",
        duration_diagnostics(predictions, ground_truth),
    )
    write_json(
        output_dir / f"conditioning_diagnostics{suffix}.json",
        conditioning_diagnostics(predictions),
    )
    if args.diagnostic_pairs > 0:
        write_json(
            output_dir / f"task_query_diagnostics{suffix}.json",
            task_query_diagnostics(model, dataset, device, args.diagnostic_pairs),
        )
    if args.visualize:
        gt_by_key = defaultdict(list)
        for path in ground_truth:
            gt_by_key[path["name"], path["task"], path["condition"]].append(path)
        visualization_dir = output_dir / f"visualization{suffix}"
        ground_truth_dir = output_dir / f"ground_truth{suffix}"
        prediction_dir = output_dir / f"prediction{suffix}"
        for index, prediction in enumerate(predictions):
            key = (prediction["name"], prediction["task"], prediction["condition"])
            reference = gt_by_key[key][0]
            filename = _attention_map_filename(index, prediction)
            save_scanpath_visualization(
                dataset.image_root / prediction["name"],
                reference,
                prediction,
                visualization_dir / filename,
                ground_truth_output_path=ground_truth_dir / filename,
                prediction_output_path=prediction_dir / filename,
            )
            print(index)
    print(f"Evaluation output: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main(get_args_parser().parse_args())
