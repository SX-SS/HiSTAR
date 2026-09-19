import json
import random
from functools import lru_cache
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

CENTER_POINT = (112.0, 112.0)
CENTER_DURATION_SECONDS = 0.3
FIXATION_FILL_COLOR = "#EFD5D5"
FIXATION_EDGE_COLOR = "#BD8585"
SACCADE_COLOR = "#C36969"
FIXATION_FILL_ALPHA = 0.8
MIN_FIXATION_RADIUS = 6.0
MAX_FIXATION_RADIUS = 18.0
DURATION_REFERENCE_SECONDS = 1.0
MODEL_IMAGE_SIZE = 224
STANDALONE_IMAGE_SIZE = 896
COMPOSITE_IMAGE_SIZE = 512
SCANPATH_SCALE = STANDALONE_IMAGE_SIZE / MODEL_IMAGE_SIZE
SCANPATH_LINE_WIDTH = 8
FIXATION_OUTLINE_WIDTH = 8
FIXATION_FONT_SIZE = 48


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def save_checkpoint(path, model, optimizer, epoch, args, **metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": args,
            **metadata,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def write_json(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(values, stream, indent=2, ensure_ascii=False)


def _prepare_scanpath(scanpath):
    x = np.asarray(scanpath["X"], dtype=float)
    y = np.asarray(scanpath["Y"], dtype=float)
    durations = np.asarray(scanpath.get("T", np.ones(len(x)) * 0.2), dtype=float)
    if len(x) != len(y) or len(x) != len(durations):
        raise ValueError("Scanpath X/Y/T lengths must agree for visualization")
    if not len(x) or not np.allclose((x[0], y[0]), CENTER_POINT):
        x = np.insert(x, 0, CENTER_POINT[0])
        y = np.insert(y, 0, CENTER_POINT[1])
        durations = np.insert(durations, 0, CENTER_DURATION_SECONDS)
    else:
        durations = durations.copy()
        durations[0] = CENTER_DURATION_SECONDS
    prepared = dict(scanpath)
    prepared.update({"X": x, "Y": y, "T": durations})
    return prepared


def _fixation_radii(durations):
    safe_durations = np.nan_to_num(
        np.asarray(durations, dtype=float),
        nan=0.0,
        posinf=DURATION_REFERENCE_SECONDS,
        neginf=0.0,
    )
    normalized = (
        np.clip(safe_durations, 0.0, DURATION_REFERENCE_SECONDS)
        / DURATION_REFERENCE_SECONDS
    )
    return MIN_FIXATION_RADIUS + (MAX_FIXATION_RADIUS - MIN_FIXATION_RADIUS) * np.sqrt(
        normalized
    )


def _rgba(hex_color, alpha=255):
    value = hex_color.lstrip("#")
    return tuple((int(value[index : index + 2], 16) for index in (0, 2, 4))) + (alpha,)


@lru_cache(maxsize=1)
def _scanpath_font():
    try:
        return ImageFont.truetype(
            font_manager.findfont("DejaVu Sans"), FIXATION_FONT_SIZE
        )
    except (OSError, ValueError):
        return ImageFont.load_default()


def _render_scanpath(image, scanpath):
    x = np.asarray(scanpath["X"], dtype=float)
    y = np.asarray(scanpath["Y"], dtype=float)
    radii = _fixation_radii(np.asarray(scanpath["T"], dtype=float))
    base = image.convert("RGBA")
    canvas = base.copy()
    line_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    line_draw = ImageDraw.Draw(line_layer)
    for start_index in range(len(x) - 1):
        start = np.array((x[start_index], y[start_index]), dtype=float)
        end = np.array((x[start_index + 1], y[start_index + 1]), dtype=float)
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if distance <= radii[start_index] + radii[start_index + 1] or distance == 0.0:
            continue
        direction = delta / distance
        clipped_start = start + direction * radii[start_index]
        clipped_end = end - direction * radii[start_index + 1]
        line_draw.line(
            tuple((clipped_start * SCANPATH_SCALE).tolist())
            + tuple((clipped_end * SCANPATH_SCALE).tolist()),
            fill=_rgba(SACCADE_COLOR),
            width=SCANPATH_LINE_WIDTH,
        )
    canvas = Image.alpha_composite(canvas, line_layer)
    font = _scanpath_font()
    for order, (point_x, point_y, radius) in enumerate(zip(x, y, radii), start=1):
        center_x = float(point_x * SCANPATH_SCALE)
        center_y = float(point_y * SCANPATH_SCALE)
        scaled_radius = float(radius * SCANPATH_SCALE)
        bounds = (
            center_x - scaled_radius,
            center_y - scaled_radius,
            center_x + scaled_radius,
            center_y + scaled_radius,
        )
        mask = Image.new("L", base.size, 0)
        ImageDraw.Draw(mask).ellipse(bounds, fill=255)
        canvas.paste(base, (0, 0), mask)
        marker_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        marker_draw = ImageDraw.Draw(marker_layer)
        marker_draw.ellipse(
            bounds,
            fill=_rgba(FIXATION_FILL_COLOR, round(255 * FIXATION_FILL_ALPHA)),
            outline=_rgba(FIXATION_EDGE_COLOR),
            width=FIXATION_OUTLINE_WIDTH,
        )
        text = str(order)
        text_bounds = marker_draw.textbbox((0, 0), text, font=font)
        text_width = text_bounds[2] - text_bounds[0]
        text_height = text_bounds[3] - text_bounds[1]
        marker_draw.text(
            (center_x - text_width / 2, center_y - text_height / 2 - text_bounds[1]),
            text,
            font=font,
            fill=_rgba("#333333"),
        )
        canvas = Image.alpha_composite(canvas, marker_layer)
    return canvas.convert("RGB")


def _setup_axis(axis, image, title=None):
    axis.imshow(
        image,
        cmap="gray",
        extent=(0, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 0),
        interpolation="antialiased",
    )
    if title:
        axis.set_title(title)
    axis.set_xlim(0, MODEL_IMAGE_SIZE)
    axis.set_ylim(MODEL_IMAGE_SIZE, 0)
    axis.set_aspect("equal")
    axis.axis("off")


def _save_rendered_scanpath(image, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


@lru_cache(maxsize=8)
def _load_visualization_images(image_path):
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        standalone = source.resize(
            (STANDALONE_IMAGE_SIZE, STANDALONE_IMAGE_SIZE), Image.Resampling.LANCZOS
        )
        composite = source.resize(
            (COMPOSITE_IMAGE_SIZE, COMPOSITE_IMAGE_SIZE), Image.Resampling.LANCZOS
        )
    return (standalone, composite)


def save_scanpath_visualization(
    image_path,
    ground_truth,
    prediction,
    output_path,
    *,
    ground_truth_output_path=None,
    prediction_output_path=None,
):
    standalone_source, composite_source = _load_visualization_images(
        str(Path(image_path).resolve())
    )
    ground_truth = _prepare_scanpath(ground_truth)
    prediction = _prepare_scanpath(prediction)
    rendered_ground_truth = _render_scanpath(standalone_source, ground_truth)
    rendered_prediction = _render_scanpath(standalone_source, prediction)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
    for axis, panel, title in zip(
        axes,
        (composite_source, rendered_ground_truth, rendered_prediction),
        ("Image", "Ground Truth", "Prediction"),
    ):
        _setup_axis(axis, panel, title)
    figure.suptitle(f"{prediction['task']} | {prediction['name']}", fontsize=10)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    if ground_truth_output_path is not None:
        _save_rendered_scanpath(rendered_ground_truth, ground_truth_output_path)
    if prediction_output_path is not None:
        _save_rendered_scanpath(rendered_prediction, prediction_output_path)
