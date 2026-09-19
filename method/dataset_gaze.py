import json
from functools import lru_cache
from pathlib import Path
import torch
from Data.dataset import GazeSearchDataset as BaseGazeSearchDataset
from Data.dataset import GazeSearchImageDataset as BaseGazeSearchImageDataset
from Data.dataset import IMAGE_SIZE

TEXT_DIM = 128
NUM_REGIONS = 49
DEFAULT_TEXT_MODEL = "microsoft/BiomedVLP-CXR-BERT-specialized"
_ENCODERS = {}
_FEATURES = {}


def _global_text(value):
    findings = " ".join(
        (
            f"Location: {item.get('location', 'uncertain')}. Boundary: {item.get('boundary', 'uncertain')}. Characteristics: {item.get('characteristics', 'uncertain')}. Remarks: {item.get('remarks', '')}."
            for item in value.get("findings", [])
        )
    )
    return f"Anatomical overview: {value['anatomical_overview']}. {findings} Global summary: {value['global_summary']}."


def _local_text(value):
    return f"Anatomy: {value['anatomy']}. Location: {value['location']}. Abnormality: {value['abnormality']}. Boundary: {value['boundary']}. Characteristics: {value['characteristics']}. Summary: {value['summary']}."


def _text_encoder(model_name):
    if model_name not in _ENCODERS:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Install transformers to encode the clinical descriptions"
            ) from error
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval()
        model.requires_grad_(False)
        _ENCODERS[model_name] = (tokenizer, model)
    return _ENCODERS[model_name]


def _encode_descriptions(global_value, local_values, model_name):
    tokenizer, model = _text_encoder(model_name)
    texts = [_global_text(global_value)] + [
        _local_text(value) for value in local_values
    ]
    tokens = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        features = model.get_projected_text_embeddings(
            input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
        ).float()
    return (features[0], features[1:])


class _GazeMixin:
    def _initialize_gaze_data(self, text_model):
        self.text_model = text_model
        self._attention_points = {}
        for record in self.records:
            key = (record["name"], record["task"])
            points = list(zip(record["X"], record["Y"]))
            if points:
                points[0] = (IMAGE_SIZE / 2, IMAGE_SIZE / 2)
            self._attention_points.setdefault(key, set()).update(points)

    def _load_text_features(self, image_name):
        sample_dir = self.data_root / "Text" / Path(image_name).stem
        key = (str(sample_dir), self.text_model)
        if key not in _FEATURES:
            global_path = sample_dir / "global.json"
            local_path = sample_dir / "local.json"
            if not global_path.is_file() or not local_path.is_file():
                raise FileNotFoundError(
                    f"Missing global/local descriptions in {sample_dir}"
                )
            global_value = json.loads(global_path.read_text(encoding="utf-8"))
            local_values = json.loads(local_path.read_text(encoding="utf-8"))
            if len(local_values) != NUM_REGIONS:
                raise ValueError(
                    f"Expected {NUM_REGIONS} local regions in {local_path}"
                )
            global_feature, local_features = _encode_descriptions(
                global_value, local_values, self.text_model
            )
            if tuple(global_feature.shape) != (TEXT_DIM,):
                raise ValueError(
                    f"Expected global feature [{TEXT_DIM}] in {global_path}"
                )
            if tuple(local_features.shape) != (NUM_REGIONS, TEXT_DIM):
                raise ValueError(
                    f"Expected local features [{NUM_REGIONS},{TEXT_DIM}] in {local_path}"
                )
            _FEATURES[key] = (global_feature, local_features)
        global_feature, local_features = _FEATURES[key]
        return {
            "global_text_feature": global_feature,
            "local_text_features": local_features,
        }

    @lru_cache(maxsize=256)
    def _load_attention_map(self, image_name, task):
        points = self._attention_points[image_name, task]
        fixation_map = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
        for x_coordinate, y_coordinate in points:
            x_index = min(max(round(float(x_coordinate)), 0), IMAGE_SIZE - 1)
            y_index = min(max(round(float(y_coordinate)), 0), IMAGE_SIZE - 1)
            fixation_map[y_index, x_index] = 1
        sigma = 16.0
        radius = round(4 * sigma)
        axis = torch.arange(-radius, radius + 1)
        kernel = torch.exp(
            -(axis[:, None] ** 2 + axis[None, :] ** 2) / (2 * sigma**2)
        )
        values = torch.zeros_like(fixation_map)
        for y_index, x_index in fixation_map.nonzero():
            y_index = int(y_index)
            x_index = int(x_index)
            y_start = max(0, y_index - radius)
            y_end = min(IMAGE_SIZE, y_index + radius + 1)
            x_start = max(0, x_index - radius)
            x_end = min(IMAGE_SIZE, x_index + radius + 1)
            kernel_y = y_start - (y_index - radius)
            kernel_x = x_start - (x_index - radius)
            values[y_start:y_end, x_start:x_end] += kernel[
                kernel_y : kernel_y + y_end - y_start,
                kernel_x : kernel_x + x_end - x_start,
            ]
        maximum = values.max()
        if maximum > 0:
            values /= maximum
        return {"attention_map": values.unsqueeze(0)}


class GazeSearchDataset(_GazeMixin, BaseGazeSearchDataset):
    def __init__(self, *args, text_model=DEFAULT_TEXT_MODEL, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_gaze_data(text_model)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample.update(self._load_text_features(sample["image_name"]))
        sample.update(self._load_attention_map(sample["image_name"], sample["task"]))
        return sample


class GazeSearchImageDataset(_GazeMixin, BaseGazeSearchImageDataset):
    def __init__(self, *args, text_model=DEFAULT_TEXT_MODEL, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_gaze_data(text_model)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample.update(self._load_text_features(sample["image_name"]))
        return sample
