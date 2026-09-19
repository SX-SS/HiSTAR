import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "Data"
DEFAULT_ANNOTATIONS = "finding_visual_search_coco_format_train_test_filtered_max_6_split_train_valid_test_2024-07-22_shuffled.json"
IMAGE_SIZE = 224
MAX_HISTORY = 6
TASK_NAMES = (
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "enlarged cardiomediastinum",
    "fracture",
    "lung lesion",
    "lung opacity",
    "pleural effusion",
    "pleural other",
    "pneumonia",
    "pneumothorax",
    "support devices",
)
TASK_TO_ID = {name: index for index, name in enumerate(TASK_NAMES)}


def _image_transform():
    return v2.Compose(
        [
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _load_records(data_root, split, annotations):
    annotation_path = data_root / annotations
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"GazeSearch annotation file not found: {annotation_path}"
        )
    with annotation_path.open("r", encoding="utf-8") as stream:
        records = [row for row in json.load(stream) if row["split"] == split]
    if not records:
        raise ValueError(f"No records found for split={split!r} in {annotation_path}")
    unknown = sorted({row["task"] for row in records} - set(TASK_NAMES))
    if unknown:
        raise ValueError(f"Unknown finding names: {unknown}")
    return records


def _scanpath_tensors(record):
    coordinates = torch.tensor(
        np.column_stack((record["X"], record["Y"])), dtype=torch.float32
    )
    durations = torch.tensor(record["T"], dtype=torch.float32)
    if len(coordinates) != len(durations) or len(coordinates) == 0:
        raise ValueError(
            f"Invalid scanpath in {record['name']}: X/Y/T lengths disagree"
        )
    if len(coordinates) > MAX_HISTORY:
        coordinates = coordinates[:MAX_HISTORY]
        durations = durations[:MAX_HISTORY]
    coordinates[0] = IMAGE_SIZE / 2
    return (coordinates, durations)


class GazeSearchDataset(Dataset):
    def __init__(
        self,
        data_root=DEFAULT_DATA_ROOT,
        split="train",
        annotations=DEFAULT_ANNOTATIONS,
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        self.image_root = self.data_root / "images"
        self.records = _load_records(self.data_root, split, annotations)
        self.transform = _image_transform()
        self.examples = []
        for record_index, record in enumerate(self.records):
            length = min(len(record["X"]), MAX_HISTORY)
            self.examples.extend(
                ((record_index, target_index) for target_index in range(1, length))
            )
            self.examples.append((record_index, length))

    def __len__(self):
        return len(self.examples)

    def _read_image(self, name):
        path = self.image_root / name
        if not path.is_file():
            raise FileNotFoundError(f"GazeSearch image not found: {path}")
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))

    def __getitem__(self, index):
        record_index, target_index = self.examples[index]
        record = self.records[record_index]
        coordinates, durations = _scanpath_tensors(record)
        sequence_length = len(coordinates)
        is_terminal = target_index == sequence_length
        history = torch.zeros(MAX_HISTORY, 2, dtype=torch.float32)
        history[:target_index] = coordinates[:target_index] / IMAGE_SIZE
        padding_mask = torch.ones(MAX_HISTORY, dtype=torch.bool)
        padding_mask[:target_index] = False
        if is_terminal:
            target_coordinate = coordinates[-1] / IMAGE_SIZE
            target_duration = torch.tensor(0.0)
        else:
            target_coordinate = coordinates[target_index] / IMAGE_SIZE
            target_duration = durations[target_index]
        return {
            "image": self._read_image(record["name"]),
            "task_id": torch.tensor(TASK_TO_ID[record["task"]], dtype=torch.long),
            "task": record["task"],
            "history": history,
            "padding_mask": padding_mask,
            "history_length": torch.tensor(target_index, dtype=torch.long),
            "previous_coordinate": (coordinates[target_index - 1] / IMAGE_SIZE).clamp(
                0.0, 1.0
            ),
            "target_coordinate": target_coordinate.clamp(0.0, 1.0),
            "target_duration": target_duration.float(),
            "termination": torch.tensor(float(is_terminal)),
            "image_name": record["name"],
            "subject": int(record["subject"]),
        }


class GazeSearchImageDataset(Dataset):
    def __init__(
        self, data_root=DEFAULT_DATA_ROOT, split="test", annotations=DEFAULT_ANNOTATIONS
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        self.image_root = self.data_root / "images"
        self.records = _load_records(self.data_root, split, annotations)
        self.transform = _image_transform()
        unique = {}
        for record in self.records:
            key = (record["name"], record["task"], record["condition"])
            unique.setdefault(key, record)
        self.items = list(unique.values())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        record = self.items[index]
        path = self.image_root / record["name"]
        with Image.open(path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return {
            "image": image_tensor,
            "task_id": torch.tensor(TASK_TO_ID[record["task"]], dtype=torch.long),
            "task": record["task"],
            "condition": record["condition"],
            "image_name": record["name"],
            "image_path": str(path),
        }


def ground_truth_scanpaths(dataset):
    return [
        {
            "name": row["name"],
            "task": row["task"],
            "condition": row["condition"],
            "X": list(map(float, row["X"])),
            "Y": list(map(float, row["Y"])),
            "T": list(map(float, row["T"])),
        }
        for row in dataset.records
    ]
