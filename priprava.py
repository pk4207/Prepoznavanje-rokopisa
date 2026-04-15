import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow_datasets as tfds

absl_logging: Any | None = None
try:
    from absl import logging as absl_logging
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR / "project_data"
PREPARED_DIR = PROJECT_DIR / "prepared"
CACHE_DIR = PROJECT_DIR / "cache"
DATA_FILE = PREPARED_DIR / "emnist_balanced_data.npz"
INFO_FILE = PREPARED_DIR / "dataset_info.json"
LABELS_FILE = PREPARED_DIR / "emnist_labels.txt"
TFDS_DIR = CACHE_DIR / "tfds"
TFDS_EMNIST_DIR = TFDS_DIR / "emnist"
TFDS_BALANCED_DIR = TFDS_EMNIST_DIR / "balanced"

DATASET_NAME = "emnist/balanced"
REQUIRED_INFO_KEYS = {
    "dataset_name",
    "train_examples",
    "test_examples",
    "num_classes",
    "image_shape",
    "label_names",
}
EMNIST_BALANCED_LABELS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C/c", "D", "E", "F", "G", "H", "I/i", "J/j",
    "K/k", "L/l", "M/m", "N", "O/o", "P/p", "Q", "R", "S/s", "T",
    "U/u", "V/v", "W/w", "X/x", "Y/y", "Z/z",
    "a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t",
]
MERGED_LETTERS = [
    "C/c", "I/i", "J/j", "K/k", "L/l", "M/m", "O/o", "P/p",
    "S/s", "U/u", "V/v", "W/w", "X/x", "Y/y", "Z/z",
]


def fix_emnist_images(images):
    images = np.transpose(images, (0, 2, 1, 3))
    images = images.astype("float32") / 255.0
    return images


def validate_data(x_train, y_train, x_test, y_test, num_classes):
    if x_train.ndim != 4 or x_test.ndim != 4:
        raise ValueError("Slike morajo imeti 4 dimenzije: (N, 28, 28, 1).")
    if x_train.shape[1:] != (28, 28, 1) or x_test.shape[1:] != (28, 28, 1):
        raise ValueError("Napacna oblika slik.")
    if len(np.unique(y_train)) != num_classes:
        raise ValueError("Stevilo razredov se ne ujema.")
    if y_train.min() != 0 or y_test.min() != 0:
        raise ValueError("Oznake morajo zaceti pri 0.")


def prepared_files_exist():
    return DATA_FILE.exists() and INFO_FILE.exists() and LABELS_FILE.exists()


def load_existing_info():
    if not INFO_FILE.exists():
        raise SystemExit("Manjka dataset_info.json.")
    return json.loads(INFO_FILE.read_text(encoding="utf-8"))


def prepared_files_are_valid():
    try:
        info = load_existing_info()
        if not REQUIRED_INFO_KEYS.issubset(info):
            return False
        if info["dataset_name"] != DATASET_NAME:
            return False

        labels = LABELS_FILE.read_text(encoding="utf-8").splitlines()
        if len(labels) != info["num_classes"]:
            return False

        with np.load(DATA_FILE, allow_pickle=False) as data:
            x_train, y_train = data["x_train"], data["y_train"]
            x_test, y_test = data["x_test"], data["y_test"]

        validate_data(x_train, y_train, x_test, y_test, info["num_classes"])
        if len(x_train) != info["train_examples"] or len(x_test) != info["test_examples"]:
            return False

        return True
    except Exception:
        return False


def load_emnist():
    error = None
    try:
        return tfds.load(
            DATASET_NAME,
            split=["train", "test"],
            as_supervised=True,
            batch_size=-1,
            with_info=True,
            data_dir=str(TFDS_DIR),
            download=True,
        )
    except Exception as first_error:
        error = first_error
        if find_incomplete_tfds_cache() is not None:
            print("Najden nedokoncan TFDS cache.")
            shutil.rmtree(TFDS_EMNIST_DIR, ignore_errors=True)
            try:
                return tfds.load(
                    DATASET_NAME,
                    split=["train", "test"],
                    as_supervised=True,
                    batch_size=-1,
                    with_info=True,
                    data_dir=str(TFDS_DIR),
                    download=True,
                )
            except Exception as retry_error:
                error = retry_error
        raise SystemExit(
            f"Napaka pri nalaganju EMNIST Balanced.\nPodrobnosti: {error}"
        ) from error


def find_incomplete_tfds_cache():
    if not TFDS_BALANCED_DIR.exists():
        return None
    for version_dir in TFDS_BALANCED_DIR.iterdir():
        if version_dir.is_dir() and not (version_dir / "dataset_info.json").exists():
            return version_dir
    return None


def main():
    PROJECT_DIR.mkdir(exist_ok=True)
    PREPARED_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    if absl_logging is not None:
        absl_logging.set_verbosity(absl_logging.ERROR)
        absl_logging.set_stderrthreshold(absl_logging.ERROR)

    if prepared_files_exist() and prepared_files_are_valid():
        info = load_existing_info()
        print("EMNIST Balanced je ze pripravljen.")
        print(f"Razredi: {info['num_classes']}, Train: {info['train_examples']}, Test: {info['test_examples']}")
        return

    if prepared_files_exist():
        print("Obstojece datoteke niso veljavne.")

    (train_data, test_data), info = load_emnist()

    x_train, y_train = tfds.as_numpy(train_data)
    x_test, y_test = tfds.as_numpy(test_data)

    x_train = fix_emnist_images(x_train)
    x_test = fix_emnist_images(x_test)
    y_train = y_train.astype("int64")
    y_test = y_test.astype("int64")

    label_names = list(EMNIST_BALANCED_LABELS)
    validate_data(x_train, y_train, x_test, y_test, len(label_names))

    dataset_info = {
        "dataset_name": DATASET_NAME,
        "train_examples": int(len(x_train)),
        "test_examples": int(len(x_test)),
        "num_classes": len(label_names),
        "image_shape": [28, 28, 1],
        "merged_letters": MERGED_LETTERS,
        "label_names": label_names,
    }

    np.savez(DATA_FILE, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    INFO_FILE.write_text(json.dumps(dataset_info, indent=2, ensure_ascii=False), encoding="utf-8")
    LABELS_FILE.write_text("\n".join(label_names), encoding="utf-8")

    print(f"Podatki pripravljeni: {DATA_FILE}")
    print(f"Razredi: {len(label_names)}, Train: {len(x_train)}, Test: {len(x_test)}")


if __name__ == "__main__":
    main()
