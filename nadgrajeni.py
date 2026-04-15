import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
from keras import layers

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR / "project_data"
PREPARED_DIR = PROJECT_DIR / "prepared"
RESULTS_DIR = BASE_DIR / "rezultati"
DATA_FILE = PREPARED_DIR / "emnist_balanced_data.npz"
INFO_FILE = PREPARED_DIR / "dataset_info.json"

MODEL_NAME = "nadgrajeni"
BATCH_SIZE = 128
EPOCHS = 40
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42
INITIAL_LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EMA_MOMENTUM = 0.99
TRANSLATION_FACTOR = 0.08


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", help="Mapa trenutnega runa")
    return parser.parse_args()


def resolve_run_dir(run_dir_arg):
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
    else:
        timestamp = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        run_dir = RESULTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_data():
    if not DATA_FILE.exists():
        raise SystemExit("Najprej zazeni priprava.py")
    with np.load(DATA_FILE, allow_pickle=False) as data:
        return data["x_train"], data["y_train"], data["x_test"], data["y_test"]


def load_info():
    if not INFO_FILE.exists():
        raise SystemExit("Najprej zazeni priprava.py")
    return json.loads(INFO_FILE.read_text(encoding="utf-8"))


def get_output_paths(run_dir):
    model_dir = run_dir / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    return {
        "model_dir": model_dir,
        "model_file": model_dir / "model.keras",
        "results_file": model_dir / "results.txt",
        "summary_file": model_dir / "model_summary.txt",
    }


def write_mapping_file(run_dir, dataset_info):
    labels = list(dataset_info["label_names"])
    mapping = {
        "dataset_name": dataset_info["dataset_name"],
        "num_classes": dataset_info["num_classes"],
        "image_shape": dataset_info["image_shape"],
        "merged_letters": dataset_info.get("merged_letters", []),
        "labels": labels,
        "class_mapping": [
            {"index": i, "label": label} for i, label in enumerate(labels)
        ],
    }
    (run_dir / "mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def build_optimizer(steps_per_epoch):
    learning_rate = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        decay_steps=steps_per_epoch * EPOCHS,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        use_ema=True,
        ema_momentum=EMA_MOMENTUM,
    )
    return optimizer


def build_model(num_classes, steps_per_epoch):
    optimizer = build_optimizer(steps_per_epoch)

    augmentation = keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
        layers.RandomTranslation(
            height_factor=TRANSLATION_FACTOR,
            width_factor=TRANSLATION_FACTOR,
        ),
    ], name="augmentation")

    inputs = keras.Input(shape=(28, 28, 1))

    x = augmentation(inputs)

    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Flatten()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=MODEL_NAME)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def estimate_inference_speed(model, x_test):
    sample = x_test[:min(1000, len(x_test))]
    model.predict(sample[:min(32, len(sample))], verbose=0)
    start = time.perf_counter()
    model.predict(sample, batch_size=BATCH_SIZE, verbose=0)
    elapsed = time.perf_counter() - start
    return (elapsed * 1000) / len(sample)


class EpochLogger(keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(
            f"epoch {epoch + 1:02d}/{self.total_epochs:02d}"
            f" | acc {logs.get('accuracy', 0):.4f}"
            f" | val_acc {logs.get('val_accuracy', 0):.4f}"
        )


def main():
    args = parse_args()
    PROJECT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    keras.utils.set_random_seed(RANDOM_SEED)

    run_dir = resolve_run_dir(args.run_dir)
    paths = get_output_paths(run_dir)

    x_train, y_train, x_test, y_test = load_data()
    indices = np.random.permutation(len(x_train))
    x_train, y_train = x_train[indices], y_train[indices]
    train_samples = math.ceil(len(x_train) * (1 - VALIDATION_SPLIT))
    steps_per_epoch = math.ceil(train_samples / BATCH_SIZE)

    info = load_info()
    num_classes = info["num_classes"]
    write_mapping_file(run_dir, info)

    model = build_model(num_classes, steps_per_epoch)

    callbacks = [
        keras.callbacks.SwapEMAWeights(swap_on_epoch=True),
        keras.callbacks.ModelCheckpoint(
            filepath=str(paths["model_file"]),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
        EpochLogger(EPOCHS),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
        ),
    ]

    start_time = time.perf_counter()
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=0,
    )
    training_time = time.perf_counter() - start_time

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    inference_ms = estimate_inference_speed(model, x_test)
    model.save(paths["model_file"])

    lines = []
    model.summary(print_fn=lines.append)
    paths["summary_file"].write_text("\n".join(lines), encoding="utf-8")

    epochs_ran = len(history.history["loss"])
    best_val_accuracy = max(history.history["val_accuracy"])
    model_size_mb = paths["model_file"].stat().st_size / (1024 * 1024)

    paths["results_file"].write_text("\n".join([
        "MODEL REPORT",
        f"model_name={MODEL_NAME}",
        f"dataset={info['dataset_name']}",
        f"run_dir={run_dir}",
        "",
        "HYPERPARAMETERS",
        f"batch_size={BATCH_SIZE}",
        f"epochs_requested={EPOCHS}",
        f"epochs_ran={epochs_ran}",
        f"validation_split={VALIDATION_SPLIT}",
        f"random_seed={RANDOM_SEED}",
        "optimizer=AdamW",
        f"initial_learning_rate={INITIAL_LEARNING_RATE}",
        f"weight_decay={WEIGHT_DECAY}",
        "learning_rate_schedule=CosineDecay",
        f"decay_steps={steps_per_epoch * EPOCHS}",
        "use_ema=True",
        f"ema_momentum={EMA_MOMENTUM}",
        f"augmentation=random_rotation_0.05,random_zoom_0.05,random_translation_{TRANSLATION_FACTOR}",
        f"head=Flatten+Dense(128)+Dropout(0.4)+Dense({num_classes})",
        "",
        "MODEL INFO",
        f"num_classes={num_classes}",
        f"trainable_parameters={model.count_params()}",
        f"model_size_mb={model_size_mb:.2f}",
        "",
        "RESULTS",
        f"best_val_accuracy={best_val_accuracy:.4f}",
        f"test_loss={test_loss:.4f}",
        f"test_accuracy={test_accuracy:.4f}",
        f"training_time_seconds={training_time:.2f}",
        f"inference_ms_per_image={inference_ms:.4f}",
    ]), encoding="utf-8")

    print(f"{MODEL_NAME} | {run_dir.name} | val_acc {best_val_accuracy:.4f} | test_acc {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
