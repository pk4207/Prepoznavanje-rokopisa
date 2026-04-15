import argparse
import csv
import json
import os
from html import escape
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "rezultati"
DATA_FILE = BASE_DIR / "project_data" / "prepared" / "emnist_balanced_data.npz"
OUTPUT_DIR_NAME = "primerjava"
OUTPUT_FILE_NAME = "porocilo.txt"
RESULTS_FILE_NAME = "results.txt"
MODEL_NAMES = ("osnovni", "nadgrajeni")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", help="Mapa runa. Ce ni podana, se uporabi zadnji run.")
    parser.add_argument("--output-dir", help="Mapa za porocilo.")
    return parser.parse_args()


def find_latest_run_dir():
    if not RESULTS_DIR.exists():
        raise SystemExit("Mapa rezultati/ ne obstaja. Najprej zazeni trening.")
    run_dirs = [p for p in RESULTS_DIR.iterdir() if p.is_dir()]
    if not run_dirs:
        raise SystemExit("Ni najdenih run map.")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def parse_result_value(value):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_results(run_dir, model_name):
    results_file = run_dir / model_name / RESULTS_FILE_NAME
    if not results_file.exists():
        raise SystemExit(f"Manjka: {results_file}")
    results = {}
    for line in results_file.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        results[key.strip()] = parse_result_value(value.strip())
    return results


def load_mapping(run_dir):
    mapping_file = run_dir / "mapping.json"
    if not mapping_file.exists():
        raise SystemExit(f"Manjka: {mapping_file}")
    return json.loads(mapping_file.read_text(encoding="utf-8"))


def load_test_data():
    if not DATA_FILE.exists():
        raise SystemExit("Manjka pripravljen dataset. Najprej zazeni priprava.py")
    with np.load(DATA_FILE, allow_pickle=False) as data:
        return data["x_test"], data["y_test"]


def predict_classes(run_dir, model_name, x_test):
    model_file = run_dir / model_name / "model.keras"
    if not model_file.exists():
        raise SystemExit(f"Manjka model: {model_file}")
    model = keras.models.load_model(model_file)
    predictions = model.predict(x_test, batch_size=256, verbose=0)
    return np.argmax(predictions, axis=1).astype(np.int64)


def compute_confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def normalize_confusion_matrix(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix, row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums != 0,
    )


def write_confusion_csv(file_path, matrix, labels, float_values=False):
    with file_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix, strict=False):
            if float_values:
                writer.writerow([label, *[f"{v:.6f}" for v in row]])
            else:
                writer.writerow([label, *row.tolist()])


def cell_color(value, max_val):
    ratio = float(value) / max(float(max_val), 0.001)
    return f"rgb({int(255 - ratio * 30)},{int(244 - ratio * 80)},{int(248 - ratio * 120)})"


def text_color(value, max_val):
    ratio = float(value) / max(float(max_val), 0.001)
    return "#fffaf3" if ratio >= 0.58 else "#1f2223"


def write_confusion_svg(file_path, matrix, labels, title, normalized=False):
    n = len(labels)
    cell = 18 if n > 40 else 20
    left, top, bottom, right = 96, 58, 156, 42
    footer = 56
    w = left + n * cell + right
    h = top + n * cell + bottom + footer
    max_val = float(np.max(matrix)) if matrix.size else 0.0

    def format_value(value):
        return f"{value:.2f}" if normalized else str(int(value))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fffaf3" />',
        f'<text x="{left}" y="28" font-size="18" font-family="Arial, sans-serif" font-weight="700" fill="#1f2223">{escape(title)}</text>',
        f'<text x="{left}" y="46" font-size="11" font-family="Arial, sans-serif" fill="#65706d">Temnejsa celica = vec primerov</text>',
        f'<rect x="{left}" y="{top}" width="{n * cell}" height="{n * cell}" fill="none" stroke="#d8cfbf" stroke-width="1.2" />',
    ]

    for i, label in enumerate(labels):
        y = top + i * cell
        lines.append(
            f'<text x="{left - 10}" y="{y + 12}" text-anchor="end" font-size="9" '
            f'font-family="Arial, sans-serif" fill="#1f2223">{escape(label)}</text>'
        )

    for j, label in enumerate(labels):
        x = left + j * cell + cell / 2
        y = top + n * cell + 26
        lines.append(
            f'<text x="{x}" y="{y}" text-anchor="end" transform="rotate(-55 {x} {y})" '
            f'font-size="9" font-family="Arial, sans-serif" fill="#1f2223">{escape(label)}</text>'
        )

    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            x = left + j * cell
            y = top + i * cell
            sw = "1.6" if i == j else "1"
            sc = "#c97d67" if i == j else "#e4ddd0"
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{cell_color(v, max_val)}" stroke="{sc}" stroke-width="{sw}" />'
            )
            if i == j:
                lines.append(
                    f'<text x="{x + cell / 2}" y="{y + 13}" text-anchor="middle" '
                    f'font-size="8" font-family="Arial, sans-serif" fill="{text_color(v, max_val)}">{escape(format_value(v))}</text>'
                )

    lines.append("</svg>")
    file_path.write_text("\n".join(lines), encoding="utf-8")


def compare_metric(metric, results_a, results_b, higher_is_better=True):
    a, b = float(results_a[metric]), float(results_b[metric])
    if higher_is_better:
        winner = MODEL_NAMES[0] if a > b else MODEL_NAMES[1] if b > a else "izenaceno"
    else:
        winner = MODEL_NAMES[0] if a < b else MODEL_NAMES[1] if b < a else "izenaceno"
    return winner, a, b


def safe_ratio(num, den):
    return float(num) / max(float(den), 0.01)


def format_metric_value(val):
    return str(int(val)) if float(val).is_integer() else f"{val:.4f}"


def format_head_to_head_line(label, metric_result, unit=""):
    winner, osnovni_value, nadgrajeni_value = metric_result
    unit_suffix = f" {unit}" if unit else ""
    return (
        f"{label}: {winner} "
        f"(osnovni={format_metric_value(osnovni_value)}{unit_suffix}, "
        f"nadgrajeni={format_metric_value(nadgrajeni_value)}{unit_suffix})"
    )


def main():
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run_dir()
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    res_a = load_results(run_dir, MODEL_NAMES[0])
    res_b = load_results(run_dir, MODEL_NAMES[1])

    acc = compare_metric("test_accuracy", res_a, res_b, higher_is_better=True)
    val_acc = compare_metric("best_val_accuracy", res_a, res_b, higher_is_better=True)
    speed = compare_metric("training_time_seconds", res_a, res_b, higher_is_better=False)
    infer = compare_metric("inference_ms_per_image", res_a, res_b, higher_is_better=False)
    size = compare_metric("model_size_mb", res_a, res_b, higher_is_better=False)
    params = compare_metric("trainable_parameters", res_a, res_b, higher_is_better=False)

    acc_gap_pp = (float(res_b["test_accuracy"]) - float(res_a["test_accuracy"])) * 100
    val_gap_pp = (float(res_b["best_val_accuracy"]) - float(res_a["best_val_accuracy"])) * 100
    time_ratio = safe_ratio(res_b["training_time_seconds"], res_a["training_time_seconds"])

    if acc_gap_pp >= 1.0:
        recommended, reason = MODEL_NAMES[1], "Nadgrajeni model doseze opazno visjo natancnost."
    elif acc_gap_pp > 0:
        recommended, reason = MODEL_NAMES[1], "Nadgrajeni model je nekoliko natancnejsi."
    elif acc_gap_pp < 0:
        recommended, reason = MODEL_NAMES[0], "Nadgrajeni model ni izboljsal natancnosti."
    else:
        recommended, reason = MODEL_NAMES[0], "Osnovni model je preprostejsi in hitrejsi."

    mapping = load_mapping(run_dir)
    x_test, y_test = load_test_data()

    pred_a = predict_classes(run_dir, MODEL_NAMES[0], x_test)
    pred_b = predict_classes(run_dir, MODEL_NAMES[1], x_test)

    cm_a = compute_confusion_matrix(y_test, pred_a, mapping["num_classes"])
    cm_b = compute_confusion_matrix(y_test, pred_b, mapping["num_classes"])
    cm_a_norm = normalize_confusion_matrix(cm_a)
    cm_b_norm = normalize_confusion_matrix(cm_b)

    write_confusion_csv(output_dir / "osnovni_cm.csv", cm_a, mapping["labels"])
    write_confusion_csv(output_dir / "osnovni_cm_norm.csv", cm_a_norm, mapping["labels"], float_values=True)
    write_confusion_csv(output_dir / "nadgrajeni_cm.csv", cm_b, mapping["labels"])
    write_confusion_csv(output_dir / "nadgrajeni_cm_norm.csv", cm_b_norm, mapping["labels"], float_values=True)

    write_confusion_svg(output_dir / "osnovni_cm.svg", cm_a, mapping["labels"], "Osnovni model — confusion matrix")
    write_confusion_svg(output_dir / "osnovni_cm_norm.svg", cm_a_norm, mapping["labels"], "Osnovni model — normalized", normalized=True)
    write_confusion_svg(output_dir / "nadgrajeni_cm.svg", cm_b, mapping["labels"], "Nadgrajeni model — confusion matrix")
    write_confusion_svg(output_dir / "nadgrajeni_cm_norm.svg", cm_b_norm, mapping["labels"], "Nadgrajeni model — normalized", normalized=True)

    report = "\n".join([
        "PRIMERJAVA MODELOV",
        f"run_dir={run_dir}",
        f"dataset={res_a['dataset']}",
        "",
        "HEAD TO HEAD",
        format_head_to_head_line("test_accuracy", acc),
        f"test_accuracy_razlika_pp={acc_gap_pp:+.2f}",
        format_head_to_head_line("best_val_accuracy", val_acc),
        f"val_accuracy_razlika_pp={val_gap_pp:+.2f}",
        format_head_to_head_line("training_time_seconds", speed, "s"),
        f"cas_treniranja_razmerje={time_ratio:.2f}x",
        format_head_to_head_line("inference_ms_per_image", infer, "ms"),
        format_head_to_head_line("model_size_mb", size, "MB"),
        format_head_to_head_line("trainable_parameters", params),
        "",
        "PRIPOROCILO",
        f"priporocen_model={recommended}",
        f"razlog={reason}",
    ])

    (output_dir / OUTPUT_FILE_NAME).write_text(report, encoding="utf-8")

    print(f"Run: {run_dir.name}")
    print(f"Porocilo: {output_dir / OUTPUT_FILE_NAME}")
    print(f"Test acc razlika: {acc_gap_pp:+.2f} pp | Priporocen: {recommended}")


if __name__ == "__main__":
    main()
