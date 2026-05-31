#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MPC_DIR = ROOT / "MPC"


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    notebook_stem: str
    time_file: str
    color: str


METHODS = [
    MethodSpec("first_principles", "First principles", "mpc_first_principles", "time_fp.txt", "#4C78A8"),
    MethodSpec("rnn", "RNN", "mpc_rnn", "time_RNN.txt", "#F58518"),
    MethodSpec("lstm", "LSTM", "mpc_lstm", "time_LSTM.txt", "#54A24B"),
    MethodSpec("icrnn", "ICRNN", "mpc_icrnn", "time_ICRNN.txt", "#E45756"),
    MethodSpec("iclstm", "ICLSTM", "mpc_iclstm", "time_ICLSTM.txt", "#B279A2"),
]

SERIES_KEYS = ["x1_record", "x2_record", "u1_record", "u2_record", "time_record"]


def parse_printed_list(stream_text: str, key: str) -> np.ndarray | None:
    match = re.search(rf"{key}:\s*(\[[^\]]*\])", stream_text, re.S)
    if not match:
        return None
    values = ast.literal_eval(match.group(1))
    return np.asarray(values, dtype=float)


def load_from_executed_notebook(spec: MethodSpec) -> dict[str, np.ndarray] | None:
    notebook_path = MPC_DIR / f"{spec.notebook_stem}.executed.ipynb"
    if not notebook_path.exists():
        return None

    notebook = json.loads(notebook_path.read_text())
    stream_text = ""
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                stream_text += "".join(output.get("text", ""))

    result: dict[str, np.ndarray] = {}
    for key in SERIES_KEYS:
        values = parse_printed_list(stream_text, key)
        if values is None:
            return None
        result[key] = values
    return result


def load_from_result_dir(spec: MethodSpec) -> dict[str, np.ndarray] | None:
    result_dir = MPC_DIR / "results" / spec.key
    if not result_dir.exists():
        return None

    file_map = {
        "x1_record": result_dir / "x1.txt",
        "x2_record": result_dir / "x2.txt",
        "u1_record": result_dir / "u1.txt",
        "u2_record": result_dir / "u2.txt",
        "time_record": result_dir / spec.time_file,
    }
    if not all(path.exists() for path in file_map.values()):
        return None

    return {key: np.atleast_1d(np.loadtxt(path)) for key, path in file_map.items()}


def load_method_data(spec: MethodSpec) -> dict[str, np.ndarray] | None:
    return load_from_result_dir(spec) or load_from_executed_notebook(spec)


def format_summary(data: dict[str, np.ndarray]) -> str:
    time_values = data["time_record"]
    return (
        f"n={len(time_values)}, avg={time_values.mean():.2f}s, "
        f"median={np.median(time_values):.2f}s, total={time_values.sum():.1f}s"
    )


def main() -> None:
    available: list[tuple[MethodSpec, dict[str, np.ndarray]]] = []
    missing: list[str] = []

    for spec in METHODS:
        data = load_method_data(spec)
        if data is None:
            missing.append(spec.label)
        else:
            available.append((spec, data))

    if not available:
        raise SystemExit("No MPC result data found. Run the notebooks first.")

    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax_x1 = fig.add_subplot(gs[0, 0])
    ax_x2 = fig.add_subplot(gs[0, 1])
    ax_u1 = fig.add_subplot(gs[1, 0])
    ax_u2 = fig.add_subplot(gs[1, 1])
    ax_time = fig.add_subplot(gs[2, :])

    for spec, data in available:
        ax_x1.plot(data["x1_record"], label=spec.label, color=spec.color, linewidth=2)
        ax_x2.plot(data["x2_record"], label=spec.label, color=spec.color, linewidth=2)
        ax_u1.plot(data["u1_record"], label=spec.label, color=spec.color, linewidth=2)
        ax_u2.plot(data["u2_record"], label=spec.label, color=spec.color, linewidth=2)

    for ax, title, ylabel in [
        (ax_x1, "State trajectory: x1", "x1"),
        (ax_x2, "State trajectory: x2", "x2"),
        (ax_u1, "Control input: u1", "u1"),
        (ax_u2, "Control input: u2", "u2"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Recorded step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)

    labels = [spec.label for spec, _ in available]
    avg_times = [data["time_record"].mean() for _, data in available]
    median_times = [np.median(data["time_record"]) for _, data in available]
    totals = [data["time_record"].sum() for _, data in available]
    colors = [spec.color for spec, _ in available]

    x = np.arange(len(labels))
    width = 0.25
    ax_time.bar(x - width, avg_times, width=width, label="Avg time / iter", color=colors, alpha=0.9)
    ax_time.bar(x, median_times, width=width, label="Median time / iter", color=colors, alpha=0.55)
    ax_time.bar(x + width, totals, width=width, label="Total elapsed time", color=colors, alpha=0.3)
    ax_time.set_xticks(x, labels)
    ax_time.set_title("Solve-time comparison")
    ax_time.set_ylabel("Seconds")
    ax_time.grid(True, axis="y", alpha=0.3)
    ax_time.legend()

    ax_x1.legend(loc="best")
    ax_x2.legend(loc="best")

    summary_lines = [f"{spec.label}: {format_summary(data)}" for spec, data in available]
    if missing:
        summary_lines.append("Missing: " + ", ".join(missing))
    fig.suptitle("MPC method comparison\n" + "\n".join(summary_lines), fontsize=13)

    out_path = MPC_DIR / "mpc_method_comparison.png"
    fig.savefig(out_path, dpi=180)

    print(f"saved: {out_path}")
    if missing:
        print("missing methods:", ", ".join(missing))
    else:
        print("all five methods are included")


if __name__ == "__main__":
    main()
