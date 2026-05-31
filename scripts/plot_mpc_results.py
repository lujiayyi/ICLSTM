#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "MPC"


def load_series(name: str) -> np.ndarray:
    path = ROOT / name
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    return np.loadtxt(path)


def main() -> None:
    x1 = np.atleast_1d(load_series("x1.txt"))
    x2 = np.atleast_1d(load_series("x2.txt"))
    u1 = np.atleast_1d(load_series("u1.txt"))
    u2 = np.atleast_1d(load_series("u2.txt"))
    time_cost = np.atleast_1d(load_series("time_ICLSTM.txt"))

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)

    axes[0].plot(x1, label="x1 (CA deviation)", linewidth=2)
    axes[0].plot(x2, label="x2 (T deviation)", linewidth=2)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Closed-loop State Trajectory")
    axes[0].set_xlabel("Recorded step")
    axes[0].set_ylabel("State deviation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(u1, label="u1", linewidth=2)
    axes[1].plot(u2, label="u2", linewidth=2)
    axes[1].set_title("Control Inputs")
    axes[1].set_xlabel("Recorded step")
    axes[1].set_ylabel("Control value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(np.arange(len(time_cost)), time_cost, color="#4C78A8")
    axes[2].set_title("Solver Time Per MPC Iteration")
    axes[2].set_xlabel("MPC iteration")
    axes[2].set_ylabel("Seconds")
    axes[2].grid(True, axis="y", alpha=0.3)

    summary = (
        f"iterations={len(time_cost)} | "
        f"total_time={time_cost.sum():.3f}s | "
        f"avg_time={time_cost.mean():.3f}s | "
        f"final_state=({x1[-1]:.4f}, {x2[-1]:.4f})"
    )
    fig.suptitle(f"ICLSTM MPC Results\n{summary}", fontsize=14)

    out_path = ROOT / "mpc_iclstm_results.png"
    fig.savefig(out_path, dpi=180)
    print(out_path)


if __name__ == "__main__":
    main()
