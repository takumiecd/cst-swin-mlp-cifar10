"""Plot the paired Swin MLP learning curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-cst", type=Path, help="Additional CST metrics.json"
    )
    args = parser.parse_args()

    figure, (loss_axis, accuracy_axis) = plt.subplots(
        1, 2, figsize=(10.5, 4.1), layout="constrained"
    )
    figure.suptitle("Swin MLP training on CIFAR-10", fontsize=15, fontweight="bold")
    baseline = args.run / "baseline" / "metrics.json"
    cst = args.run / "cst" / "metrics.json"
    curves = [(baseline, "Original Swin MLP", "#175d95", "-")]
    if args.reference_cst is None:
        curves.append((cst, "CST window mixers", "#af480c", "--"))
    else:
        curves.extend(
            [
                (args.reference_cst, None, "#646976", ":"),
                (cst, None, "#af480c", "--"),
            ]
        )
    records = [
        (json.loads(path.read_text()), label, color, linestyle)
        for path, label, color, linestyle in curves
    ]
    atom_counts = {
        record["config"]["atoms"] for record, label, _, _ in records if label is None
    }
    for record, label, color, linestyle in records:
        if label is None:
            config = record["config"]
            if len(atom_counts) > 1:
                label = f"CST {config['atoms']} atoms/head"
            else:
                lr = config.get("atom_lr") or config["lr"]
                label = f"CST atom LR {lr:g}"
        history = record["history"]
        epochs = [row["epoch"] for row in history]
        options = dict(label=label, color=color, linestyle=linestyle, linewidth=1.8)
        loss_axis.plot(epochs, [row["train_loss"] for row in history], **options)
        accuracy_axis.plot(
            epochs, [row["validation_accuracy"] for row in history], **options
        )

    loss_axis.set(title="Training loss", ylabel="Cross-entropy")
    accuracy_axis.set(title="Validation accuracy", ylabel="Top-1 accuracy")
    accuracy_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    for axis in (loss_axis, accuracy_axis):
        axis.set_xlabel("Epoch")
        axis.margins(x=0.015)
        axis.grid(axis="y", color="#d8e0e7", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    loss_axis.legend(frameon=False)
    figure.savefig(args.output, dpi=200, facecolor="white")


if __name__ == "__main__":
    main()
