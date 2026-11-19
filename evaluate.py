"""Evaluate the final paired Swin MLP checkpoints on the official CIFAR-10 test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision import datasets, transforms

from train import (
    load_swin,
    make_model,
    validation_accuracy,
)


def load_pair(run: Path) -> dict:
    records = {
        variant: json.loads((run / variant / "metrics.json").read_text())
        for variant in ("baseline", "cst")
    }
    baseline, cst = records.values()
    if any(
        baseline[field] != cst[field]
        for field in ("model", "shared_initial_parameters_sha256", "split_sha256")
    ):
        raise ValueError("The two runs have different model or data conditions")
    for field in ("epochs", "seed", "batch_size", "workers", "lr"):
        if baseline["config"][field] != cst["config"][field]:
            raise ValueError(f"The two runs have different {field}")
    for variant, record in records.items():
        config = record["config"]
        epochs = [row["epoch"] for row in record["history"]]
        if config["variant"] != variant or epochs != list(
            range(1, config["epochs"] + 1)
        ):
            raise ValueError(f"Incomplete or mislabelled training record for {variant}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--swin-source",
        type=Path,
        default=Path(__file__).resolve().parent / "third_party" / "swin_mlp.py",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = load_pair(args.run)
    swin = load_swin(args.swin_source)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dataset = datasets.CIFAR10(
        args.data,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616),
                ),
            ]
        ),
    )
    reference = records["baseline"]["config"]
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=reference["batch_size"],
        num_workers=reference["workers"],
        pin_memory=True,
    )
    results = {}
    for variant, record in records.items():
        config = record["config"]
        model = make_model(
            swin,
            variant,
            config["atoms"],
            config["seed"],
            embed_dim=config["embed_dim"],
            window_size=config["window_size"],
        )
        checkpoint = torch.load(
            args.run / variant / "checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model"])
        if not all(parameter.isfinite().all() for parameter in model.parameters()):
            raise RuntimeError(f"Nonfinite model parameters for {variant}")
        accuracy = validation_accuracy(model.cuda(), loader)
        results[variant] = {
            "epoch": config["epochs"],
            "examples": len(dataset),
            "correct": round(accuracy * len(dataset)),
            "top1_accuracy": accuracy,
        }
    output = {"split": "official CIFAR-10 test", "results": results}
    with args.output.open("x") as file:
        json.dump(output, file, indent=2)
        file.write("\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
