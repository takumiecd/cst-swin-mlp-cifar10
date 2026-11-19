"""Train the official Swin MLP implementation with optional CST window mixing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from math import isfinite
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter

import torch
from torch import nn

SWIN_COMMIT = "f82860bfb5225915aca09c3227159ee9e1df874d"
SWIN_SHA256 = "749c6f6601d6d0e83303e24fc317b992ddb6260e4907f8f0bcfa722cea15d872"
MODEL_OPTIONS = {
    "img_size": 32,
    "patch_size": 2,
    "num_classes": 10,
    "embed_dim": 64,
    "depths": (2, 2),
    "num_heads": (2, 4),
    "window_size": 4,
    "mlp_ratio": 2.0,
    "drop_path_rate": 0.0,
}


def load_swin(path: Path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != SWIN_SHA256:
        raise ValueError(f"Expected Microsoft's swin_mlp.py from {SWIN_COMMIT}")
    spec = importlib.util.spec_from_file_location("microsoft_swin_mlp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CSTSpatialMix(nn.Module):
    """One CSTLinear per head, shared over windows and within-head channels."""

    def __init__(self, original: nn.Conv1d, window_side: int, atom_count: int):
        super().__init__()
        from torchcst import (
            BandwidthBounds,
            CSTLinear,
            TriweightSpec,
            chart_presets,
            kernels,
        )

        self.sites = window_side**2
        self.heads = original.groups
        self.bias = original.bias
        chart = chart_presets.regular_grid((window_side, window_side))
        kernel = kernels.product(
            kernels.l2(TriweightSpec()),
            coordinates=kernels.polar(
                amplitude_max=1.0,
                w_c=0.1,
                bounds=BandwidthBounds(
                    minimum=0.4, birth=1.6, maximum=2.4, upper_floor=0.4
                ),
            ),
        )
        self.mixers = nn.ModuleList(
            CSTLinear(
                charts=(chart, chart),
                atoms=atom_count,
                kernel=kernel,
                backend="factored",
            )
            for _ in range(self.heads)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        windows = inputs.unflatten(1, (self.heads, self.sites)).transpose(2, 3)
        mixed = torch.stack(
            [mixer(windows[:, head]) for head, mixer in enumerate(self.mixers)],
            dim=1,
        )
        mixed = mixed.transpose(2, 3).flatten(1, 2)
        return mixed + self.bias[None, :, None]


def make_model(
    swin,
    variant: str,
    atoms: int,
    seed: int,
    *,
    embed_dim: int = MODEL_OPTIONS["embed_dim"],
    window_size: int = MODEL_OPTIONS["window_size"],
) -> nn.Module:
    torch.manual_seed(seed)
    model = swin.SwinMLP(
        **(MODEL_OPTIONS | {"embed_dim": embed_dim, "window_size": window_size})
    )
    if variant == "cst":
        for block in model.modules():
            if isinstance(block, swin.SwinMLPBlock):
                block.spatial_mlp = CSTSpatialMix(
                    block.spatial_mlp, block.window_size, atoms
                )
    return model


def shared_parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if ".spatial_mlp." not in name or name.endswith(".bias"):
            digest.update(name.encode())
            digest.update(parameter.detach().numpy().tobytes())
    return digest.hexdigest()


def make_loaders(args):
    from torchvision import datasets, transforms

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_images = datasets.CIFAR10(
        args.data,
        train=True,
        download=args.download,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    validation_images = datasets.CIFAR10(
        args.data,
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    order = torch.randperm(
        len(train_images), generator=torch.Generator().manual_seed(905)
    )
    val_indices, train_indices = order[:5000], order[5000:]
    options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_images, train_indices.tolist()),
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **options,
    )
    validation = torch.utils.data.DataLoader(
        torch.utils.data.Subset(validation_images, val_indices.tolist()), **options
    )
    return train, validation, hashlib.sha256(order.numpy().tobytes()).hexdigest()


def update(model, optimizer, images, targets):
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(images), targets)
    loss.backward()
    optimizer.step()
    return loss.detach()


def cuda_peak():
    return {
        "allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def benchmark(args, loader, model, optimizer, atom_parameters):
    timing_ms = []
    batch_to_update_ms = []
    losses = []
    iterator = iter(loader)
    initial_atoms = [parameter.detach().clone() for parameter in atom_parameters]
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.warmup_steps + args.benchmark_steps):
        batch_start = perf_counter()
        images, targets = next(iterator)
        if step == 0:
            first_batch_hash = hashlib.sha256(
                images.numpy().tobytes() + targets.numpy().tobytes()
            ).hexdigest()
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        torch.cuda.synchronize()
        if step == args.warmup_steps:
            peak_with_warmup = cuda_peak()
            torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        loss = update(model, optimizer, images, targets)
        torch.cuda.synchronize()
        end = perf_counter()
        if step >= args.warmup_steps:
            timing_ms.append(1000 * (end - start))
            batch_to_update_ms.append(1000 * (end - batch_start))
        losses.append(loss.item())

    peak = cuda_peak()
    if not torch.tensor(losses).isfinite().all():
        raise RuntimeError("Training produced a nonfinite loss")
    for parameter, initial in zip(atom_parameters, initial_atoms, strict=True):
        if parameter.grad is None or not parameter.grad.isfinite().all():
            raise RuntimeError("Missing or nonfinite CST atom gradient")
        if not parameter.isfinite().all() or torch.equal(parameter, initial):
            raise RuntimeError("CST atom parameters did not receive a finite update")
    return {
        "first_batch_sha256": first_batch_hash,
        "steps_measured": len(timing_ms),
        "step_ms": timing_ms,
        "median_ms": median(timing_ms),
        "p90_ms": quantiles(timing_ms, n=10, method="inclusive")[8],
        "batch_to_update_ms": batch_to_update_ms,
        "median_batch_to_update_ms": median(batch_to_update_ms),
        "peak": peak,
        "peak_including_warmup": {
            key: max(value, peak[key]) for key, value in peak_with_warmup.items()
        },
        "losses": losses,
        "cst_heads_verified": len(atom_parameters),
    }


@torch.no_grad()
def validation_accuracy(model, loader) -> float:
    model.eval()
    correct = torch.zeros((), dtype=torch.int64, device="cuda")
    for images, targets in loader:
        logits = model(images.cuda(non_blocking=True))
        correct += (logits.argmax(dim=1) == targets.cuda(non_blocking=True)).sum()
    return correct.item() / len(loader.dataset)


def train(args, train_loader, validation_loader, model, optimizer):
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = torch.zeros((), device="cuda")
        for images, targets in train_loader:
            loss = update(
                model,
                optimizer,
                images.cuda(non_blocking=True),
                targets.cuda(non_blocking=True),
            )
            total_loss += loss * len(targets)
        row = {
            "epoch": epoch,
            "train_loss": total_loss.item() / len(train_loader.dataset),
            "validation_accuracy": validation_accuracy(model, validation_loader),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        schedule.step()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": schedule.state_dict(),
        },
        args.output / "checkpoint.pt",
    )
    return {"history": history}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "cst"), required=True)
    parser.add_argument(
        "--swin-source",
        type=Path,
        default=Path(__file__).resolve().parent / "third_party" / "swin_mlp.py",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--atoms", type=int, default=32, help="Atoms per CST head")
    parser.add_argument("--embed-dim", type=int, default=MODEL_OPTIONS["embed_dim"])
    parser.add_argument(
        "--window-size", type=int, choices=(4, 8), default=MODEL_OPTIONS["window_size"]
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--atom-lr", type=float, help="Initial CST atom group rate (defaults to --lr)"
    )
    parser.add_argument(
        "--benchmark-steps", type=int, default=0, help="Run a bounded timing pilot"
    )
    parser.add_argument("--warmup-steps", type=int, default=20)
    args = parser.parse_args()
    if args.benchmark_steps == 1 or args.benchmark_steps < 0 or args.warmup_steps < 1:
        parser.error(
            "benchmark requires at least two measured steps and one warmup step"
        )
    if (
        min(args.batch_size, args.atoms, args.epochs) <= 0
        or args.workers < 0
        or args.lr <= 0
    ):
        parser.error("invalid training configuration")
    if args.atom_lr is not None and (
        args.variant != "cst" or not isfinite(args.atom_lr) or args.atom_lr <= 0
    ):
        parser.error("--atom-lr must be positive and finite and requires --variant cst")
    if args.embed_dim < 2 or args.embed_dim % MODEL_OPTIONS["num_heads"][0]:
        parser.error(
            "embed-dim must be positive and divisible by the first-stage head count"
        )
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    import timm
    import torchvision

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    size_options = {"embed_dim": args.embed_dim, "window_size": args.window_size}
    model = make_model(
        load_swin(args.swin_source), args.variant, args.atoms, args.seed, **size_options
    )
    shared_hash = shared_parameter_hash(model)
    model.cuda()
    optimizer_options = {"lr": args.lr, "weight_decay": 0.05, "foreach": True}
    atom_parameters = []
    if args.variant == "cst":
        from torchcst import CSTAdamW

        for module in model.modules():
            if isinstance(module, CSTSpatialMix):
                atom_parameters.extend(layer.atoms.p for layer in module.mixers)
        atom_ids = {id(parameter) for parameter in atom_parameters}
        optimizer = CSTAdamW(
            [
                {
                    "params": [p for p in model.parameters() if id(p) not in atom_ids],
                },
                {
                    "params": atom_parameters,
                    "weight_decay": 0.0,
                    "lr": args.atom_lr if args.atom_lr is not None else args.lr,
                },
            ],
            model=model,
            amplitude_weight_decay=optimizer_options["weight_decay"],
            **optimizer_options,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_options)

    train_loader, validation_loader, split_hash = make_loaders(args)
    torch.manual_seed(args.seed)
    if args.benchmark_steps and args.warmup_steps + args.benchmark_steps > len(
        train_loader
    ):
        parser.error("timing pilot must fit within one epoch")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.cuda.get_device_properties(0)
    result = {
        "config": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "model": MODEL_OPTIONS | size_options,
        "window_layout": [
            [
                {"side": block.window_size, "shift": block.shift_size}
                for block in layer.blocks
            ]
            for layer in model.layers
        ],
        "window_weight_parameters": sum(
            param.numel()
            for name, param in model.named_parameters()
            if ".spatial_mlp." in name and not name.endswith(".bias")
        ),
        "swin_commit": SWIN_COMMIT,
        "shared_initial_parameters_sha256": shared_hash,
        "split_sha256": split_hash,
        "train_examples": len(train_loader.dataset),
        "validation_examples": len(validation_loader.dataset),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": device.name,
        "device_bytes": device.total_memory,
        "device_capability": torch.cuda.get_device_capability(),
        "precision": "eager FP32, TF32 off",
        "initial_learning_rates": [group["lr"] for group in optimizer.param_groups],
    }
    if args.benchmark_steps:
        result.update(benchmark(args, train_loader, model, optimizer, atom_parameters))
        print(
            f"{args.variant}: {result['median_ms']:.2f} ms median, "
            f"{result['p90_ms']:.2f} ms p90; GPU {device.name}",
            flush=True,
        )
    else:
        result.update(train(args, train_loader, validation_loader, model, optimizer))
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
