"""Load the released Swin MLP weights and classify a CIFAR-10-style image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

from train import load_swin, make_model

HUB_REPO = "takumiecd/cst-swin-mlp-cifar10"
SWIN_SOURCE = Path(__file__).resolve().parent / "third_party" / "swin_mlp.py"
VARIANTS = {
    "baseline": ("baseline", 128),
    "cst-128": ("cst", 128),
    "cst-256": ("cst", 256),
}
CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
PREPROCESS = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)


def build_model(variant: str) -> torch.nn.Module:
    architecture, atoms = VARIANTS[variant]
    with torch.random.fork_rng(devices=[]):
        return make_model(
            load_swin(SWIN_SOURCE),
            architecture,
            atoms,
            17,
            embed_dim=256,
            window_size=8,
        )


def load_model(
    variant: str = "cst-256",
    *,
    weights: str | Path | None = None,
    device: str = "cpu",
    revision: str | None = None,
) -> torch.nn.Module:
    model = build_model(variant)
    if weights is None:
        weights = hf_hub_download(
            HUB_REPO, f"{variant}.safetensors", revision=revision, token=False
        )
    incoming = load_file(weights)
    state = model.state_dict()
    expected = {key for key, value in state.items() if isinstance(value, torch.Tensor)}
    if incoming.keys() != expected:
        raise ValueError(f"Tensor keys in {weights} do not match the {variant} model")
    state.update(incoming)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.inference_mode()
def predict_image(
    model: torch.nn.Module, image: Image.Image, top_k: int = 3
) -> list[dict]:
    device = next(model.parameters()).device
    batch = PREPROCESS(image.convert("RGB")).unsqueeze(0).to(device)
    scores, indices = model(batch).softmax(dim=1)[0].topk(top_k)
    return [
        {"label": CLASSES[index], "score": score}
        for index, score in zip(indices.tolist(), scores.tolist(), strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="cst-256")
    parser.add_argument("--weights", type=Path, help="Local safetensors file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--revision", help="Hugging Face commit or tag")
    parser.add_argument(
        "--top-k", type=int, choices=range(1, len(CLASSES) + 1), default=3
    )
    args = parser.parse_args()
    model = load_model(
        args.variant, weights=args.weights, device=args.device, revision=args.revision
    )
    with Image.open(args.image) as image:
        print(json.dumps(predict_image(model, image, args.top_k), indent=2))


if __name__ == "__main__":
    main()
