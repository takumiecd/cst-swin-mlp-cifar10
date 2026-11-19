"""Export an epoch-100 checkpoint as a verified tensor-only inference file."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from infer import VARIANTS, build_model, load_model


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = build_model(args.variant).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    reference.load_state_dict(checkpoint["model"])
    tensors = {
        key: value.detach().contiguous()
        for key, value in checkpoint["model"].items()
        if isinstance(value, torch.Tensor)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    temporary = args.output.with_suffix(".tmp")
    try:
        save_file(
            tensors, temporary, metadata={"format": "pt", "variant": args.variant}
        )
        portable = load_model(args.variant, weights=temporary)
        example = torch.randn(
            (4, 3, 32, 32), generator=torch.Generator().manual_seed(17)
        )
        torch.testing.assert_close(
            portable(example), reference(example), atol=0, rtol=0
        )
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{args.variant} matched the source checkpoint logits exactly")


if __name__ == "__main__":
    main()
