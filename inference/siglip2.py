from __future__ import annotations

import argparse
import inspect
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

CHECKPOINT_ROOT = Path(os.environ.get("FLATCLIP_CHECKPOINT_ROOT", "<path>")).expanduser()
CHECKPOINT_MAP = {
    "base": CHECKPOINT_ROOT,
    "naflex": CHECKPOINT_ROOT,
    "so400m": CHECKPOINT_ROOT,
    "large": CHECKPOINT_ROOT,
}
IMAGE_SIZE_MAP = {
    "base": 224,
    "naflex": 224,
    "so400m": 384,
    "large": 384,
}


@dataclass
class SigLIP2Bundle:
    model: torch.nn.Module
    image_processor: object
    tokenizer: object
    variant: str
    image_size: int
    checkpoint_dir: Path


def load_model(device: str, variant: str = "base") -> SigLIP2Bundle:
    if variant not in CHECKPOINT_MAP:
        raise ValueError(f"Unsupported SigLIP 2 variant: {variant}")

    checkpoint_dir = CHECKPOINT_MAP[variant]
    model = AutoModel.from_pretrained(checkpoint_dir, local_files_only=True)
    image_processor = AutoImageProcessor.from_pretrained(checkpoint_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    model.eval().to(device)
    return SigLIP2Bundle(
        model=model,
        image_processor=image_processor,
        tokenizer=tokenizer,
        variant=variant,
        image_size=IMAGE_SIZE_MAP[variant],
        checkpoint_dir=checkpoint_dir,
    )


def move_to_device(inputs: dict[str, object], device: str | torch.device) -> dict[str, object]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def preprocess_images(
    bundle: SigLIP2Bundle,
    images: Image.Image | list[Image.Image],
    device: str | torch.device | None = None,
) -> dict[str, object]:
    inputs = dict(bundle.image_processor(images=images, return_tensors="pt"))
    if device is not None:
        inputs = move_to_device(inputs, device)
    return inputs


def _call_with_supported_kwargs(fn: object, kwargs: dict[str, object]) -> object:
    target = fn.forward if isinstance(fn, torch.nn.Module) else fn
    signature = inspect.signature(target)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        supported = kwargs
    else:
        supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return fn(**supported)


def extract_features(
    bundle: SigLIP2Bundle,
    pixel_values: torch.Tensor | None = None,
    **image_inputs: object,
) -> dict[str, torch.Tensor]:
    if pixel_values is not None:
        image_inputs["pixel_values"] = pixel_values
    if "pixel_values" not in image_inputs:
        raise ValueError("extract_features requires pixel_values or processor image inputs")

    device = next(bundle.model.parameters()).device
    image_inputs = move_to_device(dict(image_inputs), device)
    image_embeds = _call_with_supported_kwargs(bundle.model.get_image_features, image_inputs)
    features: dict[str, torch.Tensor] = {
        "cls": image_embeds,
        "image_embeds": image_embeds,
    }
    vision_inputs = {**image_inputs, "output_hidden_states": False, "return_dict": True}
    if "attention_mask" not in vision_inputs and "pixel_attention_mask" in vision_inputs:
        vision_inputs["attention_mask"] = vision_inputs["pixel_attention_mask"]
    vision_outputs = _call_with_supported_kwargs(bundle.model.vision_model, vision_inputs)
    last_hidden_state = getattr(vision_outputs, "last_hidden_state", None)
    if last_hidden_state is not None and last_hidden_state.shape[1] > 1:
        features["patch_mean"] = last_hidden_state[:, 1:, :].mean(dim=1)
        features["patch_tokens"] = last_hidden_state[:, 1:, :]
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Load local SigLIP 2 and run a test forward pass.")
    parser.add_argument("--variant", choices=sorted(CHECKPOINT_MAP), default="base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    bundle = load_model(args.device, args.variant)
    image_size = bundle.image_size if args.image_size == 224 else args.image_size
    images = [Image.new("RGB", (image_size, image_size), color=(128, 128, 128)) for _ in range(args.batch_size)]
    image_inputs = preprocess_images(bundle, images, device=args.device)

    with torch.inference_mode():
        features = extract_features(bundle, **image_inputs)
        text_tokens = bundle.tokenizer(["hello world"], return_tensors="pt")
        text_features = bundle.model.get_text_features(**{k: v.to(args.device) for k, v in text_tokens.items()})

    print(f"checkpoint={CHECKPOINT_MAP[args.variant]}")
    print(f"variant={args.variant}")
    print(f"device={args.device}")
    print(f"image_size={image_size}")
    print(f"image_inputs={sorted(image_inputs)}")
    print(f"cls={tuple(features['cls'].shape)}")
    print(f"image_embeds={tuple(features['image_embeds'].shape)}")
    patch_shape = tuple(features["patch_mean"].shape) if "patch_mean" in features else None
    print(f"patch_mean={patch_shape}")
    print(f"text_features={tuple(text_features.shape)}")
    print(f"processor_type={type(bundle.image_processor).__name__}")


if __name__ == "__main__":
    main()
