#!/usr/bin/env python3
"""
generate_sd_for_crossstitch.py

Generates images from Stable Diffusion (Hugging Face diffusers) and
post-processes them for cross-stitch (alpha) patterns:
 - saves raw generation
 - creates reduced-color (adaptive palette) version
 - resizes to a pattern grid (nearest neighbor) and saves a scaled preview

Usage examples:
  python generate_sd_for_crossstitch.py --prompt "cute fox pixel art, 40x40, flat colors, pixel-art" --pattern_size 40
  python generate_sd_for_crossstitch.py --prompt_file prompts.txt --num_images 3 --n_colors 12

Notes:
 - Many HF models require a Hugging Face access token (HF_TOKEN). See README notes below.
"""

import os
import argparse
import time
import random
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps
import numpy as np

# Use diffusers + torch. Make sure these packages are installed before running.
try:
    import torch
    from diffusers import StableDiffusionPipeline, DiffusionPipeline
except Exception as e:
    raise RuntimeError(
        "Missing dependencies. Install them with:\n"
        "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # (or cpu)\n"
        "  pip install diffusers transformers accelerate safetensors"
    ) from e


def load_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5",
                  device: Optional[str] = None,
                  use_auth_token: Optional[str] = None,
                  enable_xformers: bool = True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = torch.float16 if (device.startswith("cuda") and torch.cuda.is_available()) else torch.float32

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_auth_token=use_auth_token,  # If None, diffusers will try anonymous access
        safety_checker=None
    )
    pipe = pipe.to(device)

    if enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    # If using CPU, enable attention slicing 
    if device == "cpu":
        pipe.enable_attention_slicing()

    return pipe


def generate_images(pipe,
                    prompt: str,
                    num_images: int = 1,
                    height: int = 256,
                    width: int = 256,
                    steps: int = 20,
                    guidance_scale: float = 7.5,
                    seed: Optional[int] = None):
    generator = None
    if seed is None:
        seed = random.randrange(2**31 - 1)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    results = []
    for i in range(num_images):
        out = pipe(prompt=prompt,
                   height=height,
                   width=width,
                   num_inference_steps=steps,
                   guidance_scale=guidance_scale,
                   generator=generator)
        img = out.images[0].convert("RGBA")  
        results.append(img)
        generator = torch.Generator(device=pipe.device).manual_seed(seed + i + 1)

    return results, seed


def reduce_colors_adaptive(img: Image.Image, n_colors: int = 12) -> Image.Image:
    rgb = img.convert("RGB")
    p = rgb.convert("P", palette=Image.ADAPTIVE, colors=n_colors)
    reduced = p.convert("RGB")
    return reduced


def resize_to_pattern_grid(img: Image.Image, pattern_size: int = 40) -> Image.Image:
    # If image is not square, we center-crop to square first
    w, h = img.size
    min_side = min(w, h)
    img_sq = ImageOps.fit(img, (min_side, min_side), method=Image.NEAREST, centering=(0.5, 0.5))
    small = img_sq.resize((pattern_size, pattern_size), resample=Image.NEAREST)
    return small


def save_scaled_preview(small_img: Image.Image, scale: int = 16, out_path: Path = None):
    big = small_img.resize((small_img.width * scale, small_img.height * scale), resample=Image.NEAREST)
    if out_path:
        big.save(out_path)
    return big


def ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)



def main():
    parser = argparse.ArgumentParser(description="Generate Stable Diffusion image and post-process for cross-stitch patterns.")
    parser.add_argument("--prompt", type=str, help="Text prompt for generation", default=None)
    parser.add_argument("--prompt_file", type=str, help="Path to a text file with prompts (one per line). If set, --prompt is ignored.", default=None)
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5", help="Hugging Face model id")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token if required (or set HF_TOKEN env var).")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--num_images", type=int, default=1, help="Number of generations")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--pattern_size", type=int, default=40, help="Width/height in stitches for the pattern grid (e.g., 40)")
    parser.add_argument("--n_colors", type=int, default=12, help="Number of colors to reduce to for the pattern")
    parser.add_argument("--device", type=str, default=None, help='"cuda" or "cpu" (auto-detect if omitted)')
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--preview_scale", type=int, default=16, help="Scale factor for preview image (e.g., 16)")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] loading model {args.model_id} on device {args.device or 'auto'}...")
    pipe = load_pipeline(model_id=args.model_id, device=args.device, use_auth_token=hf_token)

    prompts = []
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        if not prompts:
            raise SystemExit("No prompts found in prompt_file.")
    else:
        if not args.prompt:
            raise SystemExit("Please provide --prompt or --prompt_file")
        prompts = [args.prompt]

    total_generated = 0
    for p in prompts:
        print(f"[info] generating for prompt: {p!r}")
        images, used_seed = generate_images(pipe,
                                           prompt=p,
                                           num_images=args.num_images,
                                           height=args.height,
                                           width=args.width,
                                           steps=args.steps,
                                           guidance_scale=args.guidance_scale,
                                           seed=args.seed)
        for i, img in enumerate(images):
            timestamp = int(time.time())
            fname_base = f"sd_{timestamp}_seed{used_seed}_p{i}"
            raw_path = out_dir / f"{fname_base}_raw.png"
            reduced_path = out_dir / f"{fname_base}_reduced_{args.n_colors}colors.png"
            small_path = out_dir / f"{fname_base}_pattern_{args.pattern_size}x{args.pattern_size}.png"
            preview_path = out_dir / f"{fname_base}_pattern_preview.png"

            print(f"  - saving raw image to {raw_path}")
            img.save(raw_path)

            print(f"  - reducing to {args.n_colors} colors...")
            reduced = reduce_colors_adaptive(img, n_colors=args.n_colors)
            reduced.save(reduced_path)

            print(f"  - resizing to pattern grid {args.pattern_size}x{args.pattern_size} (nearest neighbor)")
            small = resize_to_pattern_grid(reduced, pattern_size=args.pattern_size)
            small.save(small_path)

            print(f"  - saving preview (scale={args.preview_scale}) to {preview_path}")
            save_scaled_preview(small, scale=args.preview_scale, out_path=preview_path)

            total_generated += 1

    print(f"[done] generated {total_generated} images. Outputs in {out_dir.absolute()}")


if __name__ == "__main__":
    main()
