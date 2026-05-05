"""A2: generate SVD source videos only.

Split out from `a2_generate_videos.py` so you can run generators one-by-one.
SVD is img2vid, so this script first generates input images via SD v1.5.

Outputs (relative to --out-root):
  data/source/svd/real_svd_001.mp4
  data/source_manifest.jsonl   (merged/upserted by id)

Example:
  pip install -U diffusers transformers accelerate safetensors imageio imageio-ffmpeg pillow
  python infra/scripts/a2_generate_svd.py --zip
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def configure_hf(mirror: str | None = None) -> None:
    if mirror and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = mirror
        print(f"[hf] HF_ENDPOINT set to: {mirror}")


def _is_local_model_dir(model_id_or_path: str) -> bool:
    try:
        return Path(model_id_or_path).expanduser().is_dir()
    except Exception:
        return False


def _maybe_local_files_only(model_id_or_path: str) -> bool:
    local_only = _is_local_model_dir(model_id_or_path)
    if local_only:
        print(f"[hf] Detected local model dir; forcing local_files_only=True: {model_id_or_path}")
    return local_only


def _default_svd_model_dir() -> str:
    # Cloud convention in this repo: /rivermind-data/hf_models/<org>__<repo>
    return str(Path("/rivermind-data") / "hf_models" / "stabilityai__stable-video-diffusion-img2vid-xt")


def default_out_root() -> Path:
    return Path.cwd()


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def upsert_manifest(manifest_path: Path, new_records: list[dict[str, Any]]) -> None:
    existing = _read_jsonl(manifest_path)
    by_id: dict[str, dict[str, Any]] = {r["id"]: r for r in existing if "id" in r}
    for r in new_records:
        by_id[r["id"]] = r
    merged = [by_id[k] for k in sorted(by_id.keys())]
    _write_jsonl(manifest_path, merged)


def _next_index_from_manifest(manifest_path: Path, id_prefix: str) -> int:
    max_idx = 0
    for r in _read_jsonl(manifest_path):
        rid = r.get("id")
        if not isinstance(rid, str):
            continue
        if not rid.startswith(id_prefix):
            continue
        suffix = rid[len(id_prefix) :]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return max_idx + 1


def make_prompts(n: int, seed: int) -> list[str]:
    random.seed(seed)

    # SVD is img2vid; prompts here bias toward single-subject scenes with clear motion.
    scenes = [
        "a quiet beach at sunrise",
        "a foggy forest trail",
        "a snowy street at night",
        "a greenhouse with plants",
        "a riverside walkway",
        "a parking lot after rain",
        "a small bookstore",
        "a museum hallway",
        "a subway platform",
        "a rooftop terrace",
    ]
    actions = [
        "slowly turns toward the camera",
        "walks forward and then stops",
        "waves once and smiles",
        "picks up an object and puts it down",
        "looks left, then right, then back",
        "takes two steps and points at a sign",
    ]
    subjects = [
        "a cyclist wearing a helmet",
        "a barista in an apron",
        "a musician holding a guitar",
        "a tourist with a camera",
        "a person wearing a yellow raincoat",
        "a golden retriever",
    ]
    styles = [
        "photorealistic, tripod shot, natural colors",
        "cinematic, shallow depth of field",
        "high detail, soft lighting",
    ]

    prompts: list[str] = []
    while len(prompts) < n:
        s = random.choice(subjects)
        scene = random.choice(scenes)
        act = random.choice(actions)
        st = random.choice(styles)
        p = f"{s} in {scene} {act}. {st}."
        if p not in prompts:
            prompts.append(p)
    return prompts


def _try_enable_low_vram(pipe: Any) -> None:
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pass
    try:
        pipe.enable_sequential_cpu_offload()
    except Exception:
        pass
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae.enable_slicing()
    except Exception:
        pass
    try:
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae.enable_tiling()
    except Exception:
        pass


def _maybe_empty_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _cleanup(obj: Any) -> None:
    try:
        del obj
    except Exception:
        pass
    gc.collect()
    _maybe_empty_cuda()


def _pipe_call(pipe: Any, **kwargs: Any) -> Any:
    try:
        import torch

        with torch.inference_mode():
            return pipe(**kwargs)
    except Exception:
        return pipe(**kwargs)


@dataclass
class GenSpec:
    out_dir: Path
    id_prefix: str


def generate_t2i_images(out_dir: Path, prompts: list[str], n: int, seed: int, steps: int) -> list[Path]:
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except Exception as e:
        raise RuntimeError(f"StableDiffusionPipeline unavailable: {e!r}")

    model_id = os.environ.get("SVD_T2I_MODEL", "runwayml/stable-diffusion-v1-5")
    local_only = _maybe_local_files_only(model_id)
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, local_files_only=local_only)
    except TypeError:
        if local_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            print("[hf] diffusers does not support local_files_only; set HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    _try_enable_low_vram(pipe)

    images: list[Path] = []
    for i in range(n):
        img_id = f"svd_input_{i+1:03d}"
        img_path = out_dir / f"{img_id}.png"

        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        g = torch.Generator(device=gen_device).manual_seed(seed + 10_000 + i)

        img = _pipe_call(
            pipe,
            prompt=prompts[i],
            num_inference_steps=int(steps),
            guidance_scale=7.0,
            generator=g,
        ).images[0]
        img.save(img_path)
        images.append(img_path)
        print(f"[t2i] done {img_id}")

    _cleanup(pipe)
    return images


def generate_svd(
    spec: GenSpec,
    prompts: list[str],
    n: int,
    fps: int,
    num_frames: int,
    seed: int,
    tmp_dir: Path,
    t2i_steps: int,
    svd_steps: int | None,
    start_index: int,
) -> list[dict[str, Any]]:
    try:
        import torch
        from PIL import Image
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import export_to_video
    except Exception as e:
        print(f"[svd] unavailable: {e!r}")
        return []

    model_id = os.environ.get(
        "SVD_MODEL",
        _default_svd_model_dir(),
    )
    local_only = _maybe_local_files_only(model_id)
    try:
        pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, local_files_only=local_only)
    except TypeError:
        if local_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            print("[hf] diffusers does not support local_files_only; set HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
        pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    _try_enable_low_vram(pipe)

    img_dir = tmp_dir / "svd_input_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    images = generate_t2i_images(img_dir, prompts, n=n, seed=seed, steps=t2i_steps)

    records: list[dict[str, Any]] = []
    for i, img_path in enumerate(images):
        vid_id = f"{spec.id_prefix}{start_index + i:03d}"
        mp4_path = spec.out_dir / f"{vid_id}.mp4"

        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        g = torch.Generator(device=gen_device).manual_seed(seed + 20_000 + i)

        image = Image.open(img_path).convert("RGB")
        call_kwargs: dict[str, Any] = {"image": image, "num_frames": int(num_frames), "generator": g}
        if svd_steps is not None:
            call_kwargs["num_inference_steps"] = int(svd_steps)
        try:
            out = _pipe_call(pipe, **call_kwargs)
        except TypeError:
            call_kwargs.pop("num_inference_steps", None)
            out = _pipe_call(pipe, **call_kwargs)

        video = out.frames[0] if hasattr(out, "frames") and out.frames is not None else out.videos[0]
        export_to_video(video, str(mp4_path), fps=fps)

        records.append(
            {
                "id": vid_id,
                "video_path": str(mp4_path),
                "shotir_path": None,
                "generator": "svd-img2vid-xt",
                "prompt": prompts[i],
                "meta": {
                    "fps": fps,
                    "num_frames": num_frames,
                    "input_image": str(img_path),
                    "t2i_steps": int(t2i_steps),
                    "svd_steps": int(svd_steps) if svd_steps is not None else None,
                },
            }
        )
        print(f"[svd] done {vid_id}")

    _cleanup(pipe)
    return records


def main() -> None:
    # Only configure HF mirror when we might need to download remote models.
    chosen_svd_model_id = os.environ.get("SVD_MODEL", _default_svd_model_dir())
    chosen_t2i_model_id = os.environ.get("SVD_T2I_MODEL", "runwayml/stable-diffusion-v1-5")
    if not (_is_local_model_dir(chosen_svd_model_id) and _is_local_model_dir(chosen_t2i_model_id)):
        configure_hf(mirror="https://hf-mirror.com")

    parser = argparse.ArgumentParser(description="A2: generate SVD videos only")
    parser.add_argument("--out-root", default=str(default_out_root()))
    parser.add_argument("--duration-s", type=int, default=6)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260403)
    parser.add_argument("--n", type=int, default=15, help="How many SVD videos to generate")
    parser.add_argument("--t2i-steps", type=int, default=25)
    parser.add_argument("--svd-steps", type=int, default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--keep-tmp", action="store_true")

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[a2_generate_svd] Ignoring unknown args: {unknown}")

    try:
        import PIL  # type: ignore
        import PIL._typing as pil_typing  # type: ignore

        if not hasattr(pil_typing, "_Ink"):
            raise RuntimeError("Pillow is too old for diffusers pipelines.")
        print(f"[deps] Pillow {getattr(PIL, '__version__', 'unknown')} OK")
    except Exception as e:
        print("[deps] Dependency check failed:")
        print("  ", repr(e))
        print("[deps] Fix:")
        print("  pip install -U --force-reinstall 'pillow>=11' 'diffusers>=0.30' transformers accelerate safetensors")
        return

    root = Path(args.out_root).resolve()
    data_root = root / "data"
    source_root = data_root / "source"
    tmp_dir = root / ".a2_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    svd_dir = source_root / "svd"
    svd_dir.mkdir(parents=True, exist_ok=True)

    # Keep the manifest alongside the generated videos to avoid cross-generator overwrites.
    manifest_path = svd_dir / "source_manifest.jsonl"

    n = int(args.n)
    num_frames = int(args.duration_s * args.fps + 1)
    prompts = make_prompts(n, seed=int(args.seed))

    records = generate_svd(
        GenSpec(svd_dir, "real_svd_"),
        prompts,
        n=n,
        fps=int(args.fps),
        num_frames=num_frames,
        seed=int(args.seed),
        tmp_dir=tmp_dir,
        t2i_steps=int(args.t2i_steps),
        svd_steps=args.svd_steps,
        start_index=_next_index_from_manifest(manifest_path, "real_svd_"),
    )

    for r in records:
        vp = Path(r["video_path"]).resolve()
        r["video_path"] = rel_posix(vp, root)
        r["shotir_path"] = None
        # Keep input_image path portable too.
        meta = r.get("meta")
        if isinstance(meta, dict) and meta.get("input_image"):
            meta["input_image"] = rel_posix(Path(meta["input_image"]).resolve(), root)

    upsert_manifest(manifest_path, records)
    print(f"Wrote/updated manifest: {manifest_path}")
    print(f"Generated: {len(records)}")

    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if args.zip:
        zip_base = str(root / "videogendoctor_a2_bundle")
        zip_path = shutil.make_archive(zip_base, "zip", root_dir=str(root), base_dir="data")
        print(f"Zip written: {zip_path}")


if __name__ == "__main__":
    main()
