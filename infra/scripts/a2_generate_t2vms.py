"""A2: generate Text2Video-MS source videos only.

Split out from `a2_generate_videos.py` so you can run generators one-by-one.

Outputs (relative to --out-root):
  data/source/t2vms/real_t2v_001.mp4
  data/source_manifest.jsonl   (merged/upserted by id)

Example:
  pip install -U diffusers transformers accelerate safetensors imageio imageio-ffmpeg pillow
  python infra/scripts/a2_generate_t2vms.py --zip
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


def _default_t2vms_model_dir() -> str:
    # Cloud convention in this repo: /rivermind-data/hf_models/<org>__<repo>
    return str(Path("/rivermind-data") / "hf_models" / "damo-vilab__text-to-video-ms-1.7b")


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

    # Text2Video-MS: use a distinct prompt bank to avoid overlapping with other scripts.
    scenes = [
        "a miniature diorama city",
        "a cozy cabin interior",
        "an underwater coral reef",
        "a space station corridor",
        "a colorful street market",
        "a desert with dunes",
        "a neon arcade",
        "a library with floating books",
        "a studio with a spotlight",
        "a mountaintop viewpoint",
    ]
    actions = [
        "camera slowly pans left",
        "camera gently dollies forward",
        "the subject spins once",
        "the subject jumps and lands",
        "lights flicker briefly",
        "a gust of wind moves small objects",
    ]
    subjects = [
        "a small robot",
        "a paper airplane",
        "a plush toy",
        "a cartoon cat",
        "a clay character",
        "a tiny astronaut",
    ]
    styles = [
        "stylized animation, smooth motion",
        "stop-motion look, high detail",
        "vibrant colors, clean edges",
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


def generate_t2vms(
    spec: GenSpec,
    prompts: list[str],
    n: int,
    fps: int,
    num_frames: int,
    seed: int,
    num_inference_steps: int,
    start_index: int,
) -> list[dict[str, Any]]:
    try:
        import torch
        from diffusers import TextToVideoSDPipeline
        from diffusers.utils import export_to_video
    except Exception as e:
        print(f"[t2vms] unavailable: {e!r}")
        return []

    model_id = os.environ.get(
        "T2VMS_MODEL",
        _default_t2vms_model_dir(),
    )
    local_only = _maybe_local_files_only(model_id)
    try:
        pipe = TextToVideoSDPipeline.from_pretrained(model_id, torch_dtype=torch.float16, local_files_only=local_only)
    except TypeError:
        if local_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            print("[hf] diffusers does not support local_files_only; set HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
        pipe = TextToVideoSDPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    _try_enable_low_vram(pipe)

    records: list[dict[str, Any]] = []
    for i in range(n):
        vid_id = f"{spec.id_prefix}{start_index + i:03d}"
        prompt = prompts[i]
        mp4_path = spec.out_dir / f"{vid_id}.mp4"

        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        g = torch.Generator(device=gen_device).manual_seed(seed + 30_000 + i)

        out = _pipe_call(
            pipe,
            prompt=prompt,
            num_frames=int(num_frames),
            generator=g,
            num_inference_steps=int(num_inference_steps),
        )

        video = out.frames[0] if hasattr(out, "frames") and out.frames is not None else out.videos[0]
        export_to_video(video, str(mp4_path), fps=fps)

        records.append(
            {
                "id": vid_id,
                "video_path": str(mp4_path),
                "shotir_path": None,
                "generator": "text-to-video-ms-1.7b",
                "prompt": prompt,
                "meta": {"fps": fps, "num_frames": num_frames, "num_inference_steps": int(num_inference_steps)},
            }
        )
        print(f"[t2vms] done {vid_id}")

    _cleanup(pipe)
    return records


def main() -> None:
    # Only configure HF mirror when we might need to download remote models.
    chosen_model_id = os.environ.get("T2VMS_MODEL", _default_t2vms_model_dir())
    if not _is_local_model_dir(chosen_model_id):
        configure_hf(mirror="https://hf-mirror.com")

    parser = argparse.ArgumentParser(description="A2: generate Text2Video-MS videos only")
    parser.add_argument("--out-root", default=str(default_out_root()))
    parser.add_argument("--duration-s", type=int, default=6)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260403)
    parser.add_argument("--n", type=int, default=15, help="How many T2VMS videos to generate")
    parser.add_argument("--steps", type=int, default=25, help="num_inference_steps")
    parser.add_argument("--zip", action="store_true")

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[a2_generate_t2vms] Ignoring unknown args: {unknown}")

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

    out_dir = source_root / "t2vms"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep the manifest alongside the generated videos to avoid cross-generator overwrites.
    manifest_path = out_dir / "source_manifest.jsonl"

    n = int(args.n)
    num_frames = int(args.duration_s * args.fps + 1)
    prompts = make_prompts(n, seed=int(args.seed))

    records = generate_t2vms(
        GenSpec(out_dir, "real_t2v_"),
        prompts,
        n=n,
        fps=int(args.fps),
        num_frames=num_frames,
        seed=int(args.seed),
        num_inference_steps=int(args.steps),
        start_index=_next_index_from_manifest(manifest_path, "real_t2v_"),
    )

    for r in records:
        vp = Path(r["video_path"]).resolve()
        r["video_path"] = rel_posix(vp, root)
        r["shotir_path"] = None

    upsert_manifest(manifest_path, records)
    print(f"Wrote/updated manifest: {manifest_path}")
    print(f"Generated: {len(records)}")

    if args.zip:
        zip_base = str(root / "videogendoctor_a2_bundle")
        zip_path = shutil.make_archive(zip_base, "zip", root_dir=str(root), base_dir="data")
        print(f"Zip written: {zip_path}")


if __name__ == "__main__":
    main()
