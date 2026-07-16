"""Download model checkpoints from Hugging Face Hub into models/.

This downloads files to local disk. It does NOT call hosted inference APIs.

Usage examples:
    python scripts/download_hf_checkpoints.py \
      --light-repo your-org/your-efficientnet-repo --light-file efficientnet_b4_finetuned.pth \
      --heavy-repo your-org/your-vit-repo --heavy-file vit_b16_finetuned.pth

Optional:
    --token <HF_TOKEN>

If no token is passed, the script uses environment variable HF_TOKEN when needed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


def _download(repo_id: str, filename: str, out_path: Path, token: str | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    local_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        local_dir=str(out_path.parent),
        local_dir_use_symlinks=False,
    )

    src = Path(local_file)
    if src.resolve() != out_path.resolve():
        if out_path.exists():
            out_path.unlink()
        src.replace(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download fine-tuned checkpoints from HF Hub")
    parser.add_argument("--light-repo", required=True, help="HF repo id for light model")
    parser.add_argument("--light-file", required=True, help="Checkpoint filename in light repo")
    parser.add_argument("--heavy-repo", required=True, help="HF repo id for heavy model")
    parser.add_argument("--heavy-file", required=True, help="Checkpoint filename in heavy repo")
    parser.add_argument("--token", default=None, help="HF access token (optional)")

    args = parser.parse_args()
    token = args.token or os.environ.get("HF_TOKEN")

    light_out = Path("models") / "efficientnet_b4_finetuned.pth"
    heavy_out = Path("models") / "vit_b16_finetuned.pth"

    print(f"[Download] light: {args.light_repo}/{args.light_file} -> {light_out}")
    _download(args.light_repo, args.light_file, light_out, token)

    print(f"[Download] heavy: {args.heavy_repo}/{args.heavy_file} -> {heavy_out}")
    _download(args.heavy_repo, args.heavy_file, heavy_out, token)

    print("[Download] complete")


if __name__ == "__main__":
    main()
