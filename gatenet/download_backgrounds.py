from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def _download(url: str, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "GateNet-background-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def main() -> None:
    ap = argparse.ArgumentParser(description="Download unrelated background images for MonoRace-style synthesis.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--source", choices=("picsum",), default="picsum")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    def one(i: int) -> dict:
        # picsum.photos returns arbitrary non-gate natural/urban photos. The
        # seed makes the URL stable enough for a repeatable local background set.
        url = f"https://picsum.photos/seed/gatenet-{int(args.seed)}-{i}/{int(args.width)}/{int(args.height)}"
        out_path = args.out / f"bg_{i:05d}.jpg"
        if out_path.exists():
            return {"file": out_path.name, "url": url, "status": "exists"}
        try:
            data = _download(url)
            tmp = out_path.with_suffix(".tmp")
            tmp.write_bytes(data)
            # Verify PIL can decode it, then re-save as a normal JPEG.
            img = Image.open(tmp).convert("RGB")
            img.save(out_path, quality=92, subsampling=1)
            tmp.unlink(missing_ok=True)
            if float(args.sleep) > 0:
                time.sleep(float(args.sleep))
            return {"file": out_path.name, "url": url, "status": "ok"}
        except Exception as exc:
            return {"file": out_path.name, "url": url, "status": "error", "error": repr(exc)}

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(one, i) for i in range(int(args.count))]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="downloading backgrounds"):
            rows.append(fut.result())

    rows.sort(key=lambda row: row["file"])
    ok = sum(1 for row in rows if row["status"] in {"ok", "exists"})

    meta = {
        "source": args.source,
        "count_requested": int(args.count),
        "count_downloaded": ok,
        "width": int(args.width),
        "height": int(args.height),
        "items": rows,
    }
    (args.out / "backgrounds_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "items"}, indent=2))


if __name__ == "__main__":
    main()
