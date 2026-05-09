from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(images_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    out.sort()
    return out


def ensure_empty_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_pairs(imgs: list[Path], src_labels: Path, dst_images: Path, dst_labels: Path) -> None:
    ensure_empty_dir(dst_images)
    ensure_empty_dir(dst_labels)
    for img in imgs:
        shutil.copy2(img, dst_images / img.name)
        lbl = src_labels / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, dst_labels / lbl.name)
        else:
            (dst_labels / (img.stem + ".txt")).write_text("", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Split YOLO-seg dataset into train/test folders (copy files).")
    ap.add_argument("--src", type=Path, required=True, help="Source root, containing images/ and labels/")
    ap.add_argument("--out", type=Path, required=True, help="Output root (will create train/ and test/)")
    ap.add_argument("--test-ratio", type=float, default=0.05, help="Test split ratio (default: 0.05)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    src = args.src
    images_dir = src / "images"
    labels_dir = src / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"Expected {images_dir} and {labels_dir} to exist.")

    imgs = list_images(images_dir)
    if not imgs:
        raise SystemExit(f"No images found in {images_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(imgs)
    n_test = max(1, int(round(len(imgs) * float(args.test_ratio))))
    test_imgs = imgs[:n_test]
    train_imgs = imgs[n_test:]
    if not train_imgs:
        train_imgs, test_imgs = imgs[1:], imgs[:1]

    out = args.out
    copy_pairs(train_imgs, labels_dir, out / "train" / "images", out / "train" / "labels")
    copy_pairs(test_imgs, labels_dir, out / "test" / "images", out / "test" / "labels")

    (out / "split.txt").write_text(
        f"src={src}\ncount={len(imgs)}\ntrain={len(train_imgs)}\ntest={len(test_imgs)}\nseed={args.seed}\n",
        encoding="utf-8",
    )
    print(f"Done. train={len(train_imgs)} test={len(test_imgs)} -> {out}")


if __name__ == "__main__":
    main()

