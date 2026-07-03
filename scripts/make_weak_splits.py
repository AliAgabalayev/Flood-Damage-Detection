import argparse
import csv
import random
from pathlib import Path


def discover_pairs(s1_dir: Path, label_dir: Path, image_suffix: str, label_suffix: str) -> tuple[list[tuple[str, str]], int]:
    pairs: list[tuple[str, str]] = []
    missing = 0
    for f in sorted(s1_dir.glob(f"*{image_suffix}.tif")):
        base = f.stem[: f.stem.rfind(image_suffix)]
        label = label_dir / f"{base}{label_suffix}.tif"
        if label.exists():
            pairs.append((f.name, label.name))
        else:
            missing += 1
    return pairs, missing


def write_split(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/sen1floods11_weak")
    ap.add_argument("--s1-subdir", default="S1Weak")
    ap.add_argument("--label-subdir", default="S2IndexLabelWeak")
    ap.add_argument("--out", default="data/splits/weak")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    pairs, missing = discover_pairs(
        root / args.s1_subdir, root / args.label_subdir,
        image_suffix="_S1Weak", label_suffix="_S2IndexLabelWeak",
    )
    print(f"matched pairs: {len(pairs)}, missing labels: {missing}")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_val = int(len(pairs) * args.val_frac)
    val, train = pairs[:n_val], pairs[n_val:]

    out = Path(args.out)
    write_split(out / "train.csv", train)
    write_split(out / "val.csv", val)
    print(f"train: {len(train)}, val: {len(val)} -> {out}")


if __name__ == "__main__":
    main()
