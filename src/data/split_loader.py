from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)


SPLIT_FILENAME_MAP: Dict[str, str] = {
    "train":      "train.csv",
    "val":        "val.csv",
    "validation": "val.csv",  
    "test":       "test.csv",
}


class SplitRecord(NamedTuple):    

    image_paths: List[Path]
    label_paths: List[Path]
    name: str
    csv_path: Path


def _parse_split_csv(
    csv_path: Path,
    image_root: Path,
    label_root: Path,
) -> Tuple[List[Path], List[Path]]:
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Split CSV not found: {csv_path}\n"
            f"Hint: check config.data.split_dir — current value points to "
            f"{csv_path.parent}."
        )

    image_paths: List[Path] = []
    label_paths: List[Path] = []

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for line_no, row in enumerate(reader, start=1):
            # Skip blank lines
            if not row or all(cell.strip() == "" for cell in row):
                continue
            if len(row) < 2:
                raise ValueError(
                    f"{csv_path}:{line_no}: expected 2 columns "
                    f"(image_filename, label_filename) but got {len(row)}: {row!r}"
                )
            image_paths.append(image_root / row[0].strip())
            label_paths.append(label_root / row[1].strip())

    if not image_paths:
        raise ValueError(
            f"Split CSV contains no data rows: {csv_path}\n"
            f"Verify the file is not empty and uses the expected two-column format."
        )

    logger.debug(
        "_parse_split_csv: %d pairs from %s", len(image_paths), csv_path
    )
    return image_paths, label_paths


def load_split(
    split_name: str,
    split_dir: Path | str,
    image_root: Path | str,
    label_root: Path | str,
    *,
    filename_map: Optional[Dict[str, str]] = None,
) -> SplitRecord:

    fmap = filename_map if filename_map is not None else SPLIT_FILENAME_MAP
    split_name_lower = split_name.lower()

    if split_name_lower not in fmap:
        known = ", ".join(f'"{k}"' for k in sorted(fmap))
        raise KeyError(
            f"Unknown split name {split_name!r}. "
            f"Registered names: {known}. "
            f"To add a new split, extend the filename_map argument or "
            f"SPLIT_FILENAME_MAP in data/split_loader.py."
        )

    csv_filename = fmap[split_name_lower]
    csv_path = Path(split_dir) / csv_filename

    image_paths, label_paths = _parse_split_csv(
        csv_path,
        image_root=Path(image_root),
        label_root=Path(label_root),
    )

    logger.info(
        "load_split(%r): %d samples (csv=%s)",
        split_name,
        len(image_paths),
        csv_path,
    )

    return SplitRecord(
        image_paths=image_paths,
        label_paths=label_paths,
        name=split_name_lower,
        csv_path=csv_path,
    )

def load_split_from_config(
    split_name: str,
    config: object,
    *,
    filename_map: Optional[Dict[str, str]] = None,
) -> SplitRecord:
    

    data_cfg = config.data  
    return load_split(
        split_name=split_name,
        split_dir=data_cfg.split_dir,
        image_root=Path(data_cfg.root) / data_cfg.s1_subdir,
        label_root=Path(data_cfg.root) / data_cfg.label_subdir,
        filename_map=filename_map,
    )


def load_all_splits(
    config: object,
    *,
    splits: Tuple[str, ...] = ("train", "val", "test"),
    filename_map: Optional[Dict[str, str]] = None,
) -> Dict[str, SplitRecord]:
   
    result: Dict[str, SplitRecord] = {}
    for name in splits:
        record = load_split_from_config(name, config, filename_map=filename_map)
        result[record.name] = record

    logger.info(
        "load_all_splits: loaded %d splits (%s)",
        len(result),
        ", ".join(f"{k}={len(v.image_paths)}" for k, v in result.items()),
    )
    return result
