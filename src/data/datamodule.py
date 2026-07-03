from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.dataset import Sen1FloodDataset
from data.preprocessing import default_preprocessor
from data.split_loader import load_split
from data.transforms import build_train_transforms, build_val_transforms
from utils.config import Config

logger = logging.getLogger(__name__)


def discover_image_files(
    image_dir: Path | str,
    *,
    extensions: Sequence[str] = (".tif", ".tiff"),
) -> List[Path]:
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {image_dir}\n"
            f"Hint: check config.data.root and config.data.s1_subdir."
        )
    if not image_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a directory but got a file: {image_dir}"
        )

    exts_lower = {e.lower() for e in extensions}
    found: List[Path] = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts_lower
    )

    if not found:
        raise ValueError(
            f"No TIFF files found in {image_dir}.\n"
            f"Searched extensions: {sorted(exts_lower)}"
        )

    logger.debug("discover_image_files: found %d files in %s", len(found), image_dir)
    return found


def match_image_and_label(
    image_paths: Sequence[Path | str],
    label_dir: Path | str,
    *,
    image_stem_suffix: str = "_S1Hand",
    label_stem_suffix: str = "_LabelHand",
    extensions: Sequence[str] = (".tif", ".tiff"),
) -> Tuple[List[Path], List[Path]]:
    """
    1. Strip ``image_stem_suffix`` from the image stem to get the shared
       scene ID (e.g. ``Ghana_103272_S1Hand`` → ``Ghana_103272``).
    2. Construct ``{base}{label_stem_suffix}{ext}`` and check each extension
       in order until a match is found on disk.
    3. Collect all failures and raise a single informative error
    """
    label_dir = Path(label_dir)

    if not label_dir.exists():
        raise FileNotFoundError(
            f"Label directory does not exist: {label_dir}\n"
            f"Hint: check config.data.root and config.data.label_subdir."
        )
    if not label_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a directory but got a file: {label_dir}"
        )

    aligned_images: List[Path] = []
    aligned_labels: List[Path] = []
    missing: List[str] = []

    for raw_path in image_paths:
        img_path = Path(raw_path)
        stem = img_path.stem

        if image_stem_suffix not in stem:
            base = stem
            logger.warning(
                "Image stem %r does not contain suffix %r; using full stem as scene ID.",
                stem, image_stem_suffix,
            )
        else:
            base = stem[: stem.rfind(image_stem_suffix)]

        label_stem = f"{base}{label_stem_suffix}"
        label_path: Optional[Path] = None
        for ext in extensions:
            candidate = label_dir / f"{label_stem}{ext}"
            if candidate.exists():
                label_path = candidate
                break

        if label_path is None:
            tried = [str(label_dir / f"{label_stem}{e}") for e in extensions]
            missing.append(
                f"  Image: {img_path}\n"
                f"  Expected label (tried): {', '.join(tried)}"
            )
            continue

        aligned_images.append(img_path)
        aligned_labels.append(label_path)

    if missing:
        raise FileNotFoundError(
            f"Could not find label files for {len(missing)} image(s):\n"
            + "\n".join(missing)
        )

    logger.debug(
        "match_image_and_label: matched %d pairs (label_dir=%s)",
        len(aligned_images), label_dir,
    )
    return aligned_images, aligned_labels


def discover_pairs(config: object) -> Tuple[List[Path], List[Path]]:
    data_cfg = config.data 
    image_dir = Path(data_cfg.root) / data_cfg.s1_subdir
    label_dir = Path(data_cfg.root) / data_cfg.label_subdir
    image_paths = discover_image_files(image_dir)
    return match_image_and_label(image_paths, label_dir)



class FloodDataModule(pl.LightningDataModule):

    def __init__(
        self,
        config: Config,
        train_transforms: Optional[object] = None,
        val_transforms: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.config: Config = config

        self._train_transforms = train_transforms
        if self._train_transforms is None and config.data.augment:
            self._train_transforms = build_train_transforms(config.data.image_size)
        self._val_transforms = (
            val_transforms if val_transforms is not None
            else build_val_transforms()
        )

        # Preprocessor is shared (and read-only) across all splits.
        self._preprocessor = default_preprocessor(config)

        self._split_dir: Path = Path(config.data.split_dir)
        self._image_root: Path = Path(config.data.root) / config.data.s1_subdir
        self._label_root: Path = Path(config.data.root) / config.data.label_subdir

        self.train_dataset: Optional[Sen1FloodDataset] = None
        self.val_dataset:   Optional[Sen1FloodDataset] = None
        self.test_dataset:  Optional[Sen1FloodDataset] = None


    def setup(self, stage: Optional[str] = None) -> None:
        fit_stages  = {None, "fit", "validate"}
        test_stages = {None, "test", "predict"}

        if stage in fit_stages:
            self.train_dataset = self._build_dataset("train", self._train_transforms)
            self.val_dataset   = self._build_dataset("val",   self._val_transforms)

        if stage in test_stages:
            self.test_dataset  = self._build_dataset("test",  self._val_transforms)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError(
                "train_dataset is not initialised. Call setup() before train_dataloader()."
            )
        return self._make_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError(
                "val_dataset is not initialised. Call setup() before val_dataloader()."
            )
        return self._make_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError(
                "test_dataset is not initialised. Call setup() before test_dataloader()."
            )
        return self._make_dataloader(self.test_dataset, shuffle=False)

    
    def _build_dataset(
        self,
        split_name: str,
        transforms: Optional[object],
    ) -> Sen1FloodDataset:
        record = load_split(
            split_name=split_name,
            split_dir=self._split_dir,
            image_root=self._image_root,
            label_root=self._label_root,
        )
        return Sen1FloodDataset(
            image_paths=record.image_paths,
            label_paths=record.label_paths,
            config=self.config,
            transforms=transforms,
            preprocessor=self._preprocessor,
        )

    def _make_dataloader(
        self,
        dataset: Sen1FloodDataset,
        *,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=shuffle,
            drop_last=shuffle,
            pin_memory=True,
            persistent_workers=self.config.data.num_workers > 0,
            multiprocessing_context="spawn" if self.config.data.num_workers > 0 else None,
        )

   
    @staticmethod
    def _load_split_file(csv_path: Path) -> Tuple[List[str], List[str]]:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {csv_path}\n"
                f"Hint: check config.data.split_dir → {csv_path.parent}"
            )

        image_files: List[str] = []
        label_files: List[str] = []

        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for line_no, row in enumerate(reader, start=1):
                if not row or all(cell.strip() == "" for cell in row):
                    continue
                if len(row) < 2:
                    raise ValueError(
                        f"{csv_path}:{line_no}: expected 2 columns "
                        f"(image, label) but got {len(row)}: {row!r}"
                    )
                image_files.append(row[0].strip())
                label_files.append(row[1].strip())

        if not image_files:
            raise ValueError(
                f"Split CSV contains no data rows: {csv_path}"
            )

        return image_files, label_files

    @staticmethod
    def _build_sample_paths(
        image_files: List[str],
        label_files: List[str],
        image_root: Path,
        label_root: Path,
    ) -> Tuple[List[Path], List[Path]]:
        image_paths = [image_root / fn for fn in image_files]
        label_paths = [label_root / fn for fn in label_files]
        return image_paths, label_paths

    @staticmethod
    def _validate_paths(
        image_paths: List[Path],
        label_paths: List[Path],
    ) -> None:
        missing: List[str] = []
        for img, lbl in zip(image_paths, label_paths):
            if not img.exists():
                missing.append(f"  [image] {img}")
            if not lbl.exists():
                missing.append(f"  [label] {lbl}")

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} file(s) not found on disk:\n"
                + "\n".join(missing)
            )
