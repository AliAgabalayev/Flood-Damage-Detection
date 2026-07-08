from __future__ import annotations

import csv
from pathlib import Path

COUNTRIES = [
    "Ghana", "India", "Mekong", "Nigeria", "Pakistan",
    "Paraguay", "Somalia", "Spain", "Sri-Lanka", "USA",
]


def read_rows(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [tuple(row) for row in csv.reader(f) if row]


def write_rows(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def country_of(row: tuple[str, str]) -> str:
    return row[0].split("_")[0]


def main() -> None:
    official_train = read_rows(Path("data/splits/official/train.csv"))
    official_val = read_rows(Path("data/splits/official/val.csv"))
    official_test = read_rows(Path("data/splits/official/test.csv"))
    weak_train = read_rows(Path("data/splits/weak/train.csv"))
    weak_val = read_rows(Path("data/splits/weak/val.csv"))

    all_official = official_train + official_val + official_test

    for country in COUNTRIES:
        out = Path(f"data/splits/loco/{country}")

        write_rows(out / "train9/train.csv", [r for r in official_train if country_of(r) != country])
        write_rows(out / "train9/val.csv", [r for r in official_val if country_of(r) != country])
        write_rows(out / "weak9/train.csv", [r for r in weak_train if country_of(r) != country])
        write_rows(out / "weak9/val.csv", [r for r in weak_val if country_of(r) != country])
        write_rows(out / "heldout/test.csv", [r for r in all_official if country_of(r) == country])

        print(
            f"{country:10s}  train9={sum(1 for r in official_train if country_of(r) != country)}"
            f"  weak9={sum(1 for r in weak_train if country_of(r) != country)}"
            f"  heldout={sum(1 for r in all_official if country_of(r) == country)}"
        )


if __name__ == "__main__":
    main()
