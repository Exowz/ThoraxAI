"""Data helpers: results loading and sample image discovery."""

import json
from pathlib import Path

SAMPLES_DIR = Path("samples")


def load_results():
    try:
        with open("outputs/results.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def samples_available() -> bool:
    return (
        (SAMPLES_DIR / "NORMAL").is_dir()
        and (SAMPLES_DIR / "PNEUMONIA").is_dir()
        and any((SAMPLES_DIR / "NORMAL").iterdir())
        and any((SAMPLES_DIR / "PNEUMONIA").iterdir())
    )


def list_samples(cls: str) -> list[Path]:
    d = SAMPLES_DIR / cls
    return sorted(
        [p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda p: p.name,
    )
