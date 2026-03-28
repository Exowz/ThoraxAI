"""Data helpers: results loading and sample image discovery."""

import json
from pathlib import Path

import streamlit as st
from huggingface_hub import snapshot_download

SAMPLES_DIR = Path("samples")
HF_REPO_ID = "Exowz/ThoraxAI"


def load_results():
    try:
        with open("outputs/results.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _samples_present() -> bool:
    return (
        (SAMPLES_DIR / "NORMAL").is_dir()
        and (SAMPLES_DIR / "PNEUMONIA").is_dir()
        and any((SAMPLES_DIR / "NORMAL").iterdir())
        and any((SAMPLES_DIR / "PNEUMONIA").iterdir())
    )


@st.cache_data
def ensure_samples():
    if not _samples_present():
        print("Samples manquants, telechargement depuis HF Hub...")
        snapshot_download(
            repo_id=HF_REPO_ID,
            allow_patterns=["samples/*"],
            local_dir=".",
        )
        print(f"Telechargement termine, samples present: {_samples_present()}")


def samples_available() -> bool:
    ensure_samples()
    return _samples_present()


def list_samples(cls: str) -> list[Path]:
    d = SAMPLES_DIR / cls
    return sorted(
        [p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda p: p.name,
    )
