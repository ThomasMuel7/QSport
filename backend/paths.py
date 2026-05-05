from __future__ import annotations

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALISATION_DIR = PROJECT_ROOT / "visualisation"
PDFS_DIR = PROJECT_ROOT / "pdfs"
IMAGES_DIR = PROJECT_ROOT / "images"


def data_path(*parts: str) -> Path:
    return DATA_DIR.joinpath(*parts)


def model_path(*parts: str) -> Path:
    return MODELS_DIR.joinpath(*parts)


def repo_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)