from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = REPO_ROOT / "experiments"


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture(scope="session")
def archive_dir() -> Path:
    if not ARCHIVE.is_dir():
        pytest.skip("archived experiments/ tree not available")
    return ARCHIVE
