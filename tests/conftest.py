"""Shared pytest fixtures: configuration and tiny synthetic images (no dataset download needed)."""

import numpy as np
import pytest
from PIL import Image

from scalptrack.config import load_config
from scalptrack import synthetic as syn


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def hair_rgb():
    return syn.render_hair_image(syn.straight_shafts(4, 5.0, 90.0, size=128)["mask"], seed=1)


@pytest.fixture
def png_path(tmp_path, hair_rgb):
    p = tmp_path / "hair.png"
    Image.fromarray(hair_rgb).save(p)
    return p


@pytest.fixture
def class_folder_dataset(tmp_path):
    """Tiny <root>/<split>/<class>/ export with one exact duplicate across splits and one corrupt file."""
    root = tmp_path / "proj"
    base = root / "data" / "raw" / "toy"
    rng = np.random.default_rng(0)
    for si, split in enumerate(("train", "val", "test")):
        for ci, cls in enumerate(("Class A", "Class B")):
            d = base / "Wrapper" / split / cls
            d.mkdir(parents=True)
            for k in range(4):
                img = syn.synthetic_texture_sample("synthetic_sparse_strands" if cls == "Class A" else "synthetic_flaky_surface",
                                                   seed=100 * si + 10 * ci + k, size=64)
                Image.fromarray(img).save(d / f"{k}.png")
    # exact duplicate of a train image placed in test -> leakage that must be detected
    (base / "Wrapper" / "test" / "Class A" / "dup.png").write_bytes((base / "Wrapper" / "train" / "Class A" / "0.png").read_bytes())
    (base / "Wrapper" / "train" / "Class B" / "broken.jpg").write_bytes(b"not a jpeg at all")
    (base / "Wrapper" / "readme.txt").write_text("notes")
    return root, base, rng
