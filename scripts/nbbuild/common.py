"""Cell helpers shared by every notebook definition."""

from __future__ import annotations

import textwrap

import nbformat


def md(text: str):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbformat.v4.new_code_cell(textwrap.dedent(text).strip())


def setup_cell(notebook_tag: str, extra_imports: str = "") -> object:
    src = textwrap.dedent("""
        %matplotlib inline
        from pathlib import Path
        import sys

        # Robust project-root discovery: works from notebooks/ or the project root, installed or not.
        ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs" / "default.yaml").is_file())
        if str(ROOT / "src") not in sys.path:
            sys.path.insert(0, str(ROOT / "src"))

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scalptrack.config import load_config, seed_everything, project_path, display_path
        from scalptrack.demo_data import dataset_status, development_images, synthetic_closeups, figure_path
        @@EXTRA@@

        cfg = load_config()
        rng = seed_everything(cfg["seed"])
        NB = "@@TAG@@"
        rel = lambda p: display_path(p, ROOT)  # noqa: E731 - short path for printing
        pd.set_option("display.width", 160)
        pd.set_option("display.max_columns", 40)
        plt.rcParams["figure.dpi"] = 90
        print("project root:", ROOT)
        print("config:", Path(cfg["config_path"]).name, "| seed:", cfg["seed"])
    """).strip()
    src = src.replace("@@EXTRA@@", textwrap.dedent(extra_imports).strip()).replace("@@TAG@@", notebook_tag)
    return nbformat.v4.new_code_cell(src)


def status_cell() -> object:
    return code("""
        STATUS = dataset_status(cfg)
        print(STATUS["message"])
        REAL = STATUS["available"]
        DEV = development_images(cfg, n_per_class=1, max_total=4) if REAL else []
        SYN = synthetic_closeups(cfg)
        print(f"real development images loaded: {len(DEV)} (train/validation only) | synthetic fixtures: {list(SYN)}")
    """)


def save_cell_fn() -> object:
    return code('''
        def save_fig(fig, name, provenance):
            """Save under outputs/figures/<NB>/<real|synthetic>/ ; synthetic figures get a caption and SYNTHETIC_ prefix."""
            from scalptrack.viz import synthetic_banner

            if provenance == "synthetic":
                synthetic_banner(fig)
            p = figure_path(cfg, NB, name, provenance)
            fig.savefig(p, dpi=120, bbox_inches="tight")
            print("saved", rel(p))
            return p
    ''')


def notebook(cells: list) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb
