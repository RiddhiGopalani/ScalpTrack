"""Matplotlib helpers shared by the notebooks (no ``cv2.imshow``).

Display copies are normalised here; analysis arrays are never modified.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .preprocessing import signed_to_display, to_display_uint8
from .segmentation import overlay_mask

STATUS_COLORS = {"ok": "#2e7d32", "exploratory": "#ef6c00", "synthetic_demo": "#6a1b9a", "unavailable": "#757575",
                 "failed": "#c62828", "not_requested": "#9e9e9e", "warning": "#f9a825"}


def show_grid(images: list[np.ndarray], titles: list[str], ncols: int = 4, cmap: str = "gray", size: float = 3.2,
              suptitle: str | None = None, signed: set[int] | None = None):
    """Grid of images; indices in ``signed`` are shown with a symmetric diverging map."""
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i >= n:
            continue
        im = images[i]
        if signed and i in signed:
            m = float(np.max(np.abs(im))) or 1.0
            ax.imshow(im, cmap="RdBu_r", vmin=-m, vmax=m)
        elif im.ndim == 2:
            if im.dtype == bool:
                ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            elif im.dtype == np.uint8:
                ax.imshow(im, cmap=cmap, vmin=0, vmax=255)
            else:
                ax.imshow(im, cmap=cmap)
        else:
            ax.imshow(im)
        ax.set_title(titles[i], fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


def synthetic_banner(fig, text: str = "SYNTHETIC algorithm demonstration - not real scalp data") -> None:
    """Stamp a provenance caption on the figure itself (spec Section 17.2)."""
    fig.text(0.5, 0.005, text, ha="center", va="bottom", fontsize=9, color="#6a1b9a",
             bbox=dict(facecolor="white", edgecolor="#6a1b9a", boxstyle="round,pad=0.3"))


def save(fig, path: str | Path, dpi: int = 130) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    return p


def class_score_bar(ax, scores: dict[str, float], title: str = "Class scores (model scores, not clinical probabilities)"):
    names = list(scores)[::-1]
    vals = [scores[n] for n in names]
    ax.barh(names, vals, color="#1565c0")
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)


def confusion_plot(cm: list[list[int]], labels: list[str], title: str, ax=None):
    cm = np.asarray(cm)
    if ax is None:
        _, ax = plt.subplots(figsize=(1.0 + 0.6 * len(labels), 0.8 + 0.6 * len(labels)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title, fontsize=10)
    return ax


def count_overlay(rgb: np.ndarray, mask: np.ndarray, count: dict) -> np.ndarray:
    """Hair mask overlay plus the counting line and numbered crossing positions (drawn by caller)."""
    return overlay_mask(rgb, mask, (255, 60, 60), 0.45)


def draw_count(ax, rgb: np.ndarray, mask: np.ndarray, count: dict, title: str):
    ax.imshow(count_overlay(rgb, mask, count))
    if count.get("line_xyxy") is not None:
        x0, y0, x1, y1 = count["line_xyxy"]
        ax.plot([x0, x1], [y0, y1], color="yellow", lw=1.5)
        for k, (x, y) in enumerate(count.get("run_centres_xy", []), start=1):
            ax.plot(x, y, "o", color="cyan", ms=4)
            ax.text(x + 2, y - 2, str(k), color="cyan", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def draw_width(ax, rgb: np.ndarray, width: dict, title: str):
    ax.imshow(rgb)
    s = width.get("_samples")
    if s is not None and len(s["xy"]):
        sc = ax.scatter(s["xy"][:, 0], s["xy"][:, 1], c=s["widths_px"], s=3, cmap="viridis")
        plt.colorbar(sc, ax=ax, fraction=0.046, label="apparent width (px)")
        ex = np.argwhere(s["excluded"])
        if len(ex):
            ax.scatter(ex[:, 1], ex[:, 0], s=2, c="red", label="excluded (tip/junction/border)")
            ax.legend(fontsize=7, loc="lower right")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def badges_table(ax, badges: list[dict]):
    ax.axis("off")
    ax.set_title("Capability status", fontsize=10, loc="left")
    for i, b in enumerate(badges):
        y = 1 - (i + 0.5) / len(badges)
        ax.text(0.0, y, f" {b['status']} ", color="white", fontsize=8, va="center",
                bbox=dict(facecolor=STATUS_COLORS.get(b["status"], "#455a64"), edgecolor="none", boxstyle="round"))
        ax.text(0.27, y, f"{b['capability']}: {b['reason'][:70]}", fontsize=7.5, va="center")


def result_panel(result: dict, badges: list[dict], title: str | None = None):
    """The single-image result panel of spec Section 17.1."""
    a = result["_arrays"]
    seg = a["segmentation"]
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(a["rgb"])
    x0, y0, x1, y1 = result["roi_xyxy"]
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color="yellow", lw=1.5))
    ax.set_title("Original + native ROI", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    cls = result["classification"]
    if cls.get("class_scores"):
        class_score_bar(ax, cls["class_scores"], f"Prediction: {cls['predicted_label']} [{cls['status']}]")
    else:
        ax.text(0.5, 0.5, f"Classification {cls['status']}\n{cls.get('reason', '')}", ha="center", va="center",
                fontsize=8, wrap=True)
        ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    ov = overlay_mask(a["roi_rgb"], seg["hair"], (255, 50, 50), 0.5)
    ov = overlay_mask(ov, seg["flake"], (50, 220, 255), 0.7)
    ov = overlay_mask(ov, a["edges"], (255, 255, 0), 0.6)
    ax.imshow(ov)
    ax.set_title("Hair (red) / flake (cyan) candidates, Canny (yellow)", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 3])
    hc = result["hair_count"]
    draw_count(ax, a["roi_rgb"], seg["hair"], a["count"],
               f"Reference-line crossings = {hc['value']} [{hc['status']}]\n(not a follicle count)")

    ax = fig.add_subplot(gs[1, 0])
    sw = result["shaft_width"]
    med = "n/a" if sw["median_px"] is None else f"{sw['median_px']:.1f}"
    draw_width(ax, a["roi_rgb"], a["width"], f"Apparent width median {med} px [{sw['status']}]\num: {sw.get('physical_status')}")

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    tf = result["texture_features"]
    keys = ["glcm_contrast_mean", "glcm_energy_mean", "glcm_entropy_bits_mean", "hist_entropy_bits",
            "glcm_homogeneity_mean", "canny_edge_fraction", "lab_a_mean", "v_mean"]
    tbl = ax.table(cellText=[[k, f"{tf[k]:.4g}"] for k in keys], colLabels=["texture feature", "value"],
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    ax.set_title("Texture features (classifier geometry)", fontsize=9)

    ax = fig.add_subplot(gs[1, 2:])
    badges_table(ax, badges)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


__all__ = ["show_grid", "synthetic_banner", "save", "class_score_bar", "confusion_plot", "draw_count", "draw_width",
           "badges_table", "result_panel", "signed_to_display", "to_display_uint8"]
