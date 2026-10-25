"""Exploratory visible-hair counting and apparent shaft width (REQ-05..07, spec Sections 10-11).

Quantity names are deliberately precise:
* ``isolated_strand_candidates`` - separate shaft-like components after filtering.
* ``reference_line_crossings``   - distinct mask runs crossed by a documented line.
Neither is a follicle count, and neither is hairs per cm^2.

Width is *apparent 2-D shaft width in native-ROI pixels*. Micrometres are only
produced from a documented scale (``width_um = width_px * s_um_per_px``).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from .schema import Status, measurement

_NEIGHBOUR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


# --------------------------------------------------------------------------- skeleton tools


def skeleton(mask: np.ndarray) -> np.ndarray:
    """One-pixel-wide skeleton of a boolean mask (scikit-image)."""
    m = mask.astype(bool)
    return skeletonize(m) if m.any() else np.zeros_like(m)


def neighbour_count(skel: np.ndarray) -> np.ndarray:
    """Number of 8-connected skeleton neighbours of every skeleton pixel (0 elsewhere)."""
    return cv2.filter2D(skel.astype(np.uint8), -1, _NEIGHBOUR_KERNEL, borderType=cv2.BORDER_CONSTANT) * skel


def endpoints_and_junctions(skel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Endpoints have one 8-neighbour on the skeleton; junctions have three or more."""
    n = neighbour_count(skel)
    return (n == 1) & skel, (n >= 3) & skel


def component_table(mask: np.ndarray, min_area: int = 1) -> tuple[np.ndarray, list[dict]]:
    """Label 8-connected components and describe each (area, skeleton length, tips, junctions, shape)."""
    m = mask.astype(bool)
    labels, n = ndimage.label(m, structure=np.ones((3, 3)))
    skel = skeleton(m)
    ends, junc = endpoints_and_junctions(skel)
    rows: list[dict] = []
    kept = np.zeros_like(labels)
    new_id = 0
    for lab, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        comp = labels[sl] == lab
        area = int(comp.sum())
        if area < min_area:
            continue
        new_id += 1
        kept[sl][comp] = new_id
        ys, xs = np.nonzero(comp)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        orient, elong = _moments_orientation(xs, ys)
        sk = skel[sl] & comp
        rows.append(
            {
                "id": new_id,
                "area_px": area,
                "skeleton_length_px": int(sk.sum()),
                "n_endpoints": int((ends[sl] & comp).sum()),
                "n_junction_px": int((junc[sl] & comp).sum()),
                "bbox_xyxy": [int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)],
                "centroid_xy": [float(xs.mean()), float(ys.mean())],
                "orientation_deg": orient,
                "elongation": elong,
            }
        )
    return kept, rows


def _moments_orientation(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Major-axis orientation (degrees in [0, 180), image x-right / y-down) and axis ratio."""
    if xs.size < 2:
        return 0.0, 1.0
    cov = np.cov(np.vstack([xs, ys]).astype(np.float64))
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, np.argmax(evals)]
    angle = float(np.rad2deg(np.arctan2(major[1], major[0])) % 180.0)
    lo, hi = max(float(evals.min()), 1e-9), float(evals.max())
    return angle, float(np.sqrt(hi / lo))


def shaft_like_mask(mask: np.ndarray, config: dict) -> tuple[np.ndarray, int]:
    """Drop components that are both compact and short (e.g. dark halos around bright flakes).

    A component is excluded only if ``elongation < shaft_min_elongation`` AND
    ``skeleton_length < shaft_min_length_px``; long compact shapes such as two crossing
    shafts are kept. Returns the filtered mask and the number of excluded components.
    """
    m = config["measurements"]
    labels, rows = component_table(mask, m["min_component_area"])
    keep_ids = [
        r["id"]
        for r in rows
        if not (
            r["elongation"] < m["shaft_min_elongation"] and r["skeleton_length_px"] < m["shaft_min_length_px"]
        )
    ]
    return np.isin(labels, keep_ids) & (labels > 0), len(rows) - len(keep_ids)


def dominant_orientation(components: list[dict]) -> float | None:
    """Length-weighted doubled-angle mean of component orientations (degrees), or None if no components."""
    if not components:
        return None
    w = np.array([c["skeleton_length_px"] for c in components], dtype=np.float64)
    a = np.deg2rad(np.array([c["orientation_deg"] for c in components]) * 2.0)
    if w.sum() == 0:
        w = np.ones_like(w)
    return float((np.rad2deg(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())) / 2.0) % 180.0)


# --------------------------------------------------------------------------- crossings


def perpendicular_line(
    shape_hw: tuple[int, int], hair_orientation_deg: float, position: float = 0.5
) -> tuple[float, float, float, float]:
    """Line through the ROI perpendicular to the dominant shaft direction, clipped to the ROI.

    ``position`` slides the line's anchor point along the hair direction (0..1 of
    the ROI extent along that direction); 0.5 passes through the centre.
    """
    h, w = shape_hw
    theta = np.deg2rad(hair_orientation_deg)
    hair_dir = np.array([np.cos(theta), np.sin(theta)])
    line_dir = np.array([-hair_dir[1], hair_dir[0]])
    centre = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    half_extent = 0.5 * (abs(hair_dir[0]) * (w - 1) + abs(hair_dir[1]) * (h - 1))
    anchor = centre + hair_dir * (position - 0.5) * 2.0 * half_extent
    return _clip_line(anchor, line_dir, w, h)


def _clip_line(p: np.ndarray, d: np.ndarray, w: int, h: int) -> tuple[float, float, float, float]:
    ts = []
    for axis, lim in ((0, w - 1), (1, h - 1)):
        if abs(d[axis]) > 1e-12:
            ts += [(0 - p[axis]) / d[axis], (lim - p[axis]) / d[axis]]
    pts = [p + t * d for t in ts]
    pts = [q for q in pts if -1e-6 <= q[0] <= w - 1 + 1e-6 and -1e-6 <= q[1] <= h - 1 + 1e-6]
    if len(pts) < 2:
        raise ValueError("Counting line does not intersect the ROI")
    pts.sort(key=lambda q: float(np.dot(q - p, d)))
    a, b = pts[0], pts[-1]
    return float(a[0]), float(a[1]), float(b[0]), float(b[1])


def sample_line(mask: np.ndarray, line_xyxy) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-pixel samples of ``mask`` along a line, one per pixel of length. Returns (values, xy)."""
    x0, y0, x1, y1 = line_xyxy
    n = int(np.ceil(np.hypot(x1 - x0, y1 - y0))) + 1
    xs = np.clip(np.rint(np.linspace(x0, x1, n)).astype(int), 0, mask.shape[1] - 1)
    ys = np.clip(np.rint(np.linspace(y0, y1, n)).astype(int), 0, mask.shape[0] - 1)
    return mask[ys, xs].astype(bool), np.stack([xs, ys], axis=1)


def count_runs(values: np.ndarray, merge_gap: int = 0) -> list[tuple[int, int]]:
    """Runs of True as (start, stop_exclusive); runs separated by <= ``merge_gap`` False samples merge."""
    v = np.asarray(values, dtype=bool)
    if not v.any():
        return []
    d = np.diff(np.concatenate([[0], v.astype(np.int8), [0]]))
    runs = list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))
    merged = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


# --------------------------------------------------------------------------- counting


def estimate_visible_hairs(
    rgb_roi: np.ndarray | None,
    hair_mask: np.ndarray,
    config: dict,
    provenance: str = "real",
    quality: dict | None = None,
    line_xyxy=None,
    line_position: float = 0.5,
) -> dict:
    """Candidate-component count and reference-line crossing count for a native-resolution ROI.

    ``provenance='synthetic'`` marks results as ``synthetic_demo``; real images give
    ``exploratory`` because no count labels exist. If ``quality`` says the ROI is
    not measurement-eligible, both quantities are ``unavailable`` (null).
    """
    m = config["measurements"]
    mask = hair_mask.astype(bool)
    if rgb_roi is not None and rgb_roi.shape[:2] != mask.shape:
        raise ValueError("hair_mask must have the same height/width as the ROI")
    status = Status.SYNTHETIC_DEMO if provenance == "synthetic" else Status.EXPLORATORY

    if quality is not None and not quality.get("measurement_eligible", True):
        reason = f"ROI failed measurement quality gate: {quality.get('flags')}"
        return {
            "components": measurement(
                None, "candidates", Status.UNAVAILABLE, None, reason, quantity="isolated_strand_candidates"
            ),
            "crossings": measurement(
                None, "crossings", Status.UNAVAILABLE, None, reason, quantity="reference_line_crossings"
            ),
            "component_rows": [],
            "labels": None,
            "line_xyxy": None,
            "runs": [],
            "flags": ["quality_gate"],
        }

    mask, n_non_shaft = shaft_like_mask(mask, config)
    labels, rows = component_table(mask, m["min_component_area"])
    total_skel = sum(r["skeleton_length_px"] for r in rows)
    junction_px = sum(r["n_junction_px"] for r in rows)
    junction_fraction = junction_px / total_skel if total_skel else 0.0
    flags: list[str] = []
    if junction_fraction > m["max_junction_fraction"]:
        flags.append("branching_or_touching_shafts")
    if any(r["n_endpoints"] > 2 for r in rows):
        flags.append("components_with_more_than_two_tips")

    comp_reason = (
        "Separate shaft-like mask components after removing components "
        f"< {m['min_component_area']} px and {n_non_shaft} short compact non-shaft component(s). "
        "A component is not necessarily one hair."
    )
    if flags:
        comp_reason += f" Ambiguity flags: {flags}."
    components = measurement(
        len(rows),
        "candidates",
        status,
        "connected_components_8",
        comp_reason,
        quantity="isolated_strand_candidates",
        junction_fraction=junction_fraction,
        non_shaft_components_excluded=n_non_shaft,
    )

    orient = dominant_orientation(rows)
    if line_xyxy is None:
        if orient is None:
            crossings = measurement(
                0,
                "crossings",
                status,
                "reference_line",
                "Empty hair mask: a valid measurement that found no candidates.",
                quantity="reference_line_crossings",
            )
            return {
                "components": components,
                "crossings": crossings,
                "component_rows": rows,
                "labels": labels,
                "line_xyxy": None,
                "runs": [],
                "flags": flags,
                "dominant_orientation_deg": None,
            }
        line_xyxy = perpendicular_line(mask.shape, orient, line_position)
    # Crossings use the same area-filtered mask as the component count (tiny specks are not shafts).
    values, xy = sample_line(labels > 0, line_xyxy)
    runs = count_runs(values, m["crossing_merge_gap_px"])
    crossings = measurement(
        len(runs),
        "crossings",
        status,
        "reference_line",
        f"Distinct mask runs along a line perpendicular to the dominant shaft direction "
        f"(merge gap {m['crossing_merge_gap_px']} px). One curling shaft can cross more than once.",
        quantity="reference_line_crossings",
        line_xyxy=[round(v, 2) for v in line_xyxy],
    )
    run_xy = [xy[(s + e - 1) // 2].tolist() for s, e in runs]
    return {
        "components": components,
        "crossings": crossings,
        "component_rows": rows,
        "labels": labels,
        "line_xyxy": line_xyxy,
        "runs": runs,
        "run_centres_xy": run_xy,
        "flags": flags,
        "dominant_orientation_deg": orient,
    }


# --------------------------------------------------------------------------- width


def width_samples(hair_mask: np.ndarray, config: dict) -> dict:
    """Distance-transform width samples along the skeleton, excluding tips, junctions and borders."""
    m = config["measurements"]
    mask = shaft_like_mask(hair_mask.astype(bool), config)[0] if hair_mask.any() else hair_mask.astype(bool)
    if not mask.any():
        return {
            "widths_px": np.array([]),
            "xy": np.zeros((0, 2), int),
            "skeleton": np.zeros_like(mask),
            "excluded": np.zeros_like(mask),
        }
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    skel = skeleton(mask)
    ends, junc = endpoints_and_junctions(skel)
    r = int(m["width_exclude_radius_px"])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    near_special = cv2.dilate((ends | junc).astype(np.uint8), k) > 0
    border = np.zeros_like(mask)
    border[:r, :] = border[-r:, :] = True
    border[:, :r] = border[:, -r:] = True
    valid = skel & ~near_special & ~border
    ys, xs = np.nonzero(valid)
    widths = 2.0 * dist[ys, xs].astype(np.float64) - float(m["width_offset_px"])
    return {
        "widths_px": widths,
        "xy": np.stack([xs, ys], axis=1),
        "skeleton": skel,
        "excluded": skel & ~valid,
        "distance": dist,
    }


def to_physical_width(
    width_px: float | None,
    pixel_scale_um_per_px: float | None,
    scale_source: str | None = None,
    isotropic: bool = True,
) -> dict:
    """``width_um = width_px * s_um_per_px`` - only with a documented scale and isotropic geometry."""
    if width_px is None:
        return {
            "median_um": None,
            "physical_status": str(Status.UNAVAILABLE),
            "physical_reason": "No valid pixel width",
        }
    if pixel_scale_um_per_px is None:
        return {
            "median_um": None,
            "physical_status": str(Status.UNAVAILABLE),
            "physical_reason": "No documented pixel scale for this image/ROI; micrometres cannot be derived from pixels alone",
        }
    if pixel_scale_um_per_px <= 0:
        raise ValueError("pixel scale must be positive")
    if not scale_source:
        return {
            "median_um": None,
            "physical_status": str(Status.UNAVAILABLE),
            "physical_reason": "Pixel scale supplied without provenance (scale_source); refusing to convert",
        }
    if not isotropic:
        return {
            "median_um": None,
            "physical_status": str(Status.UNAVAILABLE),
            "physical_reason": "Non-isotropic resize: a single scalar scale does not apply",
        }
    return {
        "median_um": float(width_px * pixel_scale_um_per_px),
        "physical_status": str(Status.OK),
        "physical_reason": f"Scale {pixel_scale_um_per_px} um/px from {scale_source}",
    }


def estimate_shaft_width(
    hair_mask: np.ndarray,
    config: dict,
    provenance: str = "real",
    quality: dict | None = None,
    pixel_scale_um_per_px: float | None = None,
    scale_source: str | None = None,
) -> dict:
    """Median / IQR apparent shaft width in native pixels, or an unavailable result with a reason."""
    m = config["measurements"]
    base = {
        "median_px": None,
        "iqr_px": None,
        "valid_samples": 0,
        "median_um": None,
        "pixel_scale_um_per_px": pixel_scale_um_per_px,
        "method": "distance_transform_on_skeleton",
        "unit": "px",
        "flags": [],
        "params": {
            "exclude_radius_px": m["width_exclude_radius_px"],
            "width_offset_px": m["width_offset_px"],
            "min_resolvable_width_px": m["min_resolvable_width_px"],
        },
    }
    if quality is not None and not quality.get("measurement_eligible", True):
        base.update(
            status=str(Status.UNAVAILABLE),
            reason=f"ROI failed measurement quality gate: {quality.get('flags')}",
        )
        base.update(to_physical_width(None, pixel_scale_um_per_px, scale_source))
        return base
    s = width_samples(hair_mask, config)
    w = s["widths_px"]
    if w.size == 0:
        base.update(
            status=str(Status.UNAVAILABLE),
            reason="No valid skeleton samples (empty mask, or only tips/junctions/border pixels)",
        )
        base.update(to_physical_width(None, pixel_scale_um_per_px, scale_source))
        return base
    q25, med, q75 = np.percentile(w, [25, 50, 75])
    status = Status.SYNTHETIC_DEMO if provenance == "synthetic" else Status.EXPLORATORY
    flags = []
    if med < m["min_resolvable_width_px"]:
        flags.append("below_min_resolvable_width")
    base.update(
        median_px=float(med),
        iqr_px=float(q75 - q25),
        q25_px=float(q25),
        q75_px=float(q75),
        valid_samples=int(w.size),
        flags=flags,
        status=str(status),
        reason=(
            "Apparent projected 2-D width from 2*distance-to-background on the skeleton; "
            "not the true cylindrical diameter. Unvalidated on real images."
            if provenance != "synthetic"
            else "Synthetic fixture with known width."
        ),
    )
    base.update(to_physical_width(float(med), pixel_scale_um_per_px, scale_source))
    base["_samples"] = s
    return base
