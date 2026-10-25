"""Deterministic synthetic verification fixtures.

Everything here is GENERATED, not photographed. Fixtures verify algorithm
behaviour against known truth (counts, widths, disparities). They never enter
a real-image training/test split and cannot establish performance on human
scalp images. Every returned dict carries ``provenance='synthetic'``.
"""

from __future__ import annotations

import cv2
import numpy as np

SYNTHETIC_LABEL = "SYNTHETIC - algorithm verification fixture, not real scalp data"

SKIN_RGB = np.array([214, 170, 150], dtype=np.float64)
HAIR_RGB = np.array([45, 32, 28], dtype=np.float64)


# --------------------------------------------------------------------------- shaft rasterisation


def capsule_mask(shape_hw: tuple[int, int], p0, p1, width: float) -> np.ndarray:
    """Pixels whose centre lies within ``width/2`` of the segment p0-p1 (rounded ends)."""
    h, w = shape_hw
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L2 = float(d @ d)
    t = np.zeros_like(xx) if L2 == 0 else np.clip(((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / L2, 0, 1)
    px, py = p0[0] + t * d[0], p0[1] + t * d[1]
    return (xx - px) ** 2 + (yy - py) ** 2 <= (width / 2.0) ** 2


def polyline_mask(shape_hw, points, width: float) -> np.ndarray:
    m = np.zeros(shape_hw, dtype=bool)
    for a, b in zip(points[:-1], points[1:]):
        m |= capsule_mask(shape_hw, a, b, width)
    return m


def straight_shafts(n: int, width: float, angle_deg: float = 90.0, size: int = 160,
                    spacing: float | None = None) -> dict:
    """``n`` separated parallel straight shafts of known width crossing the whole ROI.

    ``angle_deg`` is the shaft direction (90 = vertical). Truth: ``n`` components and
    ``n`` crossings of a perpendicular line.
    """
    spacing = spacing or max(width * 3.0, (size * 0.8) / max(n, 1))
    theta = np.deg2rad(angle_deg)
    hair_dir = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-hair_dir[1], hair_dir[0]])
    c = np.array([(size - 1) / 2.0, (size - 1) / 2.0])
    mask = np.zeros((size, size), dtype=bool)
    reach = size * 1.5
    for k in range(n):
        off = (k - (n - 1) / 2.0) * spacing
        base = c + normal * off
        mask |= capsule_mask((size, size), base - hair_dir * reach, base + hair_dir * reach, width)
    return {"mask": mask, "true_components": n, "true_crossings": n, "true_width_px": width,
            "angle_deg": angle_deg, "kind": "straight_separated", "provenance": "synthetic"}


def curved_shafts(n: int, width: float, amplitude: float = 8.0, size: int = 160, seed: int = 0) -> dict:
    """Separated, gently curved (sinusoidal) roughly-vertical shafts.

    The amplitude is clamped so neighbouring shafts (random phases) stay more than 4 px apart -
    wider than the 2-px crossing merge rule: ``2 * amplitude < spacing - width - 4``.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), dtype=bool)
    xs0 = np.linspace(size * 0.15, size * 0.85, n)
    if n > 1:
        spacing = xs0[1] - xs0[0]
        amplitude = float(max(0.0, min(amplitude, (spacing - width - 4.0) / 2.0 - 0.01)))
    ys = np.linspace(-5, size + 5, 60)
    for x0 in xs0:
        phase = rng.uniform(0, 2 * np.pi)
        pts = np.stack([x0 + amplitude * np.sin(ys / size * 2 * np.pi + phase), ys], axis=1)
        mask |= polyline_mask((size, size), pts, width)
    return {"mask": mask, "true_components": n, "true_crossings": n, "true_width_px": width,
            "kind": "curved_separated", "provenance": "synthetic"}


def challenging_fixtures(width: float = 5.0, size: int = 160, seed: int = 0) -> dict[str, dict]:
    """Deliberately hard cases with their *intended* truth, to expose counting/width errors."""
    rng = np.random.default_rng(seed)
    s = (size, size)
    out: dict[str, dict] = {}

    # Two shafts that cross in an X: 2 hairs, but one connected component.
    m = capsule_mask(s, (20, 10), (140, 150), width) | capsule_mask(s, (140, 10), (20, 150), width)
    out["crossing_X"] = {"mask": m, "true_hairs": 2, "kind": "crossing"}

    # Two parallel shafts touching along part of their length.
    m = capsule_mask(s, (70, -5), (70, 165), width) | polyline_mask(s, [(92, -5), (73, 60), (73, 100), (92, 165)], width)
    out["touching"] = {"mask": m, "true_hairs": 2, "kind": "touching"}

    # One shaft broken into fragments by gaps: 1 hair, several components.
    m = capsule_mask(s, (80, -5), (80, 165), width)
    for y in (40, 80, 120):
        m[y:y + 3, :] = False
    out["fragmented"] = {"mask": m, "true_hairs": 1, "kind": "fragmented"}

    # One curling shaft crossing a horizontal reference line three times.
    t = np.linspace(0, 1, 120)
    pts = np.stack([20 + 120 * t, 80 + 30 * np.sin(t * 3 * np.pi)], axis=1)
    m = polyline_mask(s, pts, width)
    out["curl_multi_crossing"] = {"mask": m, "true_hairs": 1, "kind": "curl", "reference_line": (0.0, 80.0, size - 1.0, 80.0)}

    # Salt noise sprinkled into a 3-shaft mask.
    base = straight_shafts(3, width, 90.0, size)["mask"].copy()
    noise = rng.random(s) < 0.01
    out["noisy"] = {"mask": base | noise, "true_hairs": 3, "kind": "noise"}
    for v in out.values():
        v["true_width_px"] = width
        v["provenance"] = "synthetic"
    return out


# --------------------------------------------------------------------------- image rendering


def render_hair_image(mask: np.ndarray, contrast: float = 1.0, blur_sigma: float = 0.8, noise_sigma: float = 4.0,
                      seed: int = 0, skin_rgb=SKIN_RGB, hair_rgb=HAIR_RGB, background_texture: float = 6.0) -> np.ndarray:
    """RGB close-up-like image from a hair mask. ``contrast`` < 1 makes hair closer to skin (light/grey hair)."""
    rng = np.random.default_rng(seed)
    h, w = mask.shape
    skin = np.asarray(skin_rgb, float)
    hair = skin + contrast * (np.asarray(hair_rgb, float) - skin)
    low = cv2.resize(rng.normal(0, 1, (max(2, h // 16), max(2, w // 16))), (w, h), interpolation=cv2.INTER_CUBIC)
    img = np.empty((h, w, 3), dtype=np.float64)
    img[:] = skin
    img += background_texture * low[..., None]
    img[mask.astype(bool)] = hair
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    img += rng.normal(0, noise_sigma, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def add_shadow(rgb: np.ndarray, strength: float = 0.85, region: tuple | None = None) -> np.ndarray:
    """Darken a rectangular region strongly - the 'dark shadow mistaken for hair' failure case."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = region or (0, int(h * 0.6), int(w * 0.45), h)
    out = rgb.astype(np.float64).copy()
    out[y0:y1, x0:x1] *= (1.0 - strength)
    return out.astype(np.uint8)


def add_glare(rgb: np.ndarray, centre=None, radius: float = 10.0, peak: float = 235.0) -> np.ndarray:
    """Add a small bright, desaturated highlight - the 'glare mistaken for flakes' failure case."""
    h, w = rgb.shape[:2]
    cx, cy = centre or (w * 0.7, h * 0.3)
    yy, xx = np.mgrid[0:h, 0:w]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius ** 2))[..., None]
    out = rgb.astype(np.float64) * (1 - g) + peak * g
    return np.clip(out, 0, 255).astype(np.uint8)


def add_flakes(rgb: np.ndarray, n: int = 12, seed: int = 0, radius_range=(2, 4)) -> tuple[np.ndarray, np.ndarray]:
    """Draw small white blobs (synthetic flake stand-ins); returns image and the true flake mask."""
    rng = np.random.default_rng(seed)
    h, w = rgb.shape[:2]
    out = rgb.copy()
    truth = np.zeros((h, w), dtype=np.uint8)
    for _ in range(n):
        c = (int(rng.integers(8, w - 8)), int(rng.integers(8, h - 8)))
        r = int(rng.integers(radius_range[0], radius_range[1] + 1))
        cv2.circle(truth, c, r, 1, -1)
        cv2.circle(out, c, r, (240, 236, 228), -1, lineType=cv2.LINE_AA)
    return out, truth.astype(bool)


def clean_noisy_pair(size: int = 160, noise_sigma: float = 15.0, seed: int = 0) -> dict:
    """Known clean grayscale fixture plus its noisy copy for MSE/PSNR filter comparisons."""
    from .preprocessing import add_gaussian_noise, to_gray

    clean_rgb = render_hair_image(curved_shafts(6, 4.0, size=size, seed=seed)["mask"], noise_sigma=0.0, seed=seed)
    clean = to_gray(clean_rgb)
    return {"clean": clean, "noisy": add_gaussian_noise(clean, noise_sigma, seed + 1), "noise_sigma": noise_sigma,
            "provenance": "synthetic"}


# --------------------------------------------------------------------------- classifier smoke-test set

SYNTHETIC_CLASSES = ("synthetic_sparse_strands", "synthetic_dense_strands", "synthetic_flaky_surface")


def synthetic_texture_sample(label: str, seed: int, size: int = 128) -> np.ndarray:
    """One generated texture image for a synthetic class (used only to exercise the classifier code).

    Class parameter ranges deliberately overlap (strand count, width, contrast, flakes) so the
    software check produces realistic confusions instead of a trivially separable task.
    """
    rng = np.random.default_rng(seed)
    contrast = rng.uniform(0.45, 1.0)
    amp = rng.uniform(2, 10)
    if label == "synthetic_sparse_strands":
        n, wid, flakes = int(rng.integers(4, 10)), float(rng.uniform(2.5, 4.5)), int(rng.integers(0, 6))
    elif label == "synthetic_dense_strands":
        n, wid, flakes = int(rng.integers(8, 15)), float(rng.uniform(2.0, 3.8)), int(rng.integers(0, 6))
    elif label == "synthetic_flaky_surface":
        n, wid, flakes = int(rng.integers(4, 11)), float(rng.uniform(2.5, 4.5)), int(rng.integers(4, 20))
    else:
        raise KeyError(f"Unknown synthetic class {label!r}")
    mask = curved_shafts(n, wid, amplitude=amp, size=size, seed=seed)["mask"]
    img = render_hair_image(mask, contrast=contrast, blur_sigma=rng.uniform(0.6, 1.4), noise_sigma=rng.uniform(2, 8), seed=seed)
    return add_flakes(img, n=flakes, seed=seed, radius_range=(1, 3))[0] if flakes else img


def synthetic_classification_set(n_sources_per_class: int = 30, variants_per_source: int = 3, seed: int = 0,
                                 size: int = 128) -> list[dict]:
    """Synthetic images with *groups*: each source has flipped/rotated variants sharing ``group_id``.

    The variants mimic augmentation derivatives in public datasets, so the
    group-aware split can be demonstrated: variants of one source must never be
    split across train/validation/test.
    """
    rows = []
    ops = [lambda a: a, lambda a: a[:, ::-1], lambda a: a[::-1, :], lambda a: np.rot90(a)]
    for ci, label in enumerate(SYNTHETIC_CLASSES):
        for s in range(n_sources_per_class):
            src_seed = seed * 100_000 + ci * 1000 + s
            base = synthetic_texture_sample(label, src_seed, size)
            gid = f"syn_{label}_{s:03d}"
            for v in range(variants_per_source):
                rows.append({"image": np.ascontiguousarray(ops[v % len(ops)](base)), "label": label,
                             "group_id": gid, "image_id": f"{gid}_v{v}", "provenance": "synthetic"})
    return rows


# --------------------------------------------------------------------------- stereo scene


def _value_noise(rng: np.random.Generator, shape: tuple[int, int], octaves=(8, 24, 64)) -> np.ndarray:
    """Multi-octave smooth random field in [0, 1] on a world-coordinate grid."""
    acc = np.zeros(shape)
    amp_total = 0.0
    for i, cell in enumerate(octaves):
        g = rng.random((max(2, shape[0] // cell + 2), max(2, shape[1] // cell + 2)))
        up = cv2.resize(g, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
        amp = 1.0 / (i + 1)
        acc += amp * up
        amp_total += amp
    acc /= amp_total
    fine = rng.random(shape)
    acc = 0.75 * acc + 0.25 * cv2.GaussianBlur(fine, (0, 0), 0.7)
    acc -= acc.min()
    return acc / (acc.max() or 1.0)


def render_stereo_scene(width: int = 480, height: int = 360, focal_px: float = 500.0, baseline_mm: float = 60.0,
                        seed: int = 0, include_dome: bool = True, supersample: int = 2, noise_sigma: float = 1.5) -> dict:
    """Render a rectified stereo pair of ONE synthetic 3-D scene with a known pinhole model.

    Scene (millimetres, left camera at the origin looking down +Z, right camera at X = +baseline):
    a textured background plane, two fronto-parallel textured rectangles at known
    depths and optionally a spherical dome (a curved, scalp-like *synthetic* surface).
    Both images are rendered by ray casting the same world texture, so the right
    image is not a shifted copy of the left; occlusions arise naturally.
    """
    rng = np.random.default_rng(seed)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    surfaces = [
        {"id": 0, "type": "plane", "Z": 1500.0, "x": (-2000, 2000), "y": (-2000, 2000), "name": "background"},
        {"id": 1, "type": "plane", "Z": 900.0, "x": (-330, 20), "y": (-200, 140), "name": "rect_far"},
        {"id": 2, "type": "plane", "Z": 650.0, "x": (60, 270), "y": (-60, 200), "name": "rect_near"},
    ]
    if include_dome:
        surfaces.append({"id": 3, "type": "sphere", "C": (-120.0, 150.0, 1050.0), "R": 170.0, "name": "dome"})
    tex_res = 2.0  # mm per texture cell
    tex_size = (int(4200 / tex_res), int(4200 / tex_res))
    textures = {s["id"]: _value_noise(np.random.default_rng(seed * 10 + s["id"] + 1), tex_size) for s in surfaces}

    def cast(cam_x: float, ss: int):
        hh, ww = height * ss, width * ss
        v, u = np.mgrid[0:hh, 0:ww].astype(np.float64)
        u = (u + 0.5) / ss - 0.5
        v = (v + 0.5) / ss - 0.5
        dx, dy = (u - cx) / focal_px, (v - cy) / focal_px
        depth = np.full(u.shape, np.inf)
        sid = np.full(u.shape, -1, dtype=int)
        X = np.zeros_like(u)
        Y = np.zeros_like(u)
        for s in surfaces:
            if s["type"] == "plane":
                Z = s["Z"]
                xw, yw = cam_x + dx * Z, dy * Z
                hit = (xw >= s["x"][0]) & (xw <= s["x"][1]) & (yw >= s["y"][0]) & (yw <= s["y"][1]) & (Z < depth)
                depth[hit], sid[hit], X[hit], Y[hit] = Z, s["id"], xw[hit], yw[hit]
            else:
                Cx, Cy, Cz = s["C"]
                ox = cam_x - Cx
                # Ray o + t*(dx, dy, 1); solve |o + t d - C|^2 = R^2 with o = (cam_x, 0, 0).
                a = dx * dx + dy * dy + 1.0
                b = 2.0 * (dx * ox + dy * (0 - Cy) + 1.0 * (0 - Cz))
                c = ox * ox + Cy * Cy + Cz * Cz - s["R"] ** 2
                disc = b * b - 4 * a * c
                ok = disc >= 0
                t = np.where(ok, (-b - np.sqrt(np.where(ok, disc, 0))) / (2 * a), np.inf)
                Z = t  # direction z-component is 1, so t equals depth
                hit = ok & (Z > 0) & (Z < depth)
                depth[hit], sid[hit] = Z[hit], s["id"]
                X[hit], Y[hit] = cam_x + dx[hit] * Z[hit], dy[hit] * Z[hit]
        img = np.zeros(u.shape)
        for s in surfaces:
            m = sid == s["id"]
            if not m.any():
                continue
            tu = (X[m] + 2100.0) / tex_res
            tv = (Y[m] + 2100.0) / tex_res
            tex = textures[s["id"]]
            iu = np.clip(tu, 0, tex.shape[1] - 2)
            iv = np.clip(tv, 0, tex.shape[0] - 2)
            u0, v0 = iu.astype(int), iv.astype(int)
            fu, fv = iu - u0, iv - v0
            val = (tex[v0, u0] * (1 - fu) * (1 - fv) + tex[v0, u0 + 1] * fu * (1 - fv)
                   + tex[v0 + 1, u0] * (1 - fu) * fv + tex[v0 + 1, u0 + 1] * fu * fv)
            shade = {0: 0.55, 1: 0.8, 2: 1.0, 3: 0.9}[s["id"]]
            img[m] = 25 + 210 * shade * val
        if ss > 1:
            img = img.reshape(height, ss, width, ss).mean(axis=(1, 3))
        return img, depth, sid

    left_f, _, _ = cast(0.0, supersample)
    right_f, _, _ = cast(baseline_mm, supersample)
    _, depth_l, sid_l = cast(0.0, 1)
    _, depth_r, _ = cast(baseline_mm, 1)
    left = np.clip(np.rint(left_f + rng.normal(0, noise_sigma, left_f.shape)), 0, 255).astype(np.uint8)
    right = np.clip(np.rint(right_f + rng.normal(0, noise_sigma, right_f.shape)), 0, 255).astype(np.uint8)

    gt_disp = focal_px * baseline_mm / depth_l
    # Visibility: does the left pixel's surface point appear un-occluded in the right view?
    ur = np.rint(np.arange(width)[None, :] - gt_disp).astype(int)
    in_frame = (ur >= 0) & (ur < width)
    rows = np.repeat(np.arange(height)[:, None], width, axis=1)
    zr = np.where(in_frame, depth_r[rows, np.clip(ur, 0, width - 1)], np.inf)
    visible = in_frame & (np.abs(zr - depth_l) <= 0.02 * depth_l)

    K = np.array([[focal_px, 0, cx], [0, focal_px, cy], [0, 0, 1]], dtype=np.float64)
    return {
        "left": left, "right": right,
        "gt_disparity_px": gt_disp, "gt_depth": depth_l, "surface_id": sid_l,
        "visible_in_right": visible, "out_of_frame": ~in_frame,
        "calibration": {"K1": K, "K2": K.copy(), "D1": np.zeros(5), "D2": np.zeros(5),
                        "R": np.eye(3), "T": np.array([-baseline_mm, 0.0, 0.0]), "baseline": baseline_mm,
                        "focal_px": focal_px, "units": "mm", "rectified": True,
                        "calibration_id": f"synthetic_pinhole_seed{seed}", "source": "synthetic (known by construction)"},
        "scene": {"surfaces": surfaces, "seed": seed, "texture_mm_per_cell": tex_res, "supersample": supersample,
                  "noise_sigma": noise_sigma, "units": "mm", "image_size_wh": [width, height]},
        "pair_id": f"synthetic_scene_seed{seed}",
        "provenance": "synthetic",
        "label": "SYNTHETIC algorithm demonstration - not a scalp reconstruction",
    }
