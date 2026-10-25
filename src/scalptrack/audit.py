"""Dataset discovery, audit, manifests and leakage-safe grouped splits (spec Sections 4.3-4.4).

Workflow:
1. ``scan_source`` lists every file of a downloaded export (class folders or a Roboflow YOLO export).
2. ``inspect_files`` decodes each image: readability, size, channels, SHA-256, dHash under 8 flips/rotations.
3. ``duplicate_groups`` joins exact and near-duplicates (union-find) into ``duplicate_group_id``.
4. ``assign_splits`` keeps the publisher split only when no duplicate group crosses it; otherwise it
   re-splits ~70/15/15 by independent group, stratified by class.
5. ``build_manifest`` writes one row per original image (plus linked crop rows for detection exports).

Absent patient identity, physical scale and stereo relationships are left null - never guessed.
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageOps

from .config import display_path
from .io_utils import DEFAULT_EXTENSIONS, file_sha256

MANIFEST_COLUMNS = [
    "image_id",
    "original_id",
    "relative_path",
    "dataset_source",
    "dataset_version",
    "license",
    "download_date",
    "raw_label",
    "canonical_label",
    "task_type",
    "width_px",
    "height_px",
    "channels",
    "file_hash",
    "duplicate_group_id",
    "patient_id",
    "split",
    "publisher_split",
    "roi_id",
    "parent_image_id",
    "bbox_xyxy",
    "has_hair_instance_labels",
    "has_count_labels",
    "has_segmentation_masks",
    "pixel_scale_um_per_px",
    "stereo_pair_id",
    "calibration_id",
    "exclusion_reason",
    "notes",
]
SPLITS = ("train", "validation", "test")


class DatasetMissingError(FileNotFoundError):
    """Raised with actionable placement instructions when a dataset folder is absent or empty."""


def missing_dataset_message(source_name: str, source_cfg: dict, root: Path) -> str:
    """Actionable text telling the user where to download a source and how to lay out its folders."""
    d = root / source_cfg["dir"]
    if source_cfg.get("layout") == "yolo":
        layout = (
            f"  {d}\\data.yaml\n  {d}\\train\\images\\*.jpg   {d}\\train\\labels\\*.txt\n"
            f"  {d}\\valid\\images ...          {d}\\test\\images ..."
        )
    else:
        layout = (
            f"  {d}\\<optional wrapper folder>\\train\\<Class Name>\\*.jpg\n"
            f"  {d}\\<optional wrapper folder>\\val\\<Class Name>\\*.jpg\n"
            f"  {d}\\<optional wrapper folder>\\test\\<Class Name>\\*.jpg\n"
            f"  (or, without a publisher split: {d}\\<Class Name>\\*.jpg)"
        )
    return (
        f"Dataset '{source_name}' was not found or contains no images.\n"
        f"Download it from {source_cfg.get('url')} (license: {source_cfg.get('license')}) and extract it so that:\n"
        f"{layout}\nSee docs/dataset_setup.md for step-by-step instructions."
    )


# --------------------------------------------------------------------------- scanning


@dataclass
class ScanResult:
    files: pd.DataFrame  # one row per image file
    regions: pd.DataFrame  # one row per labelled box (detection exports only)
    other_files: list[str]  # non-image files found (reported, not used)
    class_names: list[str]
    layout: str


def _norm_split(name: str, aliases: dict) -> str | None:
    return aliases.get(name.lower())


def scan_source(source_cfg: dict, root: Path, extensions=DEFAULT_EXTENSIONS) -> ScanResult:
    """List images, raw labels and publisher splits for one source folder."""
    base = root / source_cfg["dir"]
    if not base.is_dir():
        raise DatasetMissingError(str(base))
    exts = {e.lower() for e in extensions}
    all_files = sorted(p for p in base.rglob("*") if p.is_file())
    images = [p for p in all_files if p.suffix.lower() in exts]
    others = [str(p.relative_to(base)) for p in all_files if p.suffix.lower() not in exts]
    if not images:
        raise DatasetMissingError(str(base))
    if source_cfg.get("layout") == "yolo":
        return _scan_yolo(base, images, others)
    aliases = source_cfg.get("split_aliases") or {
        "train": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test": "test",
    }
    rows = []
    for p in images:
        rel = p.relative_to(base)
        parts = rel.parts
        split, label = None, None
        # Find a split folder anywhere in the path; the label is the folder directly below it.
        for i, part in enumerate(parts[:-1]):
            s = _norm_split(part, aliases)
            if s is not None:
                split = s
                label = parts[i + 1] if i + 1 < len(parts) - 1 else None
                break
        if split is None:
            label = parts[-2] if len(parts) >= 2 else None
        rows.append({"relative_path": rel.as_posix(), "publisher_split": split, "raw_label": label})
    files = pd.DataFrame(rows)
    classes = sorted(files["raw_label"].dropna().unique().tolist())
    layout = "split_class_folders" if files["publisher_split"].notna().any() else "class_folders"
    return ScanResult(files, pd.DataFrame(), others, classes, layout)


def _scan_yolo(base: Path, images: list[Path], others: list[str]) -> ScanResult:
    yaml_path = next(iter(sorted(base.rglob("data.yaml"))), None)
    names: list[str] = []
    if yaml_path is not None:
        meta = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        n = meta.get("names", [])
        names = [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
    aliases = {"train": "train", "valid": "validation", "val": "validation", "test": "test"}
    rows, regions = [], []
    for p in images:
        rel = p.relative_to(base)
        split = next((aliases[q.lower()] for q in rel.parts if q.lower() in aliases), None)
        # Roboflow layout: <split>/images/x.jpg  <->  <split>/labels/x.txt
        label_path = None
        parts = list(rel.parts)
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            parts[idx] = "labels"
            label_path = base.joinpath(*parts).with_suffix(".txt")
        boxes = _read_yolo_labels(label_path) if label_path is not None and label_path.is_file() else []
        rows.append(
            {
                "relative_path": rel.as_posix(),
                "publisher_split": split,
                "raw_label": None,
                "n_boxes": len(boxes),
                "label_file": label_path.relative_to(base).as_posix()
                if label_path and label_path.is_file()
                else None,
            }
        )
        for k, (cls, box) in enumerate(boxes):
            regions.append(
                {
                    "relative_path": rel.as_posix(),
                    "roi_id": k,
                    "class_index": cls,
                    "raw_label": names[cls] if 0 <= cls < len(names) else f"class_{cls}",
                    "bbox_norm_cxcywh": box,
                }
            )
    files = pd.DataFrame(rows)
    # Detection exports have no single image label; images with exactly one distinct box label are noted.
    reg = pd.DataFrame(regions)
    return ScanResult(files, reg, others, names, "yolo")


def _read_yolo_labels(path: Path) -> list[tuple[int, list[float]]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        vals = line.split()
        if len(vals) < 5:
            continue
        cls = int(float(vals[0]))
        nums = [float(v) for v in vals[1:]]
        if len(nums) == 4:
            out.append((cls, nums))
        else:  # polygon export: convert the polygon to its bounding box
            xs, ys = nums[0::2], nums[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            out.append((cls, [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]))
    return out


# --------------------------------------------------------------------------- per-file inspection


def dhash_bits(gray_u8: np.ndarray, size: int = 8) -> np.uint64:
    """Difference hash: 64 bits of horizontal gradient sign on a (size+1) x size thumbnail."""
    img = Image.fromarray(gray_u8).resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def dihedral_dhashes(gray_u8: np.ndarray) -> list[int]:
    """dHash of the 8 flips/90-degree rotations (augmented copies often use these)."""
    out = []
    for k in range(4):
        r = np.rot90(gray_u8, k)
        out.append(int(dhash_bits(np.ascontiguousarray(r))))
        out.append(int(dhash_bits(np.ascontiguousarray(r[:, ::-1]))))
    return out


THUMB = 32


def thumbnail(gray_u8: np.ndarray, size: int = THUMB) -> np.ndarray:
    """Square uint8 thumbnail used to verify dHash candidate pairs."""
    return np.asarray(Image.fromarray(gray_u8).resize((size, size), Image.Resampling.BOX), dtype=np.uint8)


def _dihedral(a: np.ndarray) -> list[np.ndarray]:
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out += [r, r[:, ::-1]]
    return out


def thumbnail_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Max Pearson correlation between thumbnail ``a`` and the 8 flips/rotations of ``b``."""
    x = a.astype(np.float64).ravel()
    x = x - x.mean()
    best = -1.0
    for v in _dihedral(b):
        y = v.astype(np.float64).ravel()
        y = y - y.mean()
        den = np.sqrt((x * x).sum() * (y * y).sum())
        corr = 1.0 if den == 0 and np.array_equal(a, v) else (float((x * y).sum() / den) if den else 0.0)
        best = max(best, corr)
    return best


def inspect_file(path: Path, min_side_px: int = 32) -> dict:
    """Decode one file; return readability, geometry, channel info, hashes and simple statistics."""
    rec: dict = {"readable": False, "error": None}
    try:
        rec["file_size_bytes"] = path.stat().st_size
        rec["file_hash"] = file_sha256(path)
        with Image.open(path) as im:
            im.load()
            rec["format"], rec["mode"] = im.format, im.mode
            rec["channels"] = len(im.getbands())
            im = ImageOps.exif_transpose(im)
            rgb = np.asarray(im.convert("RGB"))
        h, w = rgb.shape[:2]
        gray = np.asarray(Image.fromarray(rgb).convert("L"))
        rec.update(
            readable=True,
            width_px=int(w),
            height_px=int(h),
            mean_r=float(rgb[..., 0].mean()),
            mean_g=float(rgb[..., 1].mean()),
            mean_b=float(rgb[..., 2].mean()),
            gray_mean=float(gray.mean()),
            gray_std=float(gray.std()),
            dark_fraction=float((gray <= 20).mean()),
            bright_fraction=float((gray >= 245).mean()),
            dhash_variants=dihedral_dhashes(gray),
            thumb=thumbnail(gray),
        )
        rec["too_small"] = bool(min(h, w) < min_side_px)
    except Exception as exc:  # noqa: BLE001 - every decode failure is recorded per file, not hidden
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def inspect_files(base: Path, rel_paths: list[str], min_side_px: int = 32, workers: int = 8) -> pd.DataFrame:
    """Run ``inspect_file`` on every relative path (threaded) and return one row per file."""
    paths = [base / r for r in rel_paths]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        recs = list(ex.map(lambda p: inspect_file(p, min_side_px), paths))
    df = pd.DataFrame(recs)
    df.insert(0, "relative_path", rel_paths)
    return df


# --------------------------------------------------------------------------- duplicates


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        """Return the representative (root) of ``i``'s group, compressing the path on the way."""
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return int(i)

    def union(self, a: int, b: int) -> None:
        """Merge the groups that contain ``a`` and ``b``."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_duplicate_pairs(variant_hashes: np.ndarray, threshold: int, chunk: int = 128) -> np.ndarray:
    """Pairs (i<j) whose base dHash is within ``threshold`` bits of any of j's 8 dihedral variants.

    ``variant_hashes`` has shape (n, 8), dtype uint64; column 0 is the untransformed image.
    """
    n = variant_hashes.shape[0]
    base = variant_hashes[:, 0]
    pairs = []
    for s in range(0, n, chunk):
        b = base[s : s + chunk, None, None]
        dist = np.bitwise_count(np.bitwise_xor(b, variant_hashes[None, :, :])).min(axis=2)
        ii, jj = np.nonzero(dist <= threshold)
        ii = ii + s
        keep = jj > ii
        if keep.any():
            pairs.append(np.stack([ii[keep], jj[keep]], axis=1))
    return np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=int)


def duplicate_groups(
    file_hashes: list[str | None],
    variant_hashes: np.ndarray | None,
    threshold: int,
    thumbs: list[np.ndarray] | None = None,
    min_corr: float = 0.9,
) -> tuple[np.ndarray, dict]:
    """Union exact (SHA-256) and near duplicates. Returns group index per row and stats.

    A near-duplicate needs a dHash match (<= ``threshold`` bits under some flip/rotation) AND,
    when thumbnails are given, a 32x32 thumbnail correlation >= ``min_corr`` under some
    flip/rotation. The second check rejects distinct images whose coarse gradient pattern
    happens to agree (common for repetitive hair textures).
    """
    n = len(file_hashes)
    uf = _UnionFind(n)
    first: dict[str, int] = {}
    n_exact = 0
    for i, h in enumerate(file_hashes):
        if h is None:
            continue
        if h in first:
            uf.union(first[h], i)
            n_exact += 1
        else:
            first[h] = i
    n_candidates, n_near = 0, 0
    if variant_hashes is not None and len(variant_hashes):
        pairs = near_duplicate_pairs(variant_hashes, threshold)
        n_candidates = len(pairs)
        for i, j in pairs:
            if thumbs is not None and thumbnail_correlation(thumbs[i], thumbs[j]) < min_corr:
                continue
            n_near += 1
            uf.union(int(i), int(j))
    roots = np.array([uf.find(i) for i in range(n)])
    _, group_idx = np.unique(roots, return_inverse=True)
    sizes = np.bincount(group_idx)
    return group_idx, {
        "exact_duplicate_files": n_exact,
        "dhash_candidate_pairs": n_candidates,
        "near_duplicate_pairs": n_near,
        "verification_min_corr": min_corr if thumbs is not None else None,
        "n_groups": int(len(sizes)),
        "n_groups_size_gt1": int((sizes > 1).sum()),
        "largest_group": int(sizes.max()) if len(sizes) else 0,
    }


# --------------------------------------------------------------------------- splits


def grouped_stratified_split(
    groups: pd.Series, labels: pd.Series, fractions=(0.7, 0.15, 0.15), seed: int = 42
) -> pd.Series:
    """Assign each *group* to train/validation/test, stratified by the group's label.

    Groups are shuffled per class and allocated greedily so the image counts per
    class approach the target fractions. All members of a group share a split.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must sum to 1")
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"g": groups.values, "y": labels.values}, index=groups.index)
    group_label = df.groupby("g")["y"].first()
    group_size = df.groupby("g").size()
    assign: dict = {}
    for y in sorted(group_label.unique()):
        gs = group_label.index[group_label == y].to_numpy()
        gs = gs[rng.permutation(len(gs))]
        total = group_size[gs].sum()
        targets = np.array(fractions) * total
        counts = np.zeros(3)
        for g in gs:
            # Put the group where the remaining deficit relative to target is largest.
            k = int(np.argmax((targets - counts) / np.maximum(targets, 1e-9)))
            assign[g] = SPLITS[k]
            counts[k] += group_size[g]
    return df["g"].map(assign)


def assign_splits(df: pd.DataFrame, policy: str, fractions, seed: int) -> tuple[pd.Series, dict]:
    """Return a split per row and a report of the decision (publisher split kept or regrouped)."""
    usable = df["exclusion_reason"].isna()
    report: dict = {"policy": policy}
    pub = df.loc[usable, "publisher_split"]
    crossing = df.loc[usable].groupby("duplicate_group_id")["publisher_split"].nunique() > 1
    report["groups_crossing_publisher_split"] = int(crossing.sum())
    report["images_in_crossing_groups"] = int(
        df.loc[usable, "duplicate_group_id"].isin(crossing.index[crossing]).sum()
    )
    has_pub = pub.notna().all() and len(pub) > 0
    split = pd.Series(None, index=df.index, dtype=object)
    if policy == "publisher_if_clean" and has_pub and report["groups_crossing_publisher_split"] == 0:
        split.loc[usable] = pub
        report["decision"] = "publisher split preserved (no duplicate group crosses splits)"
    else:
        split.loc[usable] = grouped_stratified_split(
            df.loc[usable, "duplicate_group_id"], df.loc[usable, "canonical_label"], fractions, seed
        )
        report["decision"] = "re-split by independent duplicate group: " + (
            "publisher split absent"
            if not has_pub
            else f"{report['groups_crossing_publisher_split']} duplicate groups cross the publisher split"
            if report["groups_crossing_publisher_split"]
            else "policy requested regrouping"
        )
    split.loc[~usable] = "excluded"
    return split, report


# --------------------------------------------------------------------------- manifest


def canonical_label(raw: str | None) -> str | None:
    """Lower-case snake_case version of a raw label. Raw names are preserved in ``raw_label``."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    return re.sub(r"[^0-9a-z]+", "_", str(raw).strip().lower()).strip("_") or None


def _image_id(source: str, rel: str) -> str:
    return f"{source}_{hashlib.sha1(rel.encode('utf-8')).hexdigest()[:12]}"


def read_source_info(base: Path) -> dict:
    """Optional ``SOURCE_INFO.json`` written by the download script (version, download date)."""
    p = base / "SOURCE_INFO.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def build_manifest(
    cfg: dict, source_name: str, workers: int = 8, progress: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Scan, inspect, deduplicate and split one source. Returns (manifest, audit report)."""
    root = Path(cfg["root"])
    scfg = cfg["datasets"]["sources"][source_name]
    base = root / scfg["dir"]
    try:
        scan = scan_source(scfg, root, cfg["image"]["supported_extensions"])
    except DatasetMissingError as exc:
        raise DatasetMissingError(missing_dataset_message(source_name, scfg, root)) from exc
    info = read_source_info(base)
    if progress:
        print(f"[audit] {source_name}: {len(scan.files)} image files found; inspecting ...")
    insp = inspect_files(base, scan.files["relative_path"].tolist(), cfg["image"]["min_side_px"], workers)
    df = scan.files.merge(insp, on="relative_path", how="left")

    df["exclusion_reason"] = None
    df.loc[~df["readable"], "exclusion_reason"] = "unreadable_or_corrupt"
    df.loc[df["readable"] & df["too_small"].fillna(False).astype(bool), "exclusion_reason"] = "below_min_side"
    task_type = scfg.get("task_type", "image_classification")
    if scan.layout != "yolo":
        df.loc[df["raw_label"].isna() & df["exclusion_reason"].isna(), "exclusion_reason"] = "missing_label"

    readable = df["readable"].fillna(False).to_numpy()
    variants = np.zeros((len(df), 8), dtype=np.uint64)
    for i, v in enumerate(df["dhash_variants"]):
        if isinstance(v, list):
            variants[i] = np.array(v, dtype=np.uint64)
    idx = np.nonzero(readable)[0]
    gidx, dup_stats = duplicate_groups(
        [df["file_hash"].iloc[i] for i in idx],
        variants[idx],
        int(cfg["classification"]["near_duplicate_hamming"]),
        thumbs=[df["thumb"].iloc[i] for i in idx],
        min_corr=float(cfg["classification"].get("near_duplicate_min_corr", 0.9)),
    )
    df["duplicate_group_id"] = None
    df.loc[df.index[idx], "duplicate_group_id"] = [f"{source_name}_g{g:05d}" for g in gidx]
    df.loc[df["duplicate_group_id"].isna(), "duplicate_group_id"] = [
        f"{source_name}_unreadable_{i}" for i in range(int(df["duplicate_group_id"].isna().sum()))
    ]

    df["dataset_source"] = source_name
    df["dataset_version"] = info.get("version", scfg.get("version"))
    df["license"] = scfg.get("license")
    df["download_date"] = info.get("download_date")
    df["canonical_label"] = df["raw_label"].map(canonical_label)
    df["image_id"] = [_image_id(source_name, r) for r in df["relative_path"]]
    df["original_id"] = df["image_id"]
    # In detection exports the originals carry boxes, not a class; their crops carry the task.
    df["task_type"] = "detection_source_image" if scan.layout == "yolo" else task_type
    df["patient_id"] = None
    df["roi_id"] = None
    df["parent_image_id"] = None
    df["bbox_xyxy"] = None
    df["has_hair_instance_labels"] = False
    df["has_count_labels"] = False
    df["has_segmentation_masks"] = False
    df["pixel_scale_um_per_px"] = None
    df["stereo_pair_id"] = None
    df["calibration_id"] = None
    df["notes"] = None

    # Label conflicts inside one duplicate group mean the same picture carries two labels.
    if scan.layout != "yolo":
        conflict = (
            df[df["exclusion_reason"].isna()].groupby("duplicate_group_id")["canonical_label"].nunique()
        )
        bad = conflict.index[conflict > 1]
        df.loc[df["duplicate_group_id"].isin(bad) & df["exclusion_reason"].isna(), "exclusion_reason"] = (
            "duplicate_group_label_conflict"
        )
        split, split_report = assign_splits(
            df, cfg["classification"]["split_policy"], cfg["classification"]["split_fractions"], cfg["seed"]
        )
        df["split"] = split
    else:
        df["split"] = None
        split_report = {}

    region_df = pd.DataFrame()
    if scan.layout == "yolo" and len(scan.regions):
        region_df = _region_rows(df, scan.regions, source_name)
        region_df["exclusion_reason"] = None
        split, split_report = assign_splits(
            region_df.assign(publisher_split=region_df["publisher_split"]),
            cfg["classification"]["split_policy"],
            cfg["classification"]["split_fractions"],
            cfg["seed"],
        )
        region_df["split"] = split
        # Every crop of one original lives in that original's split (enforced by shared duplicate_group_id).
        parent_split = region_df.groupby("parent_image_id")["split"].first()
        df["split"] = df["image_id"].map(parent_split)
        df.loc[df["split"].isna() & df["exclusion_reason"].isna(), "exclusion_reason"] = "no_usable_boxes"
        df["split"] = df["split"].fillna("excluded")
        df["notes"] = "original image; classification uses its linked region crops"

    manifest = pd.concat([df, region_df], ignore_index=True) if len(region_df) else df
    report = {
        "source": source_name,
        "source_dir": display_path(base, root),
        "layout": scan.layout,
        "n_image_files": int(len(df)),
        "n_other_files": len(scan.other_files),
        "other_files_examples": scan.other_files[:10],
        "class_names_raw": scan.class_names,
        "n_unreadable": int((~df["readable"].fillna(False)).sum()),
        "unreadable_examples": df.loc[~df["readable"].fillna(False), ["relative_path", "error"]]
        .head(10)
        .to_dict("records"),
        "n_missing_label": int((df["exclusion_reason"] == "missing_label").sum()),
        "n_label_conflict": int((df["exclusion_reason"] == "duplicate_group_label_conflict").sum()),
        "duplicates": dup_stats,
        "split": split_report,
        "source_info": info,
    }
    return manifest, report


def _region_rows(images: pd.DataFrame, regions: pd.DataFrame, source: str) -> pd.DataFrame:
    """Derived crop rows (one per box) linked to their parent original via ``parent_image_id``."""
    parent = images.set_index("relative_path")
    rows = []
    for r in regions.itertuples(index=False):
        p = parent.loc[r.relative_path]
        if not p["readable"]:
            continue
        cx, cy, bw, bh = r.bbox_norm_cxcywh
        W, H = p["width_px"], p["height_px"]
        x0, y0 = int(max(0, np.floor((cx - bw / 2) * W))), int(max(0, np.floor((cy - bh / 2) * H)))
        x1, y1 = int(min(W, np.ceil((cx + bw / 2) * W))), int(min(H, np.ceil((cy + bh / 2) * H)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        rows.append(
            {
                "image_id": f"{p['image_id']}_r{r.roi_id}",
                "original_id": p["image_id"],
                "relative_path": r.relative_path,
                "dataset_source": source,
                "dataset_version": p["dataset_version"],
                "license": p["license"],
                "download_date": p["download_date"],
                "raw_label": r.raw_label,
                "canonical_label": canonical_label(r.raw_label),
                "task_type": "region_classification",
                "width_px": x1 - x0,
                "height_px": y1 - y0,
                "channels": p["channels"],
                "file_hash": p["file_hash"],
                "duplicate_group_id": p["duplicate_group_id"],
                "patient_id": None,
                "publisher_split": p["publisher_split"],
                "roi_id": int(r.roi_id),
                "parent_image_id": p["image_id"],
                "bbox_xyxy": [x0, y0, x1, y1],
                "has_hair_instance_labels": False,
                "has_count_labels": False,
                "has_segmentation_masks": False,
                "pixel_scale_um_per_px": None,
                "stereo_pair_id": None,
                "calibration_id": None,
                "readable": True,
                "notes": "ground-truth box crop (annotated-region classification)",
            }
        )
    return pd.DataFrame(rows)


def save_manifest(manifest: pd.DataFrame, report: dict, out_dir: Path) -> dict[str, Path]:
    """Write the manifest CSV (spec columns first), the split manifest and the audit report JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    src = report["source"]
    cols = MANIFEST_COLUMNS + [
        c for c in manifest.columns if c not in MANIFEST_COLUMNS and c not in ("dhash_variants", "thumb")
    ]
    m = manifest.copy()
    m["bbox_xyxy"] = m["bbox_xyxy"].map(lambda v: json.dumps(v) if isinstance(v, list) else None)
    paths = {
        "manifest": out_dir / f"{src}_manifest.csv",
        "splits": out_dir / f"{src}_splits.csv",
        "report": out_dir / f"{src}_audit_report.json",
    }
    m[cols].to_csv(paths["manifest"], index=False)
    m[["image_id", "original_id", "duplicate_group_id", "canonical_label", "split"]].to_csv(
        paths["splits"], index=False
    )
    paths["report"].write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return paths


def load_manifest(path: Path) -> pd.DataFrame:
    """Read a manifest CSV written by ``save_manifest`` (``bbox_xyxy`` is decoded back to lists)."""
    df = pd.read_csv(path)
    df["bbox_xyxy"] = df["bbox_xyxy"].map(lambda v: json.loads(v) if isinstance(v, str) else None)
    return df


def split_count_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Images and independent groups per class and split (what the classification report must state)."""
    m = manifest[manifest["split"].isin(SPLITS)]
    imgs = m.pivot_table(
        index="canonical_label", columns="split", values="image_id", aggfunc="count", fill_value=0
    )
    grps = m.pivot_table(
        index="canonical_label", columns="split", values="duplicate_group_id", aggfunc="nunique", fill_value=0
    )
    imgs.columns = [f"{c}_images" for c in imgs.columns]
    grps.columns = [f"{c}_groups" for c in grps.columns]
    return imgs.join(grps)


def capability_audit(manifest: pd.DataFrame | None) -> pd.DataFrame:
    """What the data can and cannot support (spec Sections 1.3, 15.3)."""
    have = manifest is not None and len(manifest) > 0

    def any_true(col):
        """True if any manifest row has a truthy value in ``col``."""
        return bool(have and manifest[col].fillna(False).astype(bool).any())

    rows = [
        (
            "condition classification labels",
            "available" if have and manifest["canonical_label"].notna().any() else "unavailable",
            "publisher folder/box labels" if have else "dataset not downloaded",
        ),
        (
            "independent patient identity",
            "unavailable",
            "no patient_id supplied by the source; grouping is by duplicate cluster",
        ),
        (
            "hair instance / count labels",
            "available" if any_true("has_count_labels") else "unavailable",
            "no per-hair annotations in the export" if have else "dataset not downloaded",
        ),
        (
            "segmentation masks",
            "available" if any_true("has_segmentation_masks") else "unavailable",
            "none in export",
        ),
        ("physical pixel scale (um/px)", "unavailable", "no documented scale; widths stay in pixels"),
        (
            "calibrated stereo pairs",
            "unavailable",
            "no verified pair/calibration; synthetic stereo demo only",
        ),
        (
            "face-containing images (Haar positive case)",
            "to be checked in notebook 07",
            "requires running the detector over the downloaded images",
        ),
    ]
    return pd.DataFrame(rows, columns=["capability", "status", "reason"])
