"""Result statuses and the measurement record used across modules (spec Section 16).

Unknown or unavailable values are ``None`` (JSON null), never zero: zero means a
valid measurement found nothing.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

SCHEMA_VERSION = "1.0"


class Status(str, Enum):
    OK = "ok"  # computed under its stated prerequisites (not clinical validity)
    EXPLORATORY = "exploratory"  # provisional result without adequate real ground truth
    SYNTHETIC_DEMO = "synthetic_demo"
    UNAVAILABLE = "unavailable"  # required data / calibration / quality absent
    FAILED = "failed"  # runtime or validation failure
    NOT_REQUESTED = "not_requested"

    def __str__(self) -> str:  # so f-strings print "ok" rather than "Status.OK"
        return self.value


def measurement(
    value: Any,
    unit: str | None,
    status: Status,
    method: str | None,
    reason: str,
    **extra: Any,
) -> dict[str, Any]:
    """Build the standard measurement record: value, unit, status, method, reason."""
    record = {"value": value, "unit": unit, "status": str(status), "method": method, "reason": reason}
    record.update(extra)
    return record


def empty_result(image_id: str | None = None) -> dict[str, Any]:
    """Skeleton analysis record matching the spec's schema illustration."""
    return {
        "schema_version": SCHEMA_VERSION,
        "image_id": image_id,
        "source": None,
        "roi_xyxy": None,
        "input_hash": None,
        "run_config_id": None,
        "quality": {"flags": [], "measurement_eligible": None},
        "classification": {
            "task_mode": None,
            "predicted_label": None,
            "class_scores": {},
            "model_id": None,
            "status": str(Status.NOT_REQUESTED),
            "reason": "Branch not run",
        },
        "hair_count": measurement(None, "candidates", Status.NOT_REQUESTED, None, "Branch not run"),
        "shaft_width": {
            "median_px": None,
            "iqr_px": None,
            "valid_samples": 0,
            "median_um": None,
            "pixel_scale_um_per_px": None,
            "status": str(Status.NOT_REQUESTED),
            "physical_status": str(Status.NOT_REQUESTED),
            "method": None,
            "reason": "Branch not run",
        },
        "stereo": {
            "pair_id": None,
            "calibration_id": None,
            "disparity_unit": "px",
            "depth_unit": None,
            "valid_coverage": None,
            "status": str(Status.UNAVAILABLE),
            "reason": "No verified stereo pair/calibration supplied with a single image",
        },
        "faces": {"boxes_xywh": [], "status": str(Status.NOT_REQUESTED), "reason": "Branch not run"},
        "coverage": {},
        "texture_features": {},
        "artifacts": {},
        "timings_s": {},
    }
