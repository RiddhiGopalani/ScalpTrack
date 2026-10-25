"""ScalpTrack: explainable scalp-condition classification and experimental hair measurement.

The package holds every reusable function used by the notebooks and scripts.
Notebooks only present results; the computation lives here so it can be tested.

This is an experimental image-analysis project. Nothing in it is a medical
diagnosis, and measurements are reported only in the units their inputs support.
"""

__version__ = "1.0.0"

from .schema import Status  # noqa: F401
