"""Engine for the NV Tutorial notebooks.

Thin, reusable layer over `core.protocols` + `core.plot_funcs` that:
- builds the per-model initial state / measurement operators,
- runs each protocol (Energy Levels, Populations, Photoluminescence, ODMR, Ramsey),
- caches the plot-relevant arrays under `examples/RawData/example_<dataset>/<dataset>.npy`
  (so notebooks can reproduce plots without re-running the slow sweeps), and
- saves figures (dual dark/light theme) into both the presentation `figures/` and
  `examples/Figures/`.

The `Tutorials/Tutorial - *.ipynb` notebooks are thin callers of these functions, toggled by
a top-of-notebook `RECOMPUTE` flag. Full-resolution sweeps are heavy; pass small sizes /
`n_points` for quick runs.

Split into ``_infra`` (registry / caching / saving / state), ``compute`` (the ``compute_*``
sweeps) and ``plots`` (the ``plot_*_result`` wrappers); everything is re-exported here so
``from examples import protocol_runs as engine`` keeps working unchanged.

UNITS: frequencies (Rabi, pump, drive, ZFS, hyperfine) in MHz; durations and sweep times in
microseconds (us); magnetic field in Gauss (G); orientation angles in radians (via B0).
"""
# ruff: noqa: F403

from ._infra import *
from .compute import *
from .plots import *
