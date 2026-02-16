# Workbench API Reference

This document describes the public-facing helpers in `TERA/Workbench` (runtime entry points, result container, visualization, and reporting). It intentionally does not restate configuration keys; see `Docs/Configuration_Ref.md` for that.

## 1. Task Execution

### `TaskRunner.run`

File: `TERA/Workbench/Task.py`

Signature:

```python
TaskRunner.run(
    config: TaskConfig,
    print_results: bool = True,
    validate_results: bool = True,
    progress_bar: bool = False
) -> ReachResult
```

Behavior:

- Dispatches to the appropriate engine based on `config.system_type`.
- Tracks runtime and fills `ReachResult.runtime`.
- If `print_results=True`, emits a standardized console report.
- If `validate_results=True`, runs validation if expected bounds/width are present on the config.

System types:

- `continuous` → `TaskRunner._run_continuous`
- `hybrid` → `TaskRunner._run_hybrid`
- `stochastic` → `TaskRunner._run_stochastic`

### `TaskRunner` helper builders

File: `TERA/Workbench/Task.py`

These are internal helpers but are useful when you need to build initial TM vectors manually.

- `create_initial_box_continuous(dim: int, bounds: list, order: int, time_var: str = "t") -> TMVector`
- `create_initial_box_stochastic(state_dim: int, bounds: list, order: int, time_var: str = "t") -> TMVector`

## 2. Result Container

### `ReachResult`

File: `TERA/Workbench/Results.py`

Constructor:

```python
ReachResult(
    flowpipe,
    system_type,
    state_vars,
    config,
    status="SUCCESS",
    runtime=0.0,
    safety_status="SAFE"
)
```

Fields:

- `flowpipe`: list of segments produced by the solver.
- `system_type`: `"continuous" | "hybrid" | "stochastic"`.
- `state_vars`: symbolic variables in order.
- `config`: a dict (typically `vars(config)` from `TaskConfig`).
- `status`: `"SUCCESS" | "PARTIAL" | "FAILED" | "CRASHED"`.
- `runtime`: seconds (float).
- `timestamp`: `%Y-%m-%d %H:%M:%S`.
- `validation_data`: dict populated by MC validation.
- `safety_status`: `"SAFE" | "BREACHED" | "N/A"`.

Methods:

- `get_final_bounds()` → interval bounds for the last segment (uses `tmv` and `time_interval_abs` when available).
- `add_validation_traces(traces)` → stores traces under `validation_data["traces"]`.
- `export_to_csv(file_path)` → exports flowpipe bounds and metadata to CSV; returns the number of rows written.

Flowpipe segment fields used by reporting/visualization:

- `tmv`: Taylor model vector for the segment (preferred for plotting bounds).
- `time_interval_abs`: interval with `.lower/.upper`.
- `mode`: hybrid mode name (hybrid plots and summaries).
- `A_l`, `c_l`, `center`, `global_center`, `local_half_widths`, `r_l`: used for affine projections when `tmv` is not available.
- `stochastic_radius`: used for stochastic envelopes.

## 3. Reporting

### `Report`

File: `TERA/Workbench/Report.py`

Entry points:

- `print_header(result)`  
  Prints a standard header with system settings and run summary.

- `check_safety(result, unsafe_sets)`  
  Checks intersection of segments with user-provided unsafe sets. Returns `"SAFE"`, `"BREACHED"`, or `"N/A"`.

- `validate(result, tolerance=1e-4)`  
  Compares final bounds against `expected_final_bounds` and/or `expected_final_width`.

- `print_stochastic_stats(result)`  
  Prints deterministic bounds plus stochastic dilation and compares MC traces against the AMGF radius.

- `print_hybrid_summary(result)`  
  Prints mode path summary and total horizon.


Typical usage is automatic inside `TaskRunner.run`, but you can call these directly for custom workflows.

## 4. Visualization

### `Visualizer.plot`

File: `TERA/Workbench/Visualizer.py`

Signature:

```python
Visualizer.plot(
    result: ReachResult,
    dims=None,
    mode="phase",
    **kwargs
)
```

Defaults:

- `dims`: uses `engine_params.plot_dims` if present, else `(0, 1)`.
- `mode`: `"phase"` for state portraits, `"time"` for time evolution.

Common `kwargs`:

- `title`: plot title override.
- `figsize`: matplotlib figure size tuple.
- `save_path`: if provided, saves the figure.
- `edgecolor`, `facecolor`, `alpha`, `linewidth`: geometry styling for continuous/hybrid.
- `projection_mode`: `"aabb"` (default) or `"slice"` for linear runs.
- `use_tmv_bounds`: prefer `tmv.bound()` if available.
- `samples`: number of time samples for `left_right` nonlinear geometry.
- `mode_colors`: dict `{mode_name: color}` for hybrid plots.
- `filled`: fill mode polygons for hybrid/time plots.

Color options (applies to `edgecolor`, `facecolor`, and `mode_colors` values):

- Named colors (e.g., `"red"`, `"black"`, `"tab:blue"`, `"tab:orange"`).
- CSS4 color names (e.g., `"cornflowerblue"`, `"tomato"`).
- Hex strings (e.g., `"#1f77b4"` or `"#1f77b4cc"` with alpha).
- Grayscale strings in `[0, 1]` (e.g., `"0.2"`).
- RGB/RGBA tuples with floats in `[0, 1]` (e.g., `(0.1, 0.2, 0.5)` or `(0.1, 0.2, 0.5, 0.6)`).

Modes:

- `phase`: 2D projection of reachable sets.
- `time`: time evolution for `continuous` and `hybrid` only.

Stochastic plotting:

- Uses `TERA.Stochastic.Plotter.StochasticPlotter`.
- Consumes `validation_data["traces"]` if present.

Example:

```python
from TERA.Workbench.Visualizer import Visualizer

fig = Visualizer.plot(result, dims=(0, 1), mode="phase", title="Reachable Set")
```

## 5. Quick Integration Pattern

```python
from TERA.Workbench.Task import TaskRunner
from TERA.Workbench.Visualizer import Visualizer

result = TaskRunner.run(cfg, print_results=True, validate_results=True)
Visualizer.plot(result, dims=(0, 1), mode="phase")
```
