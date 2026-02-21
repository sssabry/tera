# Configuration Reference

This page documents TERA’s configuration surface for continuous, hybrid, and stochastic reachability runs.

TERA configuration is provided primarily via a `TaskConfig` object plus a nested `engine_params` dictionary for engine-specific controls.

Note on defaults: defaults are currently split across `TaskConfig`, `TaskRunner`, and engine constructors. If you see conflicting defaults (e.g., QR vs ID; fixed vs adaptive), prefer what `TaskRunner` passes into the engine for the run mode you are using. See `TERA/Workbench/Task.py`.

---

## 1. Configuration Layers

### 1.1 `TaskConfig` (top-level run config)
`TaskConfig` defines the model, initial set, and base integrator settings.  
Source: `TERA/Workbench/TaskConfig.py`

Minimal skeleton:

```python
cfg = TaskConfig(
    name="My System",
    system_type="continuous",  # "continuous" | "hybrid" | "stochastic"
    vars=[...],                # state variable ordering
    f_expr=[...],              # drift (ODE) expressions (continuous/stochastic)
    initial_set=[...],         # list of Interval boxes
    time_horizon=1.0,
    order=4,
    step_size=0.05,
    engine_params={...},       # engine-specific options
)
```

### 1.2 `engine_params` (engine-specific options)

`engine_params` is a dict living inside `TaskConfig`. It contains parameters for:

- preconditioning and step control (continuous/hybrid/stochastic)
- hybrid automata (guards/resets, jump limits)
- stochastic parameters ($\delta$, diffusion `G(x)`)
- validation options (Monte Carlo diagnostics)

Source: `TERA/Workbench/Task.py` and engine constructors.

---

## 2. Core `TaskConfig` Fields

These keys are common across run modes unless noted.

| Key                     | Type | Default | Meaning |
| ----------------------- | ---: | ------: | ------- |
| `name`                  | `str` | required | Label used in logs/reports. |
| `system_type`           | `str` | required | Dispatch target: `"continuous"`, `"hybrid"`, `"stochastic"`. |
| `vars`                  | `list` | required | State variable ordering. Must match the order used in expressions and intervals. |
| `f_expr`                | `list` or `None` | `None` | Drift vector field (f(x)). Required for continuous/stochastic. |
| `initial_set`           | `list[Interval]` | `[]` | Initial state box. Length must equal `len(vars)`. |
| `initial_mode`          | `str` or `None` | `None` | Initial discrete mode (hybrid only). |
| `time_horizon`          | `float` | `1.0` | Global horizon (T). |
| `order`                 | `int` | `4` | Taylor model order used in propagation. |
| `step_size`             | `float` | `0.05` | Base integration step size. Used as default for `min_step/max_step` if not overridden. |
| `time_var`              | `str` | `"t"` | Symbol name used internally for time. |
| `remainder_estimation`  | `list[Interval]` or `None` | `None` | Initial remainder guess (heuristic). Note: not all algorithms rely on this. |
| `unsafe_sets`           | `list[dict]` or `None` | `None` | Safety specs used only in post-check/reporting. |
| `urgent_jumps_mode`     | `bool` | `True` | Hybrid urgent-jump handling. See Hybrid section. |
| `expected_final_bounds` | `list[Interval]` or `None` | `None` | Optional validation target for regression tests or benchmark comparisons. |
| `expected_final_width`  | `float` or `None` | `None` | Optional validation target for regression tests or benchmark comparisons. |

---

## 3. Continuous Reachability (`system_type="continuous"`)

### 3.1 Mode selection: `engine_params.setting`

| Value | Meaning |
| --- | --- |
| `"single_step"` | Local TM step propagation (validated single-step integration). |
| `"left_right"` | Left–right compositional propagation (shrink-wrapping style). |

Key: `engine_params.setting`  
Used by `TaskRunner._run_continuous` → `TMReach`.

### 3.2 Preconditioning: `engine_params.precondition_setup`
| Value | Meaning |
| --- | --- |
| `"QR"` | QR-type preconditioning (affine change of coordinates / orthogonalization). |
| `"ID"` | Identity / minimal preconditioning; may include stagnation heuristics if enabled. |


Key: `engine_params.precondition_setup`  
Used by `TMReach` and preconditioning components.

### 3.3 Step control (fixed vs adaptive)

| Key | Type | Default (TMReach) | Meaning |
| --- | ---: | ---: | --- |
| `engine_params.fixed_step_mode` | bool | `True` | If true, disables adaptive step logic (uses constant `step_size`). |
| `engine_params.min_step` | float | `1e-6` | Lower bound on adaptive step. |
| `engine_params.max_step` | float | `0.5` | Upper bound on adaptive step. |
| `engine_params.step_grow_width_cap` | float | `1e-2` (if None, continuous) | Caps how aggressively step size grows when widths are small. |
| `engine_params.shrink_wrap_mode` | bool | `False` | Enable Bünger-corrected shrink wrapping in left-right mode (continuous/stochastic). |
| `engine_params.verbose` | bool | `False` | Enable verbose diagnostics (shrink wrapping and invariant checks). |

### 3.4 Algebra / verification loop controls

| Key | Type | Default | Meaning |
| --- | ---: | ---: | --- |
| `engine_params.cutoff_threshold` | float | `1e-10` | Coefficient sweep cutoff for polynomial/TM simplification. |
| `engine_params.max_iterations` | int | `40` | Iteration cap in verification/refinement loops. |

### 3.5 ID-stagnation heuristics (when using `"preconditioning_setup":"ID"`)

These tune stagnation detection and scaling:

| Key | Type | Default (hybrid run) | Meaning |
| --- | ---: | ---: | --- |
| `engine_params.id_preserve_coupling_on_stagnation` | bool | `True` | Preserve off-diagonal coupling when stagnation is detected. |
| `engine_params.id_stag_c_tol` | float | `1e-12` | Stagnation tolerance on coupling metric. |
| `engine_params.id_stag_s_tol` | float | `1e-10` | Stagnation tolerance on singular values. |
| `engine_params.id_stag_od_rel_tol` | float | `1e-6` | Relative tolerance on off-diagonal stagnation. |
| `engine_params.id_stag_lambda` | float | `1.0` | Scaling factor for stagnation response. |
| `engine_params.hybrid_id_full_linear_on_stagnation` | bool | `False` | Use full linearization when stagnation is detected. |

These primarily affect runtime/tightness. Misconfiguration can cause blow-up (over-approximation) but should not produce under-approximation.

---

## 4. Hybrid Reachability (`system_type="hybrid"`)

Hybrid reachability requires a hybrid automaton definition in `engine_params.automaton`.

### 4.1 Required: `engine_params.automaton`

Key: `engine_params.automaton`  
Type: `HybridAutomaton` (see below)

Used by `TaskRunner._run_hybrid` → `HybridReach` orchestration and `ModeSolver` continuous evolution per mode.

### 4.2 Jump limits and urgent behavior

| Key | Type | Default | Meaning |
| --- | ---: | ---: | --- |
| `engine_params.max_jumps` | int | `10` (hybrid run) | Max number of discrete transitions before stopping. |
| `urgent_jumps_mode` | bool | `True` (TaskConfig) | Enables urgent processing of enabled transitions. |

Soundness note: limiting jumps (`max_jumps`) can truncate reachable behavior and therefore reduce coverage of the true reach set after that limit.

### 4.3 Guard intersection and aggregation

| Key | Type | Default | Meaning |
| --- | ---: | ---: | --- |
| `engine_params.intersection_method` | str | `"combined"` | Guard intersection strategy. |
| `engine_params.aggregation_method` | str | `"PCA"` | Aggregates multiple intersections / segments. |
| `aggregation_threshold` | int | `10` | Number of segments before aggregation triggers (HybridReach internal). |


### 4.4 Hybrid continuous solver parameters (`ModeSolver`)

Hybrid mode propagation uses a configuration set very similar to continuous `TMReach`:

- `order`, `cutoff_threshold`, `max_iterations`
- `precondition_setup`, `min_step`, `max_step`, `fixed_step_mode`, `step_grow_width_cap`
- stagnation heuristics (ID-based)

Known limitation: `engine_params.setting` is effectively forced to `"single_step"` inside `ModeSolver` (it warns if configured otherwise). Left-right mode inside hybrid modes is currently a work in progress.

Hybrid default note: `step_grow_width_cap` defaults to `1.0` in hybrid runs (via `ModeSolver`), even though continuous defaults to `1e-2` when `None`.

### 4.5 Hybrid model objects (Automaton / Mode / Transition)

The hybrid system is represented using the concrete classes in `TERA/Hybrid/HybridModel.py`:

#### `Condition`

Represents invariants/guards as inequalities `g_i(x) <= 0`.

- `constraints`: list of symbolic expressions (e.g., `[x - 5, y + x]`)

#### `ResetMap`

Represents the reset map `x' = r(x)`.

- `mapping`: dict from variable name to expression (missing variables imply identity)

#### `Mode`

- `name`: mode identifier (string)
- `ode_exprs`: drift `f(x)` (list of expressions)
- `invariant`: `Condition`
- `transitions`: list of outgoing `Transition` objects (populated on the mode)

#### `Transition`

- `source`: `Mode`
- `target`: `Mode`
- `guard`: `Condition`
- `reset`: `ResetMap`
- `label`: optional string

#### `HybridAutomaton`

- `modes`: dict keyed by mode name (constructed from a list of `Mode`)
- `state_vars`: list of symbolic variables
- `time_var`: name of time variable (string)
- `initial_sets`: list of `(mode_name, TMVector)` pairs
- `add_initial_state(mode_name, tmv)`
- `get_mode(name)`

Populated examples can be found in `Examples/Benchmarks/FLOWSTAR_benchmarks.py` (e.g., bouncing ball and spiking neuron), which match the exact constructor shapes above.

Operational semantics (TERA):

1. Propagate flowpipe in current mode with `ModeSolver`.
2. Intersect flowpipe segment with each outgoing transition guard (`intersection_method`).
3. Apply reset map to the intersected set to create new initial set(s) for target mode.
4. Enqueue new mode tasks in `HybridReach` (worklist), bounded by `max_jumps` and global horizon.

---

## 5. Stochastic Reachability (`system_type="stochastic"`)

Stochastic reachability uses:

- deterministic TM flowpipe for the drift (f(x))
- a diffusion specification (G(x))
- a probability tolerance `delta` to form $\delta$-probabilistic enclosures

### 5.1 Required stochastic keys

| Key | Type | Default | Meaning |
| --- | ---: | ---: | --- |
| `engine_params.delta` | float | `0.001` | Probability budget (failure probability). |
| `engine_params.g_expr` | list | required | Diffusion matrix definition (G(x)). |

### 5.2 Optional stochastic keys

| Key | Type | Default | Meaning |
| --- | ---: | ---: | --- |
| `engine_params.P_matrix` | matrix or None | None | Weighted norm parameter for bounds/visualization. |
| `engine_params.amgf_eps` | float or None | None | Fixes AMGF epsilon in (0,1) if provided. |

### 5.3 Monte Carlo validation (diagnostics only)

| Key | Type | Default | Meaning |
| --- | ---: | ---: | --- |
| `engine_params.mc_traces` | int | `1000` | Number of sample trajectories. |
| `engine_params.mc_dt` | float or None | None | Override simulation step. |
| `engine_params.mc_seed` | int or None | None | RNG seed for reproducibility. |

---

## 6. Worked Examples

These are real snippets taken from `Examples/Benchmarks/*` (with only surrounding imports trimmed).

### 6.1 Lotka-Volterra (Flow*)

```python
from sage.all import var
from TERA.TMCore.Interval import Interval
from TERA.Workbench.TaskConfig import TaskConfig

x, y = var("x y")
cfg = TaskConfig(
    name="Lotka-Volterra (Example 3.3.11)",
    system_type="continuous",
    vars=[x, y],
    f_expr=[1.5 * x - x * y, -3 * y + x * y],
    initial_set=[Interval(4.95, 5.05), Interval(1.95, 2.05)],
    time_horizon=4.0,
    order=5,
    step_size=0.01,
    time_var="t_aug",
    remainder_estimation=[Interval(-1e-3, 1e-3)] * 2,
    engine_params={"precondition_setup": "ID", "fixed_step_mode": True},
    expected_final_bounds=[
        Interval(1.611473542017, 1.688426503140),
        Interval(2.043880403362, 2.136095251238),
    ],
)
```

### 6.2 Bouncing Ball (Flow* hybrid)

```python
from sage.all import var
from TERA.TMCore.Interval import Interval
from TERA.Workbench.TaskConfig import TaskConfig
from TERA.Hybrid.HybridModel import HybridAutomaton, Mode, Condition, ResetMap, Transition

def build_bouncing_ball_automaton():
    x, v = var("x v")
    g = 9.81

    down_inv = Condition(constraints=[-x, v])  # x >= 0, v <= 0
    down_mode = Mode("down", [v, -g + 0.1 * v**2], down_inv)

    up_inv = Condition(constraints=[-x, -v])  # x >= 0, v >= 0
    up_mode = Mode("up", [v, -g - 0.1 * v**2], up_inv)

    bounce_guard = Condition(constraints=[x])
    bounce_reset = ResetMap(mapping={"v": -0.8 * v})
    down_mode.transitions.append(
        Transition(down_mode, up_mode, bounce_guard, bounce_reset, "alpha")
    )

    apex_guard = Condition(constraints=[v])
    up_mode.transitions.append(
        Transition(up_mode, down_mode, apex_guard, ResetMap({}), "beta")
    )

    return HybridAutomaton([down_mode, up_mode], [x, v], "t")

cfg = TaskConfig(
    name="Bouncing Ball",
    system_type="hybrid",
    vars=[var("x"), var("v")],
    initial_set=[Interval(4.9, 5.1), Interval(-0.2, 0.0)],
    initial_mode="down",
    time_horizon=5,
    order=5,
    urgent_jumps_mode=True,
    step_size=0.01,
    engine_params={
        "automaton": build_bouncing_ball_automaton(),
        "max_jumps": 20,
        "intersection_method": "domain_contraction",
        "aggregation_method": "PCA",
        "fixed_step_mode": False,
        "precondition_setup": "ID",
        "setting": "single_step",
        "cutoff_threshold": 1e-12,
        "max_iterations": 40,
    },
)
```

### 6.3 Quadratic Model (INTLAB)

```python
from sage.all import SR
from TERA.TMCore.Interval import Interval
from TERA.Workbench.TaskConfig import TaskConfig

y1, y2 = SR.var("y1, y2")
cfg = TaskConfig(
    name="Quadratic Model (INTLAB demo Example IV)",
    system_type="continuous",
    vars=[y1, y2],
    f_expr=[y2, y1**2],
    initial_set=[Interval(0.95, 1.05), Interval(-1.05, -0.95)],
    time_horizon=6.0,
    order=10,
    step_size=0.02,
    remainder_estimation=[Interval(-1e-5, 1e-5)] * 2,
    engine_params={
        "fixed_step_mode": False,
        "setting": "left_right",
        "precondition_setup": "QR",
    },
    expected_final_bounds=[
        Interval(-0.2326613072710758, 1.030222247610039),
        Interval(0.3497621000683073, 1.122446571618336),
    ],
)
```

### 6.4 Laub-Loomis (ARCH)

```python
from sage.all import var
from TERA.TMCore.Interval import Interval
from TERA.Workbench.TaskConfig import TaskConfig

ll_vars = list(var("x1 x2 x3 x4 x5 x6 x7"))
x1, x2, x3, x4, x5, x6, x7 = ll_vars
cfg = TaskConfig(
    name="Laub-Loomis (ARCH 2017/2018)",
    system_type="continuous",
    vars=ll_vars,
    f_expr=[
        1.4 * x3 - 0.9 * x1,
        2.5 * x5 - 1.5 * x2,
        0.6 * x7 - 0.8 * x2 * x3,
        2 - 1.3 * x3 * x4,
        0.7 * x1 - x4 * x5,
        0.3 * x1 - 3.1 * x6,
        1.8 * x6 - 1.5 * x2 * x7,
    ],
    initial_set=[
        Interval(1.195, 1.205),
        Interval(1.045, 1.055),
        Interval(1.495, 1.505),
        Interval(2.395, 2.405),
        Interval(0.995, 1.005),
        Interval(0.095, 0.105),
        Interval(0.445, 0.455),
    ],
    time_horizon=20.0,
    order=4,
    step_size=0.05,
    remainder_estimation=[Interval(-1e-4, 1e-4)] * 7,
    engine_params={"precondition_setup": "ID"},
    expected_final_width=0.01044,
    unsafe_sets=[{"dims": {3: Interval(4.5, float("inf"))}, "start_time": 0.0}],
)
```

### 6.5 Anaesthesia (ARCH stochastic)

```python
import numpy as np
from sage.all import var
from TERA.TMCore.Interval import Interval
from TERA.Workbench.TaskConfig import TaskConfig

x1, x2, x3 = list(var("x1 x2 x3"))
f_anes = [
    -0.00904 * x1 + 0.00170 * x2 + 0.00063 * x3,
    0.00082 * x1 - 0.00089 * x2 + 0.000005 * x3,
    0.000045 * x1 + 0.000001 * x2 - 0.000055 * x3,
]
cfg = TaskConfig(
    name="Anaesthesia (ARCH 2018)",
    system_type="stochastic",
    vars=[x1, x2, x3],
    f_expr=f_anes,
    initial_set=[Interval(1.0, 6.0), Interval(0.0, 10.0), Interval(5.0, 5.0)],
    time_horizon=10.0,
    order=4,
    step_size=0.1,
    engine_params={
        "g_expr": np.diag([np.sqrt(1e-3)] * 3).tolist(),
        "delta": 0.01,
        "plot_dims": [0, 1],
        "fixed_step_mode": False,
        "precondition_setup": "ID",
        "mc_traces": 1000,
        "mc_dt": 0.001,
        "mc_seed": 0,
    },
    expected_final_bounds=[Interval(1, 6), Interval(0, 10), Interval(0, 10)],
)
```
