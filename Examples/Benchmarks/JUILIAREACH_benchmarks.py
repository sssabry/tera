"""JuliaReach ReachabilityAnalysis.jl benchmark task definitions."""

from typing import Dict
from sage.all import var, cos
import numpy as np

from TERA.TMCore.Interval import Interval
from TERA.Workbench.TaskConfig import TaskConfig

def get_juliareach_benchmarks() -> Dict[str, TaskConfig]:
    """Return JuliaReach ReachabilityAnalysis.jl benchmarks."""
    tasks: Dict[str, TaskConfig] = {}

    # Duffing Oscillator (TMJets21a)
    omega = 1.2
    T = 2 * np.pi / omega

    x, v, t = var("x v t")
    tasks["juliareach_cont_duffing"] = TaskConfig(
        name="Duffing Oscillator (TMJets21a)",
        system_type="continuous",
        vars=[x, v],
        f_expr=[
            v,
            x - 0.3 * v - x**3 + 0.37 * cos(omega * t),
        ],
        initial_set=[
            Interval(0.9, 1.1),
            Interval(-0.1, 0.1),
        ],
        time_horizon=20 * T,
        order=5,
        step_size=0.005,
        engine_params={
            "fixed_step_mode": False,
            "precondition_setup": "ID",
            'min_step': 1e-12
        },
    )

    return tasks
