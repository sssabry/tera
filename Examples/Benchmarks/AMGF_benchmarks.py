"""AMGF benchmark task definitions."""

from typing import Dict
import math
from sage.all import SR, var, sin, cos

from TERA.TMCore.Interval import Interval
from TERA.Workbench.TaskConfig import TaskConfig


def get_AMGF_benchmarks() -> Dict[str, TaskConfig]:
    """
    benchmarks from Jafarpour et al TAC paper "Probabilisitc Reachability Analysis for Stochastic Control Systems"
    used to validate AMGF implementation
    """
    benchmarks: Dict[str, TaskConfig] = {}

    # 1D linear system
    x = var('x')
    benchmarks["amgf_stoch_linear"] = TaskConfig(
        name="AMGF Linear 1D (mu(A)=-0.4, sigma=sqrt(2))",
        system_type="stochastic",
        vars=[x],
        f_expr=[-0.4 * x],
        initial_set=[Interval(0.0, 0.0)],
        time_horizon=1.5,
        order=4,
        step_size=0.05,
        remainder_estimation=[Interval(-1e-6, 1e-6)],
        engine_params={
            'g_expr': [math.sqrt(2.0)],
            'delta': 0.001,
            "amgf_eps": 1.0 / 16.0,
            'fixed_step_mode': False,
            'precondition_setup': 'ID',
            'mc_traces': 5000,
            "mc_dt": 0.001,
            'plot_dims': [0, 0]
        }
    )

    # Inverted Pendulum
    theta, omega = var('theta, omega')
    benchmarks["stoch_pendulum"] = TaskConfig(
        name="Inverted Pendulum",
        system_type="stochastic",
        vars=[theta, omega],
        f_expr=[omega, 10.0*sin(theta) - 20.0*theta - 20.0*omega],
        initial_set=[
            Interval(-math.pi / 10.0, math.pi / 10.0),
            Interval(-0.2, 0.2)
        ],
        time_horizon=4.0,
        order=4,
        step_size=0.05,
        remainder_estimation=[Interval(-1e-4, 1e-4)] * 2,
        engine_params={
            'g_expr': [[0.0], [0.1]],
            'P_matrix': [[35.68, 2.21], [2.21, 1.27]],
            'delta': 0.001,
            "amgf_eps": 1.0 / 16.0,
            'fixed_step_mode': False,
            'precondition_setup': 'ID',
            'setting': 'single_step',
            'mc_traces': 2000,
            "mc_dt": 0.001,
            "mc_seed": 0,
            'plot_dims': [0, 1]
        }
    )

    # Nonlinear Unicycle (modified)
    px, py, th = var('px py theta')
    Kp_v, Kp_w = 0.1, 0.1
    v_ctrl = -Kp_v * (px * cos(th) + py * sin(th))
    w_ctrl = -Kp_w * th
    benchmarks["stoch_unicycle"] = TaskConfig(
        name="Nonlinear Unicycle",
        system_type="stochastic",
        vars=[px, py, th],
        f_expr=[v_ctrl * cos(th), v_ctrl * sin(th), w_ctrl],
        initial_set=[Interval(4.9, 5.1), Interval(4.9, 5.1), Interval(-2.0, -1.9)],
        time_horizon=4.0,
        order=6,
        step_size=0.05,
        remainder_estimation=[Interval(-1e-1, 1e-1)] * 3, 
        engine_params={
            'g_expr': [[0,0,0], [0,0,0], [0,0,0.1]],
            'delta': 0.001,
            'setting': 'single_step',
            'fixed_step_mode': False,
            'precondition_setup': 'QR',
            'mc_traces': 10000,
            'plot_dims': [0, 1]
        }
    )
    return benchmarks
