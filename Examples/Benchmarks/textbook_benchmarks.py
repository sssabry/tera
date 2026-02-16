"""Textbook examples benchmark task definitions."""

from typing import Dict
import math
from sage.all import SR, var, sin, cos

from TERA.TMCore.Interval import Interval
from TERA.Workbench.TaskConfig import TaskConfig


def get_textbook_benchmarks() -> Dict[str, TaskConfig]:
    """
    benchmarks from Various textbook examples reviewed
    """
    benchmarks: Dict[str, TaskConfig] = {}

    # Example 8.15 - Applied SDEs textbook
    # Duffing Van Der Pol Oscillator
    x1, x2 = var('x1 x2')

    alpha = 1.0 
    q = 0.5**2 
    sigma = math.sqrt(q) 
    benchmarks["stoch_duffing_vdp"] = TaskConfig(
        name="Stochastic Duffing-Van der Pol (Sarkka & Solin 2019, Ex. 8.15)",
        system_type="stochastic",
        vars=[x1, x2],
        # drift f(x): exactly the dt term in Eq. 8.94
        f_expr=[x2, x1*(alpha - x1**2) - x2],
        initial_set=[Interval(0.9, 1.1), Interval(-0.1, 0.1)],
        time_horizon=20.0,
        order=4,
        step_size=0.02,
        remainder_estimation=[Interval(-1e-5, 1e-5)] * 2,
        engine_params={
            # diffusion matrix in eq.8.94 is (0, x1) ^T
            # multiply by sqrt(q) to match the spectral density via standard brownian motion
            'g_expr': [[0.0], [sigma * x1]],
            'delta': 1e-1,
            'amgf_eps': 1.0/16.0,
            'fixed_step_mode': False,
            'P_matrix': [[1.5, 0.25],[0.25, 0.625]], # lyapunov-derived
            'precondition_setup': 'ID',
            'setting': 'single_step',
            'mc_traces': 2000,
            'mc_dt': 0.001,
            'mc_seed': 0,
            'plot_dims': [0, 1],
        }
    )

    x1, x2 = var('x1 x2')

    nu = 1.0
    eta = 1.0/10.0

    benchmarks["stoch_spring_model"] = TaskConfig(
        name="Spring Model (Sarkka & Solin 2019, Ex. 6.6, Eq. 6.47)",
        system_type="stochastic",
        vars=[x1, x2],
        # drift: F x from Eq. (6.47)
        f_expr=[x2, -(nu**2)*x1 - eta*x2],
        initial_set=[Interval(0.9, 1.1), Interval(-0.1, 0.1)],
        time_horizon=20.0,
        order=4,
        step_size=0.02,
        remainder_estimation=[Interval(-1e-8, 1e-8)] * 2,
        engine_params={
            # diffusion: L = [[0],[1]] from Eq. (6.47), so G = sqrt(q) * L
            'g_expr': [[0.0], [1.0]],
            'delta': 1e-3,
            'amgf_eps': 1.0/16.0,
            'P_matrix': [[10.05, 0.5], [0.5,  10.0]],
            'fixed_step_mode': False,
            'precondition_setup': 'ID',
            'setting': 'single_step',
            'mc_traces': 2000,
            'mc_dt': 0.001,
            'mc_seed': 0,
            'plot_dims': [0, 1],
        }
    )
    return benchmarks
