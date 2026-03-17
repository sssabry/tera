"""Continuous-time stochastic benchmarks from Bai Xue's paper:
"A New Framework for Bounding Reachability Probabilities of Continuous-time Stochastic Systems".
"""
from typing import Dict
import math
from sage.all import var

from TERA.TMCore.Interval import Interval
from TERA.Workbench.TaskConfig import TaskConfig


def get_benchmarks() -> Dict[str, TaskConfig]:
    benchmarks: Dict[str, TaskConfig] = {}

    x = var('x')

    benchmarks["xue_pop_growth_case1"] = TaskConfig(
        name="Population Growth Model Case 1 (Xue Example 1.1)",
        system_type="stochastic",
        vars=[x],
        f_expr=[-x],
        initial_set=[Interval(-0.8, -0.8)],
        time_horizon=100.0,
        order=4,
        step_size=0.1,
        remainder_estimation=[Interval(-1e-8, 1e-8)],
        engine_params={
            'g_expr': [[math.sqrt(2.0) / 2.0 * x]],
            'delta': 1e-3,
            'amgf_eps': 1.0 / 16.0,
            'fixed_step_mode': False,
            'precondition_setup': 'ID',
            'setting': 'single_step',
            'mc_traces': 1000,
            'mc_dt': 0.001,
            'mc_seed': 0,
            'plot_dims': [0, 0],

            # # Metadata for later probability/event post-processing
            # # State constraint: x^2 - 1 < 0  <=>  x in (-1, 1)
            # # Target set: 100 x^2 - 1 <= 0  <=>  x in [-0.1, 0.1]
            # 'constraint_set': [Interval(-1.0, 1.0)],
            # 'target_set': [Interval(-0.1, 0.1)],
            # 'paper_mc_hit_prob': 0.6556,
            # 'paper_mc_terminal_prob': 0.6556,
        }
    )

    benchmarks["xue_pop_growth_case2"] = TaskConfig(
        name="Population Growth Model Case 2 (Xue Example 1.2)",
        system_type="stochastic",
        vars=[x],
        f_expr=[-10.0 * x],
        initial_set=[Interval(-0.8, -0.8)],
        time_horizon=1.0,
        order=4,
        step_size=0.02,
        remainder_estimation=[Interval(-1e-8, 1e-8)],
        engine_params={
            'g_expr': [[math.sqrt(2.0) / 2.0 * x]],
            'delta': 1e-3,
            'amgf_eps': 1.0 / 16.0,
            'fixed_step_mode': False,
            'precondition_setup': 'ID',
            'setting': 'single_step',
            'mc_traces': 10000,
            'mc_dt': 0.001,
            'mc_seed': 0,
            'plot_dims': [0, 0],

            # # Metadata for later probability/event post-processing
            # # State constraint: x^2 - 1 < 0  <=>  x in (-1, 1)
            # # Target set: 100 x^2 - 1 <= 0  <=>  x in [-0.1, 0.1]
            # 'constraint_set': [Interval(-1.0, 1.0)],
            # 'target_set': [Interval(-0.1, 0.1)],
            # 'paper_mc_hit_prob': 1.0000,
            # 'paper_mc_terminal_prob': 1.0000,
        }
    )

    benchmarks["xue_pop_growth_case3"] = TaskConfig(
        name="Population Growth Model Case 3 (Xue Example 1.3)",
        system_type="stochastic",
        vars=[x],
        f_expr=[-x + 0.1],
        initial_set=[Interval(-0.5, -0.5)],
        time_horizon=10.0,
        order=4,
        step_size=0.05,
        remainder_estimation=[Interval(-1e-8, 1e-8)],
        engine_params={
            'g_expr': [[x**2]],
            'delta': 1e-3,
            'amgf_eps': 1.0 / 16.0,
            'fixed_step_mode': False,
            'precondition_setup': 'ID',
            'setting': 'single_step',
            'mc_traces': 10000,
            'mc_dt': 0.001,
            'mc_seed': 0,
            'plot_dims': [0, 0],

            # # Metadata for later probability/event post-processing
            # # State constraint: x^2 - 1 < 0  <=>  x in (-1, 1)
            # # Target set: 100 x^2 - 1 <= 0  <=>  x in [-0.1, 0.1]
            # 'constraint_set': [Interval(-1.0, 1.0)],
            # 'target_set': [Interval(-0.1, 0.1)],
            # 'paper_mc_hit_prob': 0.9842,
            # 'paper_mc_terminal_prob': 0.4623,
        }
    )

    x, y = var('x y')

    benchmarks["xue_poly_2d"] = TaskConfig(
        name="2D Polynomial SDE (Xue Example 2)",
        system_type="stochastic",
        vars=[x, y],
        f_expr=[
            -x + 1,
            10.0 * y + x
        ],
        initial_set=[
            Interval(-0.5, -0.5),
            Interval(0.5, 0.5)
        ],
        time_horizon=1.0,
        order=4,
        step_size=0.02,
        remainder_estimation=[Interval(-1e-6, 1e-6)] * 2,
        engine_params={
            # G(x,y) = [[x^2, 0],
            #           [0,   -x]]
            'g_expr': [
                [x**2, 0.0],
                [0.0, -x]
            ],
            'delta': 1e-3,
            'amgf_eps': 1.0 / 16.0,
            'fixed_step_mode': False,
            'precondition_setup': 'ID',
            'setting': 'single_step',
            'mc_traces': 10000,
            'mc_dt': 0.001,
            'mc_seed': 0,
            'plot_dims': [0, 1],

            # # Metadata for later probability/event post-processing
            # # State constraint: x^2 + y^2 <= 1
            # # Target set: x^2 + y^2 <= 0.01
            # 'constraint_radius_sq': 1.0,
            # 'target_radius_sq': 0.01,
            # 'paper_mc_hit_prob': 0.0,
            # 'paper_mc_terminal_prob': 0.0,
        }
    )

    return benchmarks