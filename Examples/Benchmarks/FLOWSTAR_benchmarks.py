"""Flow* benchmark task definitions."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sage.all import SR, var, sin, cos, exp
import numpy as np

from TERA.Workbench.TaskConfig import TaskConfig
from TERA.TMCore.Interval import Interval
from TERA.Hybrid.HybridModel import HybridAutomaton, Mode, Condition, ResetMap, Transition

def build_bouncing_ball_automaton():
    """Build the bouncing ball hybrid automaton."""
    x, v = var("x v")
    g = 9.81

    # Mode: Down
    down_inv = Condition(constraints=[-x, v])  # x >= 0, v <= 0
    down_mode = Mode("down", [v, -g + 0.1 * v**2], down_inv)

    # Mode: Up
    up_inv = Condition(constraints=[-x, -v])  # x >= 0, v >= 0
    up_mode = Mode("up", [v, -g - 0.1 * v**2], up_inv)

    # Transition: Bounce (Down -> Up)
    bounce_guard = Condition(constraints=[x])
    bounce_reset = ResetMap(mapping={"v": -0.8 * v})
    down_mode.transitions.append(
        Transition(down_mode, up_mode, bounce_guard, bounce_reset, "alpha")
    )

    # Transition: Apex (Up -> Down)
    apex_guard = Condition(constraints=[v])
    up_mode.transitions.append(
        Transition(up_mode, down_mode, apex_guard, ResetMap({}), "beta")
    )

    return HybridAutomaton([down_mode, up_mode], [x, v], "t")


def build_spiking_neuron_1_automaton():
    """Build the spiking neuron hybrid automaton."""
    v, u = var("v u")
    C, vr, vt, I, a, b = 100, -60, -40, 70, 0.03, -2

    # Low Mode
    m1_inv = Condition(constraints=[v + 40])
    m1_dyn = [
        (0.7 * (v - vr) * (v - vt) - u + I) / C,
        a * (b * (v - vr) - u),
    ]
    mode_low = Mode("v_low", m1_dyn, m1_inv)

    # High Mode
    m2_inv = Condition(constraints=[v - 35, -40 - v])
    m2_dyn = [
        (7 * (v - vr) * (v - vt) - u + I) / C,
        a * (b * (v - vr) - u),
    ]
    mode_high = Mode("v_high", m2_dyn, m2_inv)

    # Transitions
    mode_low.transitions.append(
        Transition(mode_low, mode_high, Condition([-40 - v]), ResetMap({}), "to_high")
    )

    spike_reset = ResetMap({"v": -50, "u": u + 100})
    mode_high.transitions.append(
        Transition(mode_high, mode_low, Condition([v - 35]), spike_reset, "spike")
    )

    return HybridAutomaton([mode_low, mode_high], [v, u], "t")


def build_2d_stable_system_automaton():
    """Build the 2D stable system hybrid automaton."""
    x, y = var("x y")

    # Mode L1
    l1_inv = Condition(constraints=[-(x - y + 3)])
    l1_mode = Mode("l1", [-y, x**2], l1_inv)

    # Mode L2
    l2_inv = Condition(constraints=[-(y + 2), y - 3])
    l2_mode = Mode("l2", [-y, x**3], l2_inv)

    # Transitions
    l1_to_l2_guard = Condition(constraints=[x - y + 3])
    l1_mode.transitions.append(
        Transition(l1_mode, l2_mode, l1_to_l2_guard, ResetMap({}), "l1_to_l2")
    )

    l2_to_l1_guard = Condition(constraints=[y + 2])
    l2_mode.transitions.append(
        Transition(l2_mode, l1_mode, l2_to_l1_guard, ResetMap({}), "l2_to_l1")
    )

    return HybridAutomaton([l1_mode, l2_mode], [x, y], "t")


def build_stable_sys_3d_automaton():
    """Build the 3D stable system hybrid automaton."""
    x, y, z = var("x y z")

    # Mode L1
    l1_dyn = [
        -9 * (x - 2)
        - 7 * (y + 2)
        + (z - 1)
        + 0.2 * (x - 2) * (y + 2)
        + 0.1 * (y + 2) * (z - 1)
        + 0.1 * (x - 2) * (z - 1)
        + 0.5 * (z - 1) ** 2,
        6 * (x - 2) + 4 * (y + 2) + (z - 1),
        3 * (x - 2) + 2 * (y + 2) - 2.5 * (z - 1),
    ]
    l1_mode = Mode("l1", l1_dyn, Condition([]))

    # Mode L2
    l2_dyn = [
        2.2 * x + 3.6 * y + 3.9 * z,
        3 * x + 2.4 * y + 3.4 * z - 0.01 * x**2,
        -5 * x - 5.4 * y - 6.7 * z,
    ]
    l2_mode = Mode("l2", l2_dyn, Condition([]))

    # Transition L1 -> L2
    guard_constraints = [
        x - 2.3,
        1.7 - x,
        y + 1.7,
        -2.3 - y,
        z - 1.3,
        0.7 - z,
    ]
    l1_mode.transitions.append(
        Transition(l1_mode, l2_mode, Condition(guard_constraints), ResetMap({}), "jump")
    )

    return HybridAutomaton([l1_mode, l2_mode], [x, y, z], "t")


def get_chen_continuous():
    """ continuous system examples from chen's thesis ch3"""

    tasks: Dict[str, TaskConfig] = {}
    
    # Jet Engine
    jx, jy = var('x y')
    tasks["cont_jet_engine"] = TaskConfig(
        name="Jet Engine (Example 3.3.9)",
        system_type="continuous",
        vars=[jx, jy],
        f_expr=[-jy - 1.5*jx**2 - 0.5*jx**3 - 0.5, 3*jx - jy],
        initial_set=[Interval(0.9, 1.1), Interval(0.9, 1.1)],
        time_horizon=10.0,
        order=4,
        step_size=0.02,
        time_var='t_aug',
        remainder_estimation=[Interval(-1e-5, 1e-5)] * 2,
        engine_params={'precondition_setup': 'ID', 'setting': 'single_step', 'fixed_step_mode': True},
        expected_final_width=0.0264 # from chen's thesis table 5.2
    )

    # Lotka-Volterra 
    x, y = var('x y')
    tasks["cont_lotka_volterra"] = TaskConfig(
        name="Lotka-Volterra (Example 3.3.11)",
        system_type="continuous",
        vars=[x, y],
        f_expr=[1.5*x - x*y, -3*y + x*y],
        initial_set=[Interval(4.95, 5.05), Interval(1.95, 2.05)],
        time_horizon=4.0,
        order=5,
        step_size=0.01,
        time_var='t_aug',
        remainder_estimation=[Interval(-1e-3, 1e-3)] * 2,
        engine_params={'precondition_setup': 'ID', 'fixed_step_mode': True},
        expected_final_bounds=[
            Interval(1.611473542017, 1.688426503140),
            Interval(2.043880403362, 2.136095251238)
        ]
    )

    # TODO: Verify Spring Pendulum setup against Chen's thesis configuration.
        # Spring Pendulum (Case 1: Point Mass)
    sr, st, svr, svt = var('r theta vr vtheta')
    sp_dyn = [svr, svt, sr*(svt**2) + 9.8*cos(st) - 2*(sr-1), -(2*svr*svt + 9.8*sin(st))/sr]
    tasks["cont_spring_pendulum_c1"] = TaskConfig(
        name="Spring Pendulum (Example 3.3.12 - Case 1)",
        system_type="continuous",
        vars=[sr, st, svr, svt],
        f_expr=sp_dyn,
        initial_set=[Interval(1.2, 1.2), Interval(0.5, 0.5), Interval(0.0, 0.0), Interval(0.0, 0.0)],
        time_horizon=10.0,
        order=6,
        step_size=0.01,
        remainder_estimation=[Interval(-1e-3, 1e-3)] * 4,
        engine_params={'precondition_setup': 'QR', 'fixed_step_mode': False}
    )

    # Spring Pendulum (Case 2: Small Box)
    tasks["cont_spring_pendulum_c2"] = TaskConfig(
        name="Spring Pendulum (Example 3.3.12 - Case 2)",
        system_type="continuous",
        vars=[sr, st, svr, svt],
        f_expr=sp_dyn,
        initial_set=[Interval(1.19, 1.21), Interval(0.49, 0.51), Interval(-0.01, 0.01), Interval(-0.01, 0.01)],
        time_horizon=5.0,
        order=6,
        step_size=0.01,
        remainder_estimation=[Interval(-1e-3, 1e-3)] * 4,
        engine_params={'precondition_setup': 'QR', 'fixed_step_mode': False}
    )

    # Spring Pendulum (Case 3: Large Box)
    tasks["cont_spring_pendulum_c3"] = TaskConfig(
        name="Spring Pendulum (Example 3.3.12 - Case 3)",
        system_type="continuous",
        vars=[sr, st, svr, svt],
        f_expr=sp_dyn,
        initial_set=[Interval(1.0, 1.6), Interval(0.4, 1.0), Interval(-0.3, 0.3), Interval(-0.3, 0.3)],
        time_horizon=1.0,
        order=6,
        step_size=0.01,
        remainder_estimation=[Interval(-1e-3, 1e-3)] * 4,
        engine_params={'precondition_setup': 'QR', 'fixed_step_mode': False}
    )

    # Lorenz System
    xl, yl, zl = var('x y z')
    tasks["cont_lorenz_chen"] = TaskConfig(
        name="Lorenz (Example 3.4.2)",
        system_type="continuous",
        vars=[xl, yl, zl],
        f_expr=[10*(yl - xl), xl*(28 - zl) - yl, xl*yl - (8/3)*zl],
        initial_set=[Interval(14.999, 15.001), Interval(14.999, 15.001), Interval(35.999, 36.001)],
        time_horizon=7.0,
        order=7,
        step_size=0.02,
        time_var='t_lor',
        remainder_estimation=[Interval(-1e-5, 1e-5)] * 3,
        engine_params={'fixed_step_mode': False, 'setting': 'single_step', 'precondition_setup': 'ID', "cutoff_threshold":1e-12},
        expected_final_width=0.3751 # from chens thesis table 5.2
    )

    # Rössler Attractor
    rx, ry, rz = var('x y z')
    tasks["cont_rossler"] = TaskConfig(
        name="Rössler Attractor (Example 3.4.3)",
        system_type="continuous",
        vars=[rx, ry, rz],
        f_expr=[-ry - rz, rx + 0.2*ry, 0.2 + rz*(rx - 5.7)],
        initial_set=[Interval(-0.2, 0.2), Interval(-8.6, -8.2), Interval(-0.2, 0.2)],
        time_horizon=4.0,
        order=10,
        step_size=0.01,
        time_var='t_ross',
        engine_params={'fixed_step_mode': True, 'precondition_setup': 'ID'}
    )

    # Van der Pol
    xv, yv = var('x y')
    tasks["cont_vanderpol"] = TaskConfig(
        name="Van der Pol (Example 5.2.1)",
        system_type="continuous",
        vars=[xv, yv],
        f_expr=[yv, (1 - xv**2)*yv - xv],
        initial_set=[Interval(1.25, 1.55), Interval(2.35, 2.45)],
        time_horizon=7.0,
        order=5,
        step_size=0.02,
        remainder_estimation=[Interval(-1e-4, 1e-4)] * 2,
        engine_params={'setting': 'single_step', 'precondition_setup': 'ID', 'fixed_step_mode': True}
    )

    # 7D Biological Model 
    vars_bio = list(var('x1 x2 x3 x4 x5 x6 x7'))
    x1, x2, x3, x4, x5, x6, x7 = vars_bio
    f_bio = [
        -0.4*x1 + 5*x3*x4, 0.4*x1 - x2, x2 - 5*x3*x4,
        5*x5*x6 - 5*x3*x4, -5*x5*x6 + 5*x3*x4, 0.5*x7 - 5*x5*x6, -0.5*x7 + 5*x5*x6
    ]
    tasks["cont_bio"] = TaskConfig(
        name="Biological Model I (Example 5.2.4)",
        system_type="continuous",
        vars=vars_bio,
        f_expr=f_bio,
        initial_set=[Interval(0.99, 1.01)] * 7,
        time_horizon=2.0,
        order=4,
        step_size=0.01,
        time_var='t_aug',
        remainder_estimation=[Interval(-1e-5, 1e-5)] * 7,
        engine_params={'precondition_setup': 'ID', 'setting': 'single_step', 'fixed_step_mode': True}
    )


    # NOTE: TABLE 5.1 CONTAINS RESULTS OF BIOLOGICAL MODEL 5.2.4, ROSSLER, BRUSSELATOR, JET ENGINE, COUPLED VAN DER POL
    return tasks


def get_chen_hybrid():
    """ hybrid system examples from chen's thesis ch4"""
    tasks: Dict[str, TaskConfig] = {}

    
    tasks["hybrid_bouncing_ball"] = TaskConfig(
        name="Bouncing Ball",
        system_type="hybrid",
        vars=[var('x'), var('v')],
        initial_set=[Interval(4.9, 5.1), Interval(-0.2, 0.0)],
        initial_mode="down",
        time_horizon=5,
        order=5,
        urgent_jumps_mode=True,
        step_size=0.01,
        engine_params={
            'automaton': build_bouncing_ball_automaton(),
            'max_jumps': 20,
            'intersection_method': 'domain_contraction',
            'aggregation_method': 'PCA',
            'fixed_step_mode': False,
            'precondition_setup': 'ID',
            'setting': 'single_step',
            'cutoff_threshold': 1e-12,
            'max_iterations': 40
        }
    )

    # Spiking Neuron 1 
    tasks["hybrid_neuron_1"] = TaskConfig(
        name="Spiking Neuron Model 1",
        system_type="hybrid",
        vars=[var('v'), var('u')],
        initial_set=[Interval(-61.0, -59.0), Interval(-1.0, 1.0)],
        initial_mode="v_low",
        time_horizon=1000.0,
        urgent_jumps_mode=False,
        order=4,
        step_size=0.02,
        engine_params={
            'automaton': build_spiking_neuron_1_automaton(),
            'max_jumps': 1000,
            'intersection_method': 'domain_contraction', 
            'aggregation_method': 'PCA',
            'fixed_step_mode': False,
            'cutoff_threshold': 1e-12,
            'min_step': 1e-4,
            'max_step': 0.1,
            'precondition_setup': 'ID',
            'setting': 'single_step',
            'max_iterations': 40
        }
    )

    # 2D stable system
    tasks["hybrid_stable_2d"] = TaskConfig(
        name="2D Stable System",
        system_type="hybrid",
        vars=[var('x'), var('y')],
        initial_set=[Interval(0.9, 1.1), Interval(-1.1, -0.9)],
        initial_mode="l1",
        time_horizon=20.0,
        urgent_jumps_mode=True,
        order=6,
        step_size=0.01,
        engine_params={
            'automaton': build_2d_stable_system_automaton(),
            'max_jumps': 10,
            'intersection_method': 'domain_contraction',
            'aggregation_method': 'PCA',
            'fixed_step_mode': False,
            'precondition_setup': 'ID',
            'setting': 'single_step',
            'cutoff_threshold': 1e-12,
            'max_iterations': 40
        }
    )

    # 3D Stable System 
    tasks["hybrid_stable_3d"] = TaskConfig(
        name="3D Stable System",
        system_type="hybrid",
        vars=[var('x'), var('y'), var('z')],
        initial_set=[Interval(3.0, 3.5), Interval(-3.0, -2.5), Interval(1.0, 1.5)],
        initial_mode="l1",
        time_horizon=20.0,
        urgent_jumps_mode=False,
        order=6,
        step_size=0.01,
        engine_params={
            'automaton': build_stable_sys_3d_automaton(),
            'max_jumps': 10,
            'intersection_method': 'domain_contraction',
            'aggregation_method': 'PCA',
            'precondition_setup': 'ID',
            'fixed_step_mode': False
        }
    )
    return tasks
