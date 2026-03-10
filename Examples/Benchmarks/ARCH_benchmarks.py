"""ARCH benchmark task definitions."""
from typing import Any, Dict, List, Optional
from sage.all import SR, var, sin, cos, sqrt, tan, pi
import numpy as np

from TERA.TMCore.Interval import Interval
from TERA.Workbench.TaskConfig import TaskConfig
from TERA.Hybrid.HybridModel import HybridAutomaton, Mode, Condition, ResetMap, Transition


def _laubloomis_f_expr(vars_ll):
    x1, x2, x3, x4, x5, x6, x7 = vars_ll
    return [
        1.4 * x3 - 0.9 * x1,
        2.5 * x5 - 1.5 * x2,
        0.6 * x7 - 0.8 * x2 * x3,
        2 - 1.3 * x3 * x4,
        0.7 * x1 - x4 * x5,
        0.3 * x1 - 3.1 * x6,
        1.8 * x6 - 1.5 * x2 * x7,
    ]


def _laubloomis_init_set(W: float):
    # Center point from ARCH19
    c = [1.2, 1.05, 1.5, 2.4, 1.0, 0.1, 0.45]
    return [Interval(ci - W, ci + W) for ci in c]


def get_arch19_cont_laubloomis_variants() -> Dict[str, TaskConfig]:
    """ARCH 2019 Laub-Loomis variants: W=0.01, 0.05, 0.1."""
    tasks: Dict[str, TaskConfig] = {}

    ll_vars = list(var("x1 x2 x3 x4 x5 x6 x7"))
    f_ll = _laubloomis_f_expr(ll_vars)

    common = dict(
        system_type="continuous",
        vars=ll_vars,
        f_expr=f_ll,
        time_horizon=20.0,
        order=4,
        step_size=0.05,
        remainder_estimation=[Interval(-1e-4, 1e-4)] * 7,
        engine_params={"precondition_setup": "ID"},
    )

    # W = 0.01, unsafe x4 >= 4.5
    tasks["arch19_cont_laubloomis_w001"] = TaskConfig(
        name="Laub-Loomis (ARCH 2019) W=0.01",
        initial_set=_laubloomis_init_set(0.01),
        unsafe_sets=[{"dims": {3: Interval(4.5, float("inf"))}, "start_time": 0.0}],
        **common,
    )

    # W = 0.05, unsafe x4 >= 4.5
    tasks["arch19_cont_laubloomis_w005"] = TaskConfig(
        name="Laub-Loomis (ARCH 2019) W=0.05",
        initial_set=_laubloomis_init_set(0.05),
        unsafe_sets=[{"dims": {3: Interval(4.5, float("inf"))}, "start_time": 0.0}],
        **common,
    )

    # W = 0.1, unsafe x4 >= 5.0
    tasks["arch19_cont_laubloomis_w01"] = TaskConfig(
        name="Laub-Loomis (ARCH 2019) W=0.1",
        initial_set=_laubloomis_init_set(0.1),
        unsafe_sets=[{"dims": {3: Interval(5.0, float("inf"))}, "start_time": 0.0}],
        **common,
    )

    return tasks


## HYBRID AUTOMATON BUILDERS
def build_lotka_volterra_crossing_automaton(Qx=1.0, Qy=1.0, R=0.15):
    """Build the ARCH20 Lotka-Volterra crossing automaton."""
    x, y, cnt = var("x y cnt")

    phi = (x - Qx)**2 + (y - Qy)**2 - R**2

    dx = 3*x - 3*x*y
    dy = x*y - y

    g_out_to_in = Condition(constraints=[phi], strict=False) # phi <= 0
    g_in_to_out = Condition(constraints=[-phi], strict=False) # phi >= 0
    
    inv_outside = Condition(constraints=[-phi], strict=True) # phi > 0
    inv_inside  = Condition(constraints=[phi], strict=True) # phi < 0

    outside = Mode("outside", [dx, dy, SR(0)], inv_outside)
    inside  = Mode("inside",  [dx, dy, SR(1)], inv_inside)

    id_reset = ResetMap({})

    outside.transitions.append(
        Transition(outside, inside, g_out_to_in, id_reset, "enter")
    )
    inside.transitions.append(
        Transition(inside, outside, g_in_to_out, id_reset, "exit")
    )

    return HybridAutomaton([outside, inside], [x, y, cnt], "t")

def build_space_rendezvous_automaton():
    """Build the ARCH 2018 space rendezvous automaton."""
    from sage.all import sqrt

    x, y, vx, vy = var("x y vx vy")
    t = var("t")

    # physical constants from ARCH 2018 Sec 3.4.1
    mu = 3.986e14 * 60**2
    r = 42164.0e3
    mc = 500.0
    n = sqrt(mu / r**3)
    rc = sqrt((r + x) ** 2 + y**2)

    # controllers
    K1 = [
        [-28.8287, 0.1005, -1449.9754, 0.0046],
        [-0.087, -33.2562, 0.00462, -1451.5013],
    ]
    K2 = [
        [-288.0288, 0.1312, -9614.9898, 0],
        [-0.1312, -288, 0, -9614.9883],
    ]
    K0 = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]

    def get_dyn(K):
        ux = K[0][0] * x + K[0][1] * y + K[0][2] * vx + K[0][3] * vy
        uy = K[1][0] * x + K[1][1] * y + K[1][2] * vx + K[1][3] * vy
        return [
            vx,
            vy,
            n**2 * x + 2 * n * vy + mu / r**2 - (mu / rc**3) * (r + x) + ux / mc,
            n**2 * y - 2 * n * vx - (mu / rc**3) * y + uy / mc,
        ]

    tan30 = tan(pi / 6)

    # invariants
    # approaching: x <= -100 and t <= 150
    inv_app = Condition([
        x + 100,
        t - 150,
    ])

    # rendezvous_attempt:
    inv_rend = Condition([
        -(x + 100),
        x * tan30 - y,
        x * tan30 + y,
        vx**2 + vy**2 - 3.3**2,
        t - 150,
    ])

    # aborting: no invariant here; collision avoidance is checked afterwards
    inv_abort = Condition([])

    mode_app   = Mode("approaching",          get_dyn(K1), inv_app)
    mode_rend  = Mode("rendezvous_attempt",   get_dyn(K2), inv_rend)
    mode_abort = Mode("aborting",             get_dyn(K0), inv_abort)

    # controller switch at x >= -100
    guard_to_rend = Condition([-(x + 100)])

    # nondeterministic abort available for t >= 120
    # upper end t <= 150 is enforced by source-mode invariants
    abort_guard = Condition([120 - t])

    id_reset = ResetMap({})

    mode_app.transitions.append(Transition(mode_app, mode_rend, guard_to_rend, id_reset, "to_rendezvous"))
    mode_app.transitions.append(Transition(mode_app, mode_abort, abort_guard, id_reset, "abort_early"))
    mode_rend.transitions.append(Transition(mode_rend, mode_abort, abort_guard, id_reset, "abort_late"))

    return HybridAutomaton([mode_app, mode_rend, mode_abort],[x, y, vx, vy], "t")



# ARCH: Continuous benchmarks by year

def get_arch_continuous_2017() -> Dict[str, TaskConfig]:
    """Return ARCH 2017 continuous benchmarks."""
    tasks: Dict[str, TaskConfig] = {}

    # Van der Pol Oscillator (ARCH 2017)
    x, y = var("x y")
    tasks["arch17_cont_vanderpol"] = TaskConfig(
        name="Van der Pol (ARCH 2017)",
        system_type="continuous",
        vars=[x, y],
        f_expr=[y, y - x - (x**2) * y],
        initial_set=[Interval(1.25, 1.55), Interval(2.35, 2.45)],
        time_horizon=7.0,
        order=5,
        step_size=0.02,
        remainder_estimation=[Interval(-1e-4, 1e-4)] * 2,
        engine_params={"precondition_setup": "ID", "fixed_step_mode": True, "plot_dims": (0, 1)},
        expected_final_width=0.02434,
        unsafe_sets=[{"dims": {1: Interval(2.75, float('inf'))}, "start_time": 0.0}],
    )

    # Laub-Loomis (ARCH 17/18)
    ll_vars = list(var("x1 x2 x3 x4 x5 x6 x7"))
    x1, x2, x3, x4, x5, x6, x7 = ll_vars
    tasks["arch17_cont_laubloomis"] = TaskConfig(
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
        initial_set=_laubloomis_init_set(0.01),
        time_horizon=20.0,
        order=4,
        step_size=0.05,
        remainder_estimation=[Interval(-3e-4, 3e-4)] * 7,
        engine_params={"precondition_setup": "ID"},
        expected_final_width=0.01044,  # Flow* in x_4 dimension from ARCH 2018
        unsafe_sets=[{"dims": {3: Interval(4.5, float('inf'))}, "start_time": 0.0}],
    )

    # Quadrotor (ARCH 2017)
    q_vars = list(var("x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12"))
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = q_vars

    g, R, l, M_rotor, M = 9.81, 0.1, 0.5, 0.1, 1.0
    m = M + 4 * M_rotor
    Jx = (2.0 / 5.0) * M * (R**2) + 2 * (l**2) * M_rotor
    Jy = Jx
    Jz = (2.0 / 5.0) * M * (R**2) + 4 * (l**2) * M_rotor

    F = m * g - 10 * (x3 - 1.0) + 3 * x6
    tau_phi = -(x7 - 0.0) - x10
    tau_theta = -(x8 - 0.0) - x11
    tau_psi = 0

    tasks["arch17_cont_quadrotor"] = TaskConfig(
        name="Quadrotor (ARCH 2017)",
        system_type="continuous",
        vars=q_vars,
        f_expr=[
            (cos(x8) * cos(x9)) * x4
            + (sin(x7) * sin(x8) * cos(x9) - cos(x7) * sin(x9)) * x5
            + (cos(x7) * sin(x8) * cos(x9) + sin(x7) * sin(x9)) * x6,
            (cos(x8) * sin(x9)) * x4
            + (sin(x7) * sin(x8) * sin(x9) + cos(x7) * cos(x9)) * x5
            + (cos(x7) * sin(x8) * sin(x9) - sin(x7) * cos(x9)) * x6,
            (sin(x8) * x4) - (sin(x7) * cos(x8) * x5) - (cos(x7) * cos(x8) * x6),
            (x12 * x5) - (x11 * x6) - (g * sin(x8)),
            (x10 * x6) - (x12 * x4) + (g * cos(x8) * sin(x7)),
            (x11 * x4) - (x10 * x5) + (g * cos(x8) * cos(x7)) - (F / m),
            x10 + (sin(x7) * (sin(x8) / cos(x8)) * x11) + (cos(x7) * (sin(x8) / cos(x8)) * x12),
            (cos(x7) * x11) - (sin(x7) * x12),
            ((sin(x7) / cos(x8)) * x11) + ((cos(x7) / cos(x8)) * x12),
            (((Jy - Jz) / Jx) * x11 * x12) + ((1 / Jx) * tau_phi),
            (((Jz - Jx) / Jy) * x10 * x12) + ((1 / Jy) * tau_theta),
            (((Jx - Jy) / Jz) * x10 * x11) + ((1 / Jz) * tau_psi),
        ],
        initial_set=[Interval(-0.4, 0.4)] * 6 + [Interval(0.0, 0.0)] * 6,
        time_horizon=5.0,
        order=4,
        step_size=0.02,
        time_var="ta",
        remainder_estimation = [
            Interval(-1.05e-4, 1.05e-4),  # x1
            Interval(-1.05e-4, 1.05e-4),  # x2
            Interval(-1.02e-4, 1.02e-4),  # x3
            Interval(-1.25e-4, 1.25e-4),  # x4
            Interval(-1.25e-4, 1.25e-4),  # x5
            Interval(-1.12e-4, 1.12e-4),  # x6
            Interval(-1.02e-4, 1.02e-4),  # x7
            Interval(-1.02e-4, 1.02e-4),  # x8

            Interval(-8.0e-5, 8.0e-5),    # x9  (conservative default)
            Interval(-5.0e-5, 5.0e-5),    # x10 (rates: smaller)
            Interval(-5.0e-5, 5.0e-5),    # x11
            Interval(-5.0e-5, 5.0e-5),    # x12
        ],
        engine_params={
            "precondition_setup": "ID",
            "fixed_step_mode": False,
            "max_step": 0.05,
            "cutoff_threshold": 1e-8,
            "step_grow_width_cap": 5e-2
        },
        expected_final_width=0.0003103,
        # missing reachability goal region - future fix
        unsafe_sets=[{"dims": {2: Interval(1.4, float('inf'))}, "start_time": 0.0},{"dims": {2: Interval(float('-inf'), 0.9)}, "start_time": 1.0}]
    )

    return tasks


def get_arch_continuous_2018() -> Dict[str, TaskConfig]:
    """Return ARCH 2018 continuous benchmarks."""
    tasks: Dict[str, TaskConfig] = {}

    # Van der Pol Oscillator (ARCH 2018)
    x, y = var("x y")
    tasks["arch18_cont_vanderpol"] = TaskConfig(
        name="Van der Pol (ARCH 2018)",
        system_type="continuous",
        vars=[x, y],
        f_expr=[y, y - x - (x**2) * y],
        initial_set=[Interval(1.25, 1.55), Interval(2.35, 2.45)],
        time_horizon=7.0,
        order=6,
        step_size=0.04,
        remainder_estimation=[Interval(-1e-4, 1e-4)] * 2,
        engine_params={"precondition_setup": "ID", "fixed_step_mode": True, "plot_dims": (0, 1)},
        expected_final_width=0.01214,
        unsafe_sets=[{"dims": {1: Interval(2.75, float('inf'))}, "start_time": 0.0}],

    )

    return tasks


def get_arch_continuous_2019() -> Dict[str, TaskConfig]:
    """Return ARCH 2019 continuous benchmarks."""
    tasks: Dict[str, TaskConfig] = {}

    # Van der Pol 2019 (mu = 2.0)
    x, y = var("x y")
    tasks["arch19_cont_vanderpol_mu2"] = TaskConfig(
        name="Van der Pol (ARCH 2019) mu=2.0",
        system_type="continuous",
        vars=[x, y],
        f_expr=[y, 2.0 * (1 - x**2) * y - x],
        initial_set=[Interval(1.55, 1.85), Interval(2.35, 2.45)],
        time_horizon=8.0,
        order=5,
        step_size=0.02,
        remainder_estimation=[Interval(-1e-4, 1e-4)] * 2,
        engine_params={"precondition_setup": "ID", "fixed_step_mode": True},
        unsafe_sets=[{"dims": {1: Interval(4.0, float("inf"))}, "start_time": 0.0}]
    )

    tasks.update(get_arch19_cont_laubloomis_variants())

    return tasks


def get_arch_continuous_2020() -> Dict[str, TaskConfig]:
    """Return ARCH 2020 continuous benchmarks."""
    tasks: Dict[str, TaskConfig] = {}

    # Production-Destruction (ARCH 2020)
    x, y, z = var("x y z")
    tasks["arch20_cont_production_destruction"] = TaskConfig(
        name="Production-Destruction (ARCH 2020)",
        system_type="continuous",
        vars=[x, y, z],
        f_expr=[-(x * y) / (1 + x), (x * y) / (1 + x) - 0.3 * y, 0.3 * y],
        initial_set=[Interval(9.5, 10.0), Interval(0.01, 0.01), Interval(0.01, 0.01)],
        time_horizon=100.0,
        order=3,
        step_size=0.02,
        remainder_estimation=[Interval(-0.1, 0.1)] * 3,
        expected_final_width=1.4e-22,
        engine_params={'setting': 'single_step', 'fixed_step_mode': False}
    )

    # Coupled Van der Pol (ARCH 2020)
    x1, y1, x2, y2 = var("x1 y1 x2 y2")
    tasks["arch20_cont_coupled_vanderpol"] = TaskConfig(
        name="Coupled Van der Pol (ARCH 2020)",
        system_type="continuous",
        vars=[x1, y1, x2, y2],
        f_expr=[y1, 1.0 * (1 - x1**2) * y1 - 2 * x1 + x2, y2, 1.0 * (1 - x2**2) * y2 - 2 * x2 + x1],
        initial_set=[
            Interval(1.25, 1.55),
            Interval(2.35, 2.45),
            Interval(1.25, 1.55),
            Interval(2.35, 2.45),
        ],
        time_horizon=7.0,
        order=6,
        step_size=0.01,
        remainder_estimation=[Interval(-1e-4, 1e-4)] * 4,
        engine_params={"precondition_setup": "ID"},
    )

    return tasks


def get_arch_continuous_2021() -> Dict[str, TaskConfig]:
    """Return ARCH 2021 continuous benchmarks."""
    tasks: Dict[str, TaskConfig] = {}

    # Robertson stiff chemical reaction (ARCH 2021)
    x1, x2, x3 = var("x1 x2 x3")
    tasks["arch21_cont_robertson_setup3"] = TaskConfig(
        name="Robertson Setup 3 (ARCH 2021)",
        system_type="continuous",
        vars=[x1, x2, x3],
        f_expr=[
            -0.4 * x1 + 1e3 * x2 * x3,
            0.4 * x1 - 1e3 * x2 * x3 - 1e7 * (x2**2),
            1e7 * (x2**2),
        ],
        initial_set=[Interval(1.0, 1.0), Interval(0.0, 0.0), Interval(0.0, 0.0)],
        time_horizon=40.0,
        order=10,
        step_size=0.001,
        expected_final_width=1.2e-9,  # JuliaReach
        engine_params={"setting": "single_step", "fixed_step_mode": False},
    )

    return tasks


# ARCH: Hybrid benchmarks by year

def get_arch_hybrid_2020() -> Dict[str, TaskConfig]:
    """Return ARCH 2020 hybrid benchmarks."""
    tasks: Dict[str, TaskConfig] = {}

    eps = 0.008
    tasks["arch20_hybrid_lovo20"] = TaskConfig(
        name="Lotka-Volterra Crossing (ARCH 2020)",
        system_type="hybrid",
        vars=list(var("x y cnt")),
        initial_set=[Interval(1.3 - eps, 1.3 + eps),Interval(1.0, 1.0),Interval(0.0, 0.0),],
        initial_mode="outside",
        time_horizon=3.64,
        order=3,
        urgent_jumps_mode=True,
        step_size=0.01,
        engine_params={
            "automaton": build_lotka_volterra_crossing_automaton(),
            "max_jumps": 10,
            'min_step': 1e-3,
            'max_step': 0.01,
            "fixed_step_mode": False,
            'intersection_method': 'domain_contraction',
            'aggregation_method': "PCA",
            "initial_split": {
                "enabled": True,
                "dim": 0,
                "parts": 5
            },
        },
    )

    tasks["arch20_hybrid_space_rendezvous"] = TaskConfig(
        name="Space Rendezvous (ARCH 2018)",
        system_type="hybrid",
        vars=list(var("x y vx vy")),
        initial_set=[Interval(-925, -875), Interval(-425, -375), Interval(0, 0), Interval(0, 0)],
        initial_mode="approaching",
        time_horizon=200.0,
        order=5,
        step_size=0.1,
        urgent_jumps_mode=False,
        remainder_estimation=[Interval(-1e-3, 1e-3)] * 4,
        engine_params={
            "automaton": build_space_rendezvous_automaton(),
            "max_jumps": 5,
            "fixed_step_mode": False,
            "min_step": 0.001,
            "max_step": 0.5,
            "plot_dims": (0, 1),
            "cutoff_threshold": 1e-6,
        },
    )

    return tasks

# return all ARCH tasks (merged)

def get_arch_benchmarks() -> Dict[str, TaskConfig]:
    """Return all ARCH benchmarks merged into one dict."""
    tasks: Dict[str, TaskConfig] = {}

    # Continuous
    for fn in (
        get_arch_continuous_2017,
        get_arch_continuous_2018,
        get_arch_continuous_2019,
        get_arch_continuous_2020,
        get_arch_continuous_2021,
    ):
        tasks.update(fn())

    # Hybrid
    for fn in (get_arch_hybrid_2018, get_arch_hybrid_2020):
        tasks.update(fn())

    return tasks
