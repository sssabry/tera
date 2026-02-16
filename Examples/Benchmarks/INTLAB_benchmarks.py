"""INTLAB benchmark task definitions."""

from typing import Dict
from sage.all import SR, var, pi

from TERA.TMCore.Interval import Interval
from TERA.Workbench.TaskConfig import TaskConfig


def get_intlab_benchmarks() -> Dict[str, TaskConfig]:
    """
    INTLAB benchmarks (used to validate left_right mode).
    """
    tasks: Dict[str, TaskConfig] = {}

    # Lotka Volterra
    y1, y2 = var("y1 y2")
    tasks["intlab_cont_lotka_volterra"] = TaskConfig(
        name="Lotka-Volterra (INTLAB demo Example II)",
        system_type="continuous",
        vars=[y1, y2],
        f_expr=[
            2.0 * y1 * (1.0 - y2),
            (y1 - 1.0) * y2,
        ],
        initial_set=[Interval(0.95, 1.05), Interval(2.95, 3.05)],
        time_horizon=5.5,
        order=8, #18,
        step_size=0.03, # INTLAB h0
        time_var="t_lovo",
        remainder_estimation=[
            Interval(-1e-3, 1e-3),
            Interval(-1e-3, 1e-3),
        ],

        engine_params={
            "setting": "left_right",
            "precondition_setup": "QR",
            "fixed_step_mode": False,
            "min_step": 0.003, # INTLAB h_min
            "plot_dims": (0, 1),
        },

        # From INTLAB demo page
        expected_final_bounds=[
            Interval(0.7801, 1.1818),
            Interval(2.9294, 3.0493),
        ],
    )

    # Lorenz System 
    x, y, z = var("x y z")
    tasks["intlab_cont_lorenz"] = TaskConfig(
        name="Lorenz (INTLAB demo Example III)",
        system_type="continuous",
        vars=[x, y, z],
        f_expr=[10 * (y - x), x * (28 - z) - y, x * y - (8 / 3) * z],
        initial_set=[Interval(-8.001, -7.998), Interval(7.998, 8.001), Interval(26.998, 27.001)],
        time_horizon=3.0,
        order=10,
        step_size=0.02,
        time_var="t_lor",
        remainder_estimation=[Interval(-1e-11, 1e-11)] * 3,
        engine_params={
            "setting": "left_right", "fixed_step_mode": False, 
            "precondition_setup": "QR", "min_step": 1e-3},
        expected_final_bounds=[
            Interval(-0.8241, -0.7892),
            Interval(-0.2722, -0.2274),
            Interval(19.2187, 19.2350),
        ],
    )

    # Quadratic Model
    y1, y2 = SR.var("y1, y2")
    tasks["intlab_cont_quadratic"] = TaskConfig(
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
            "fixed_step_mode": False, "setting": "left_right", 
            "precondition_setup": "QR"},
        expected_final_bounds=[
            Interval(-0.2326613072710758, 1.030222247610039),
            Interval(0.3497621000683073, 1.122446571618336),
        ],
    )

    # Kepler problem
    y1, y2, y3, y4, y5, y6 = var("y1 y2 y3 y4 y5 y6")

    g = -0.9986
    r2 = y1**2 + y2**2 + y3**2
    inv_r3 = r2**(-SR(3) / 2) # (y1^2+y2^2+y3^2)^(-3/2)
    acc = g * inv_r3
    y0_mid = [
        -1.77269098191512,
        +0.1487214852342955,
        -0.07928350462244194,
        +0.2372031916516237,
        -0.612524538758628,
        +0.04583217572165624,
    ]
    r_small = 0.5e-7
    r_big = 0.5e-6
    y0_rad = [r_small, r_small, r_small, r_big, r_big, r_big]

    tasks["intlab_cont_kepler"] = TaskConfig(
        name="Kepler Asteroid Motion (INTLAB Example VI)",
        system_type="continuous",
        vars=[y1, y2, y3, y4, y5, y6],
        f_expr=[
            y4,
            y5,
            y6,
            y1 * acc,
            y2 * acc,
            y3 * acc,
        ],
        initial_set=[
            Interval(y0_mid[0] - y0_rad[0], y0_mid[0] + y0_rad[0]),
            Interval(y0_mid[1] - y0_rad[1], y0_mid[1] + y0_rad[1]),
            Interval(y0_mid[2] - y0_rad[2], y0_mid[2] + y0_rad[2]),
            Interval(y0_mid[3] - y0_rad[3], y0_mid[3] + y0_rad[3]),
            Interval(y0_mid[4] - y0_rad[4], y0_mid[4] + y0_rad[4]),
            Interval(y0_mid[5] - y0_rad[5], y0_mid[5] + y0_rad[5]),
        ],
        # Tf = 46*pi (≈ 144.513...)
        time_horizon=float(46 * pi),
        order=10,
        step_size=0.1, # INTLAB h0
        time_var="t_kep",
        remainder_estimation=[Interval(-1e-8, 1e-8)] * 6,
        engine_params={
            "fixed_step_mode": False,
            "min_step": 0.001, # INTLAB h_min
            "setting": "left_right",
            "precondition_setup": "QR",
            'cutoff_threshold': 1e11
        },
        # NOTE: INTLAB prints an interval time t
        expected_final_bounds=[
            Interval(-9.037484979419382e-01, -9.031648541269390e-01),  # y1
            Interval(-7.498352814258061e-01, -7.495860766707676e-01),  # y2
            Interval(+8.639455697886244e-03, +8.676296493769255e-03),  # y3
            Interval(-2.231962160116047e-01, -2.228443126846558e-01),  # y4
            Interval(-3.970013350237614e-01, -3.967086925730203e-01),  # y5
            Interval(+6.026482731604484e-02, +6.026908958738871e-02),  # y6
        ],
    )

    return tasks
