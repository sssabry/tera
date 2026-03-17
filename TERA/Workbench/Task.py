"""Task execution entry points and helpers."""

from TERA.TMFlow.TMReach import TMReach
from TERA.Hybrid.HybridReach import HybridReach
from TERA.Stochastic.StochasticReach import StochasticTMReach
from TERA.Workbench.Results import ReachResult
from TERA.Workbench.Report import Report
from TERA.Workbench.TaskConfig import TaskConfig
from TERA.TMCore.TMComputer import init_taylor_model
from TERA.TMCore.TMVector import TMVector
from TERA.Stochastic.Simulator import MonteCarloValidator 
from sage.all import PolynomialRing, RIF
from TERA.TMCore.Interval import Interval
from TERA.TMCore.Polynomial import Polynomial
from TERA.TMCore.TaylorModel import TaylorModel
from TERA.TMCore.TMVector import TMVector

import time
import math
def create_initial_box_continuous(dim: int, bounds: list, order: int, time_var: str = "t") -> TMVector:
    """Create a TMVector for a continuous initial box (including time variable).

    Args:
        dim: State dimension.
        bounds: List of Interval bounds for each dimension.
        order: Taylor model order.

    Returns:
        TMVector representing the initial set.
    """
    if len(bounds) != dim:
        raise ValueError(f"create_initial_box_continuous: dim={dim} but len(bounds)={len(bounds)}")

    var_names = tuple([f"x{i}" for i in range(dim)] + [time_var])
    ring = PolynomialRing(RIF, names=var_names)
    gens = ring.gens()

    dom = [Interval(-1.0, 1.0) for _ in range(dim)] + [Interval(0.0, 0.0)]
    ref = tuple([0.0] * (dim + 1))

    tms = []
    for i, iv in enumerate(bounds):
        lo = float(iv.lower)
        hi = float(iv.upper)
        mid = RIF((lo + hi) * 0.5)
        rad = RIF((hi - lo) * 0.5)

        sage_poly = ring(mid) + ring(rad) * gens[i]
        poly = Polynomial(_poly=sage_poly, _ring=ring)

        tm = TaylorModel(
            poly=poly,
            rem=Interval(0.0, 0.0),
            domain=list(dom),
            ref_point=ref,
            max_order=order
        )
        tm.sweep()
        tms.append(tm)

    # time component: use the last generator directly (tau)
    tau_poly = Polynomial(_poly=gens[-1], _ring=ring)
    tau_tm = TaylorModel(
        poly=tau_poly,
        rem=Interval(0.0, 0.0),
        domain=list(dom),
        ref_point=ref,
        max_order=order
    )
    tau_tm.sweep()
    tms.append(tau_tm)

    return TMVector(tms)

def create_initial_box_hybrid(var_syms: list, bounds: list, order: int) -> TMVector:
    """creates a state-only TMV for a hybrid initial box (1 TM per state var)
    """
    if len(bounds) != len(var_syms):
        raise ValueError(
            f"create_initial_box_hybrid: len(bounds)={len(bounds)} but len(var_syms)={len(var_syms)}"
        )

    state_var_names = [str(v) for v in var_syms]
    tms = [
        init_taylor_model(v, state_var_names, bounds, order=order, expand_function=False)
        for v in var_syms
    ]
    return TMVector(tms)


def _clone_interval_box(bounds: list) -> list:
    """Deep-copy an interval box as a fresh list of Interval objects."""
    return [Interval(float(iv.lower), float(iv.upper)) for iv in bounds]


def _split_interval_box_once(bounds: list, split_dim: int, parts: int) -> list:
    """Split one interval box along a single dimension into `parts` sub-boxes."""
    if parts <= 1:
        return [_clone_interval_box(bounds)]

    if split_dim < 0 or split_dim >= len(bounds):
        raise ValueError(
            f"_split_interval_box_once: split_dim={split_dim} out of range for len(bounds)={len(bounds)}"
        )

    iv = bounds[split_dim]
    lo = float(iv.lower)
    hi = float(iv.upper)

    if hi < lo:
        raise ValueError(
            f"_split_interval_box_once: invalid interval on dim {split_dim}: [{lo}, {hi}]"
        )

    # Degenerate interval: nothing to split
    if hi == lo:
        return [_clone_interval_box(bounds)]

    width = (hi - lo) / float(parts)
    boxes = []

    for k in range(parts):
        a = lo + k * width
        b = hi if k == parts - 1 else lo + (k + 1) * width

        piece = _clone_interval_box(bounds)
        piece[split_dim] = Interval(a, b)
        boxes.append(piece)

    return boxes


def get_hybrid_initial_boxes(initial_set: list, engine_params: dict) -> list:
    """return 1+ hybrid init boxes according to user config (if any)
    config format: engine_params["initial_split"] = { "enabled": True, "dims": [0] or 0
    "parts": [5] or 5}
    """
    base_box = _clone_interval_box(initial_set)

    split_cfg = engine_params.get("initial_split", None)
    if not split_cfg:
        return [base_box]

    if not bool(split_cfg.get("enabled", False)):
        return [base_box]

    dims = split_cfg.get("dims", split_cfg.get("dim", None))
    parts = split_cfg.get("parts", None)

    if dims is None:
        raise ValueError(
            "initial_split enabled but no split dimension provided. "
            "Use {'enabled': True, 'dims': [0], 'parts': [5]}."
        )

    if isinstance(dims, int):
        dims = [dims]
    else:
        dims = list(dims)

    if parts is None:
        parts = [2] * len(dims)
    elif isinstance(parts, int):
        parts = [parts] * len(dims)
    else:
        parts = list(parts)

    if len(parts) != len(dims):
        raise ValueError(
            f"initial_split mismatch: len(dims)={len(dims)} but len(parts)={len(parts)}"
        )

    boxes = [base_box]
    for split_dim, n_parts in zip(dims, parts):
        next_boxes = []
        for box in boxes:
            next_boxes.extend(_split_interval_box_once(box, int(split_dim), int(n_parts)))
        boxes = next_boxes

    return boxes

def create_initial_box_stochastic(state_dim: int, bounds: list, order: int, time_var: str = "t") -> TMVector:
    """Create a TMVector for a stochastic initial box including time.

    Args:
        state_dim: State dimension.
        bounds: List of Interval bounds for each dimension.
        order: Taylor model order.
        time_var: Name of the time variable.

    Returns:
        TMVector representing the initial set with time variable.
    """
    if len(bounds) != state_dim:
        raise ValueError(f"create_initial_box_with_time: state_dim={state_dim} but len(bounds)={len(bounds)}")

    var_names = tuple([f"x{i}" for i in range(state_dim)] + [time_var])
    ring = PolynomialRing(RIF, names=var_names)
    gens = ring.gens()

    # domain: normalized state + time (start pinned at 0; reach() will set [0,h] per step)
    dom = [Interval(-1.0, 1.0) for _ in range(state_dim)] + [Interval(0.0, 0.0)]
    ref = tuple([0.0] * (state_dim + 1))

    tms = []

    # state components
    for i, iv in enumerate(bounds):
        mid = RIF(iv.midpoint())
        rad = RIF(iv.radius())
        sage_poly = ring(mid) + ring(rad) * gens[i]
        poly = Polynomial(_poly=sage_poly, _ring=ring)
        tm = TaylorModel(poly=poly, rem=Interval(0.0, 0.0), domain=list(dom), ref_point=ref, max_order=order)
        tm.sweep()
        tms.append(tm)

    # time component: use the last generator directly (tau)
    tau_poly = Polynomial(_poly=gens[-1], _ring=ring)
    tau_tm = TaylorModel(poly=tau_poly, rem=Interval(0.0, 0.0), domain=list(dom), ref_point=ref, max_order=order)
    tau_tm.sweep()
    tms.append(tau_tm)

    return TMVector(tms)


class TaskRunner:
    """central execution engine for the TERA"""

    @staticmethod
    def run(config: TaskConfig, print_results: bool = True, validate_results: bool = True, progress_bar: bool = False) -> ReachResult:
        print(f"\n[TaskRunner] Initializing {config.system_type} task: {config.name}")
        
        start_time = time.perf_counter()
        if config.system_type == "continuous":
            result = TaskRunner._run_continuous(config, progress_bar)
        elif config.system_type == "hybrid":
            result = TaskRunner._run_hybrid(config, progress_bar)
        elif config.system_type == "stochastic":
            result = TaskRunner._run_stochastic(config, validate_results=validate_results, progress_bar=progress_bar)
        else:
            raise ValueError(f"Unknown system type: {config.system_type}")
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        if result:
            result.runtime = duration

        if print_results and result:
            Report.print_header(result)

            if result.status == "SUCCESS":
                Report.print_final_set(result)

                if config.unsafe_sets:
                    result.safety_status = Report.check_safety(result, config.unsafe_sets)

                if result.system_type == 'stochastic':
                    Report.print_stochastic_stats(result)
                elif result.system_type == 'hybrid':
                    Report.print_hybrid_summary(result)
                
                if validate_results and (config.expected_final_bounds or config.expected_final_width):
                    Report.validate(result)

        return result
    
    @staticmethod
    def _run_continuous(config: TaskConfig, progress_bar: bool) -> ReachResult:
        engine = TMReach(
            ode_exprs=config.f_expr,
            state_vars=config.vars,
            order=config.order,
            cutoff_threshold=config.engine_params.get('cutoff_threshold', 1e-10),
            max_iterations=config.engine_params.get('max_iterations', 40),
            time_var=config.time_var,
            remainder_estimation=config.remainder_estimation,
            precondition_setup=config.engine_params.get('precondition_setup', 'QR'),
            fixed_step_mode=config.engine_params.get('fixed_step_mode', True),
            min_step=config.engine_params.get('min_step', 1e-6),
            max_step=config.engine_params.get('max_step', 0.5),
            step_grow_width_cap=config.engine_params.get('step_grow_width_cap', None),
            id_preserve_coupling_on_stagnation=config.engine_params.get('id_preserve_coupling_on_stagnation', False),
            id_stag_c_tol=config.engine_params.get('id_stag_c_tol', 1e-12),
            id_stag_s_tol=config.engine_params.get('id_stag_s_tol', 1e-10),
            id_stag_od_rel_tol=config.engine_params.get('id_stag_od_rel_tol', 1e-6),
            id_stag_lambda=config.engine_params.get('id_stag_lambda', 1.0),
            hybrid_id_full_linear_on_stagnation=config.engine_params.get('hybrid_id_full_linear_on_stagnation', False),
            shrink_wrap_mode=config.engine_params.get('shrink_wrap_mode', False),
            progress_bar=progress_bar
        )
        
        x0_tm = create_initial_box_continuous(len(config.vars), config.initial_set, config.order, time_var=config.time_var)
        
        flowpipe, status = engine.reach(
            config.engine_params.get('setting', 'single_step'),
            x0_tm, config.step_size, config.time_horizon
        )
        
        return ReachResult(flowpipe, "continuous", config.vars, vars(config), status)

    @staticmethod
    def _run_hybrid(config: TaskConfig, progress_bar: bool) -> ReachResult:
        automaton = config.engine_params.get('automaton')
        if automaton is None:
            raise ValueError("Hybrid task missing engine_params['automaton'].")

        ep = config.engine_params

        if hasattr(automaton, "initial_sets"):
            automaton.initial_sets = []

        initial_boxes = get_hybrid_initial_boxes(config.initial_set, ep)

        for box in initial_boxes:
            initial_tmv = create_initial_box_hybrid(config.vars, box, config.order)
            automaton.add_initial_state(config.initial_mode, initial_tmv)

        step_size = config.step_size
        min_step = ep.get('min_step', step_size * 0.1)
        max_step = ep.get('max_step', step_size * 2)
        tau_default = max(2.0 * float(min_step), 1.0 * float(step_size))

        engine_cfg = {
            'time_horizon': config.time_horizon,
            'max_jumps': ep.get('max_jumps', 10),
            'order': config.order,
            'state_vars': config.vars,
            'time_var': config.time_var,
            'urgent_jumps_mode': config.urgent_jumps_mode,
            'remainder_estimation': config.remainder_estimation,
            'precondition_setup': ep.get('precondition_setup', 'ID'),
            'setting': ep.get('setting', 'single_step'),
            'fixed_step_mode': ep.get('fixed_step_mode', False),
            'min_step': min_step,
            'max_step': max_step,
            'initial_step': step_size,
            'step_size': step_size,
            'intersection_method': ep.get('intersection_method', 'combined'),
            'aggregation_method': ep.get('aggregation_method', 'PCA'),
            'aggregation_threshold': ep.get('aggregation_threshold', 10),
            'aggregation_sample_mode': ep.get('aggregation_sample_mode', 'midpoint'),
            'remainder_contraction': ep.get('remainder_contraction', False),
            'cutoff_threshold': ep.get('cutoff_threshold', 1e-12),
            'max_iterations': ep.get('max_iterations', 40),
            'step_grow_width_cap': ep.get('step_grow_width_cap', 1.0),
            'id_preserve_coupling_on_stagnation': ep.get('id_preserve_coupling_on_stagnation', True),
            'id_stag_c_tol': ep.get('id_stag_c_tol', 1e-12),
            'id_stag_s_tol': ep.get('id_stag_s_tol', 1e-10),
            'id_stag_od_rel_tol': ep.get('id_stag_od_rel_tol', 1e-6),
            'id_stag_lambda': ep.get('id_stag_lambda', 1.0),
            'hybrid_id_full_linear_on_stagnation': ep.get('hybrid_id_full_linear_on_stagnation', False),
            'progress_bar': progress_bar,
        }

        split_cfg = ep.get("initial_split", None)
        if split_cfg and bool(split_cfg.get("enabled", False)):
            print(f"[TaskRunner] Hybrid initial-set splitting active: produced {len(initial_boxes)} initial box(es).")

        engine = HybridReach(automaton, engine_cfg)
        flowpipe = engine.compute_reachability()

        if flowpipe and len(flowpipe) > 0:
            time_horizon = float(config.time_horizon)
            tol = float(ep.get("horizon_tol", 1e-6))

            observed_t_max = None
            for seg in flowpipe:
                if not isinstance(seg, dict):
                    continue
                t_iv = seg.get("time_interval_abs", None)
                if t_iv is None:
                    continue
                try:
                    t_u = float(t_iv.upper)
                except Exception:
                    continue

                if observed_t_max is None or t_u > observed_t_max:
                    observed_t_max = t_u

            if observed_t_max is None:
                status = "FAILED"
            else:
                status = "SUCCESS" if observed_t_max + tol >= time_horizon else "PARTIAL"
        else:
            status = "FAILED"

        return ReachResult(flowpipe, "hybrid", config.vars, vars(config), status=status)
    @staticmethod
    def _run_stochastic(config: TaskConfig, validate_results: bool, progress_bar: bool) -> ReachResult:
        ep = config.engine_params

        try:
            engine = StochasticTMReach(
                delta=ep.get('delta', 0.001),
                g_exprs=ep.get('g_expr'),
                P_matrix=ep.get('P_matrix'),
                ode_exprs=config.f_expr,
                state_vars=config.vars,
                order=config.order,
                cutoff_threshold=ep.get('cutoff_threshold', 1e-10),
                max_iterations=ep.get('max_iterations', 40),
                time_var=config.time_var,
                fixed_step_mode=ep.get('fixed_step_mode', False),
                min_step=ep.get('min_step', 1e-6),
                max_step=ep.get('max_step', 0.5),
                step_grow_width_cap=ep.get('step_grow_width_cap', None),
                id_preserve_coupling_on_stagnation=ep.get('id_preserve_coupling_on_stagnation', False),
                id_stag_c_tol=ep.get('id_stag_c_tol', 1e-12),
                id_stag_s_tol=ep.get('id_stag_s_tol', 1e-10),
                id_stag_od_rel_tol=ep.get('id_stag_od_rel_tol', 1e-6),
                id_stag_lambda=ep.get('id_stag_lambda', 1.0),
                hybrid_id_full_linear_on_stagnation=ep.get('hybrid_id_full_linear_on_stagnation', False),
                shrink_wrap_mode=ep.get('shrink_wrap_mode', False),
                progress_bar=progress_bar,
                precondition_setup=ep.get('precondition_setup', 'QR'),
                remainder_estimation=config.remainder_estimation,
                amgf_eps=ep.get('amgf_eps', None)

            )
            
            x0_tm = create_initial_box_stochastic(len(config.vars), config.initial_set, config.order,  time_var=engine.time_var)
            flowpipe, status = engine.reach("single_step", x0_tm, config.step_size, config.time_horizon)
                        
            result = ReachResult(flowpipe, "stochastic", config.vars, vars(config), status)
            if getattr(engine, "L_np", None) is not None:
                result.validation_data["L_matrix"] = engine.L_np
            if getattr(engine, "use_weighted", False) and getattr(engine, "L_inv_sage", None) is not None:
                # compute diag(P^{-1}) via L_inv: P^{-1} = L^{-T} L^{-1}
                Linv = engine.L_inv_sage
                n = Linv.nrows()
                diag_vals = []
                for i in range(n):
                    s = RIF(0)
                    for k in range(n):
                        s += Linv[k, i] ** 2
                    upper = float(s.upper())
                    upper = math.nextafter(upper, math.inf)
                    diag_vals.append(upper)
                result.validation_data["diag_P_inv_upper"] = diag_vals
                result.validation_data["use_weighted"] = True
                if config.engine_params.get("P_matrix") is not None:
                    result.validation_data["P_matrix"] = config.engine_params.get("P_matrix")

            if validate_results and status == "SUCCESS" and flowpipe:
                num_traces = ep.get('mc_traces', 1000)
                sim = MonteCarloValidator(config.f_expr, ep.get('g_expr'), config.vars)
                tgrid, Xstoch, Xdet, X0 = sim.simulate_traces(config.initial_set, (0, config.time_horizon), num_traces=num_traces, dt=ep.get('mc_dt', None), return_deterministic=True, return_x0=True, seed=ep.get('mc_seed', None) )
                result.add_validation_traces((tgrid, Xstoch, Xdet, X0)) 
            
            return result

        except Exception as e:
            print(f"DEBUG: CRITICAL ERROR in _run_stochastic: {str(e)}")
            return ReachResult([], "stochastic", config.vars, vars(config), status="CRASHED")
