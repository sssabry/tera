"""Continuous reachability engine for Taylor models."""

import numpy as np
from math import sin, cos, exp, sqrt, log
import math
from sage.all import fast_callable, SR, PolynomialRing, RIF, vector, matrix, jacobian
import copy
from tqdm.auto import tqdm
from typing import List, Callable, Tuple, Optional
import time
from contextlib import contextmanager
from collections import defaultdict

from TERA.TMCore.Interval import Interval
from TERA.TMCore.Polynomial import Polynomial
from TERA.TMCore.TMVector import TMVector
from TERA.TMCore.TaylorModel import TaylorModel
from TERA.TMCore.TMComputer import init_taylor_model
from TERA.TMFlow import Precondition, Picard, Remainder

class TMReach:
    """Compute continuous reachability using Taylor models."""
    def __init__(self, ode_exprs: List, state_vars: List, order: int = 4, cutoff_threshold: float = 1e-10,
            max_iterations:int =40, remainder_estimation: Optional[List[Interval]] = None, time_var: str  = 't',
            precondition_setup: str = "QR", min_step: float = 1e-6, max_step: float = 0.5, fixed_step_mode: bool = False,
            adaptive_order: bool = False, min_order: int = 2, max_order: int = 8, progress_bar: bool = False,
            step_grow_width_cap: Optional[float] = None,
            id_preserve_coupling_on_stagnation: bool = False,
            id_stag_c_tol: float = 1e-12,
            id_stag_s_tol: float = 1e-10,
            id_stag_od_rel_tol: float = 1e-6,
            id_stag_lambda: float = 1.0,
            hybrid_id_full_linear_on_stagnation: bool = False):
        """Initialize the reachability engine."""
        self.ode_exprs = ode_exprs
        self.state_vars = state_vars
        self.state_dim = len(state_vars)
        self.order = order
        self.cutoff_threshold = cutoff_threshold
        self.time_var = time_var
        self.max_iterations = max_iterations
        self.preconditioning = precondition_setup

        self.adaptive_order = adaptive_order
        self.min_order = min_order
        self.max_order = max_order

        # adaptive step params
        self.fixed_step_mode = fixed_step_mode
        self.min_step = min_step
        self.max_step = max_step
        self.step_grow_width_cap = 1e-2 if step_grow_width_cap is None else float(step_grow_width_cap)
        
        # persistent scaling factors for ID preconditioning while retrying
        self._fixed_scales = None
        self._fixed_inv_scales = None

        self._coords_cache_step = None
        self._coords_cache = None

        # optional ID-mode coupling preservation on stagnation (default off)
        self.id_preserve_coupling_on_stagnation = bool(id_preserve_coupling_on_stagnation)
        self.id_stag_c_tol = float(id_stag_c_tol)
        self.id_stag_s_tol = float(id_stag_s_tol)
        self.id_stag_od_rel_tol = float(id_stag_od_rel_tol)
        self.id_stag_lambda = float(id_stag_lambda)
        self.hybrid_id_full_linear_on_stagnation = bool(hybrid_id_full_linear_on_stagnation)
        self._id_prev_center = None
        self._id_prev_scales = None


        # profiling stats
        self.profile_times = defaultdict(float)
        self.profile_counts = defaultdict(int)

        # progress bar toggle
        self.progress_bar = progress_bar

        # convert symbolic ODE to a callable that accepts TMVector
        self.ode_rhs = self._generate_ode_evaluator(ode_exprs, state_vars)
        self.var_names = tuple([str(v) for v in self.state_vars]) + (self.time_var,)
        
        # generate a jacobian evaluator to speed up remainder refinement
        self.jacobian_evaluator = self._generate_jacobian_evaluator(ode_exprs, state_vars)

        if remainder_estimation is None:
            # default heuristic: small epsilon box
            self.current_remainder_guess = [Interval(-1e-4, 1e-4) for _ in range(self.state_dim)]
        else:
            self.current_remainder_guess = remainder_estimation

    @contextmanager
    def _measure(self, name: str):
        """timing context manager for speed profiling"""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.profile_times[name] += dt
            self.profile_counts[name] += 1

    def print_profile_report(self):
        print(f"\n{'='*50}")
        print(f"{'Section Name':<30} | {'Calls':<8} | {'Total (s)':<10} | {'Avg (ms)':<10}")
        print(f"{'-'*50}")
        
        # sort by total time descending
        sorted_stats = sorted(self.profile_times.items(), key=lambda x: x[1], reverse=True)
        
        for name, total_time in sorted_stats:
            count = self.profile_counts[name]
            avg_ms = (total_time / count) * 1000 if count > 0 else 0
            print(f"{name:<30} | {count:<8} | {total_time:<10.4f} | {avg_ms:<10.2f}")
        print(f"{'='*50}\n")

    def _generate_ode_evaluator(self, ode_exprs: List, vars: List) -> Callable[[TMVector], TMVector]:
        """
        compiles the symbolic ODE expressions into a callable function that supports TM arithmetic

        20/11 modification - testing lambda based mapping to TaylorModel class's intrinsic funcs
        """
        # wrapper functions to dispatch based on input type
        # makes sure sin(TM) calls TM.sin() & sin(float) calls math.sin()
        def _sin(x):
            return x.sin() if hasattr(x, 'sin') else sin(x)

        def _cos(x):
            return x.cos() if hasattr(x, 'cos') else cos(x)

        def _exp(x):
            return x.exp() if hasattr(x, 'exp') else exp(x)
            
        context = {
            'sin': _sin,
            'cos': _cos,
            'exp': _exp,
            'log': log, 
        }
        var_names = [str(v) for v in vars]
        time_var = self.time_var
        lambda_args = ", ".join(var_names + [time_var])

        # uses fast_callable for speed of compilation
        compiled_funcs = []
        for expr in ode_exprs:
            # fast_callable works with python objects that support arithmetic (like TMVector/TaylorModel)
            expr_str = str(expr)
            expr_str = expr_str.replace('^', '**')

            # create lambda source to implement intrinsic wrapping
            lambda_src = f"lambda {lambda_args}: {expr_str}"

            try:
                func = eval(lambda_src, context)
                compiled_funcs.append(func)
            except Exception as e:
                raise RuntimeError(f"Failed to compile ODE expression '{expr}': {e}")
            # f_fast = fast_callable(expr, vars=vars) 
            # compiled_funcs.append(f_fast)

        def evaluator(tm_vec: TMVector) -> TMVector:
            """
            evaluates f(tm_vec) using the compiled functions

            might contain 't' (if the ODE is time varying)
            """
            
            # extract tms to pass as arguments
            args = tm_vec.tms

            # ensure time variable is available for time-varying dynamics
            if len(args) == len(vars):
                # fall back to a constant time TM (t=0) if not present
                try:
                    time_tm = TaylorModel.from_constant(0.0, args[0])
                    args = list(args) + [time_tm]
                except Exception:
                    args = list(args)
            
            # if the input vector is larger than the symbolic vars list: slice args to match vars
            if len(args) > len(vars) + 1:
                args = args[:len(vars) + 1]
            
            result_tms = []
            
            # use first as template
            template_tm = tm_vec.tms[0]

            for f_func in compiled_funcs:
                # evaluates by calling on TMVector/TaylorModel's operator overrides
                try:
                    res = f_func(*args)
                except ZeroDivisionError as exc:
                    print(f"[DBG][ode_rhs][ZeroDivisionError] expr={f_func} exc={exc}")
                    return None

                
                # check if result is a TaylorModel (or TM-like object)
                if hasattr(res, 'poly') and hasattr(res, 'remainder'):
                    result_tms.append(res)
                else:
                    # result is a scalar constant -> promote it to a constant TM
                    var_names_str = template_tm.poly.ring.variable_names()
                    constant_tm = init_taylor_model(
                        my_func=res,
                        var_names=var_names_str,
                        domains=template_tm.domain,
                        order=template_tm.max_order,
                        ref_point=template_tm.ref_point,
                        expand_function=False
                    )
                    result_tms.append(constant_tm)
            
            return TMVector(result_tms)

        return evaluator
    
    def _generate_jacobian_evaluator(self, ode_exprs: List, vars: List) -> Callable[[List[Interval]], List[List[Interval]]]:
        """
        computes the symbolic jacobian adn compiles it for fast interval evaluation
        returns a function that takes a list of Intervals and returns a J matrix
        """
        time_var_obj = SR.var(self.time_var)
        eval_vars = vars + [time_var_obj]

        # 1. Compute Symbolic Jacobian using Sage
        try:
            J_sym = jacobian(ode_exprs, vars)
        except Exception as e:
            raise RuntimeError(f"SageMath Jacobian computation failed: {e}")

        # 2. Compile each cell of the matrix
        def _sin(x): return x.sin() if hasattr(x, 'sin') else sin(x)
        def _cos(x): return x.cos() if hasattr(x, 'cos') else cos(x)
        def _exp(x): return x.exp() if hasattr(x, 'exp') else exp(x)
        def _log(x): return x.log() if hasattr(x, 'log') else log(x)
        
        context = {'sin': _sin, 'cos': _cos, 'exp': _exp, 'log': _log}
        var_names = [str(v) for v in vars]
        lambda_args = ", ".join(var_names)

        rows = J_sym.nrows()
        cols = J_sym.ncols()
        compiled_matrix = [[None for _ in range(cols)] for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                expr = J_sym[r][c]
                expr_str = str(expr).replace('^', '**')
                lambda_src = f"lambda {lambda_args}: {expr_str}"
                try:
                    compiled_matrix[r][c] = eval(lambda_src, context)
                except Exception as e:
                    if expr.is_constant():
                        val = float(expr)
                        compiled_matrix[r][c] = lambda *args, v=val: v
                    else:
                        raise RuntimeError(f"Failed to compile Jacobian[{r},{c}]: {e}")

        def jacobian_evaluator(box: List[Interval]):
            """
            Evaluates the jacobian matrix over an interval box.
            Returns:
            - J_val: list[list[Interval]] on success
            - None on failure (e.g., division by an interval containing 0)
            """
            J_val = [[None for _ in range(cols)] for _ in range(rows)]
            for r in range(rows):
                for c in range(cols):
                    try:
                        val = compiled_matrix[r][c](*box)
                    except ZeroDivisionError as exc:
                        print(f"[DBG][jacobian][ZeroDivisionError] cell=({r},{c}) expr={J_sym[r][c]}")
                        for bi, iv in enumerate(box):
                            try:
                                contains_zero = bool(iv._interval.contains_zero())
                            except Exception:
                                contains_zero = "unknown"
                            print(f"[DBG][jacobian][box] i={bi} iv={iv} contains_zero={contains_zero}")
                        # IMPORTANT: do not crash the run; signal failure to caller
                        return None
                    except Exception as exc:
                        # Treat any unexpected eval error as failure too (safe)
                        print(f"[DBG][jacobian][EvalError] cell=({r},{c}) expr={J_sym[r][c]} exc={exc}")
                        return None

                    if not isinstance(val, Interval):
                        J_val[r][c] = Interval(val, val)
                    else:
                        J_val[r][c] = val

            return J_val

        return jacobian_evaluator
        
    def _hybrid_eval_tmv_jacobian_at_point(self, tmv: TMVector, state_dimension: int,
                                           center_now: List[float], time_iv: Optional[Interval]) -> Optional[np.ndarray]:
        """
        Hybrid-only helper: evaluate TMVector Jacobian at a specific point (center + time midpoint).
        Keeps continuous semantics unchanged by only being used when _hybrid_context is True.
        """
        try:
            state_names = [str(v) for v in self.state_vars]
            name_to_idx = {n: i for i, n in enumerate(state_names)}
            t_val = 0.0
            if time_iv is not None:
                try:
                    t_val = float(time_iv.midpoint())
                except Exception:
                    t_val = float(time_iv.lower + time_iv.upper) / 2.0

            J = np.zeros((state_dimension, state_dimension), dtype=float)

            for i in range(state_dimension):
                poly_obj = tmv.tms[i].poly.poly
                gens = poly_obj.parent().gens()
                gen_by_name = {str(g): g for g in gens}

                # build evaluation point in ring order
                vals = []
                for g in gens:
                    g_name = str(g)
                    if g_name == self.time_var:
                        vals.append(t_val)
                    elif g_name in name_to_idx:
                        vals.append(float(center_now[name_to_idx[g_name]]))
                    else:
                        vals.append(0.0)

                for j in range(state_dimension):
                    state_name = state_names[j]
                    if state_name not in gen_by_name:
                        J[i, j] = 0.0
                        continue

                    dpoly = poly_obj.derivative(gen_by_name[state_name])
                    val_obj = dpoly(*vals)
                    if hasattr(val_obj, "center"):
                        val = float(val_obj.center())
                    else:
                        val = float(val_obj)
                    J[i, j] = val

            return J
        except Exception:
            return None


    def reach(self, setting: str, initial_set: TMVector, initial_step: float, time_end: float, time_start: float = 0.0):
        """
        top-level directing function that directs you to the right reachability function
        according to architectural method wanted
        """
        if setting == "single_step":
            return self.reach_single_step(initial_set, initial_step, time_end, time_start)
        
        elif setting == "left_right":
            return self.reach_left_right(initial_set, initial_step, time_end, time_start)
        else:
            raise ValueError(f"Setting value {setting} is an invalid option. Accepted are: single_step or left_right")

    def reach_single_step(self, initial_set: TMVector, initial_step: float, time_end: float, time_start: float) -> List[TMVector]:
        """
        Compute the reachable set over a time horizon
        02/12 - introducing adapting time steps as an option!!
        closer mimics chen's approach and algorithms presented in his thesis

        input:
            - initial_set: initial set of states
            - t_step: time step size (delta)
            - time_end: final time horizon

        
        returns: list of TMVectors representing the flowpipes at each step.
        """
        flowpipe_data = []
        current_tmv = initial_set
        
        t_current = time_start
        h = initial_step  
        current_order = self.order
        
        self._fixed_scales = None
        
        if self.progress_bar:
            pbar = tqdm(total=float(time_end), desc="Reachability Analysis")
        step_count = 0
        status = "SUCCESS"

        retrying_step = False

        while t_current < time_end:

            # cap step size to not exceed the time_end
            if t_current + h > time_end:
                h = time_end - t_current
            
            # attempt a reachability step
            with self._measure("Full Step (Single)"):
                result = self._advance_single_step(current_tmv, h, step_count, self.current_remainder_guess, t_current, current_order, retrying=retrying_step)

            if result['success']:
                retrying_step = False
                # step succeeded! save results:
                final_tmv = result['tmv']
                # explicitly pin the plotted/stored TMV to the refined final TMV.
                step_info = dict(result['step_info'])
                step_info['tmv'] = final_tmv
                step_info['order'] = current_order

                t_start = t_current
                t_end = t_current + h
                step_info['time_interval_abs'] = Interval(t_start, t_end)
                
                # send data to observer classes
                with self._measure("Post Step Hook"):
                    self._post_step_hook(step_info, h)

                    if getattr(self, 'stop_integration', False):
                        status = "TERMINATED_BY_HOOK"
                        break

                flowpipe_data.append(step_info)
                
                # update state
                with self._measure("Eval at t_end"):
                    current_tmv = Precondition.evaluate_at_t_end(final_tmv, h, self.time_var)
                
                t_current = t_end
                step_count += 1
                if self.progress_bar:
                    pbar.update(float(h))


                # width-aware adaptive growth: grow only if verification was easy and bounds are controlled
                if not self.fixed_step_mode:
                    state_tms = final_tmv.tms[:self.state_dim] if len(final_tmv.tms) >= self.state_dim else final_tmv.tms
                    max_w = max([tm.bound().width() for tm in state_tms])
                    can_grow = (not result['inflated']) and (float(max_w) < self.step_grow_width_cap)
                    if can_grow:
                        h = min(h * 1.2, self.max_step)
                    else:
                        h = min(h, self.max_step)

                # if in adaptive order mode and remainder is very small, try decreasing order
                if self.adaptive_order and current_order > self.min_order:
                    rem_norm = max([tm.remainder.radius() for tm in final_tmv.tms])
                    if rem_norm < self.cutoff_threshold * 10:
                        current_order -= 1
                
            else:
                # step failed! handel according to step & order mode:
                if self.adaptive_order and current_order < self.max_order:
                    # try upping order instead of shrinking step
                    # retry same t_current and h 
                    current_order += 1
                    retrying_step = True
                    continue
                
                elif not self.fixed_step_mode:
                    # if already at max order or not adaptive, shrink h
                    h *= 0.5
                    current_order = self.order
                    retrying_step = True
                    if h < self.min_step:
                        status = "FAIL_MIN_STEP"
                        break
                else:
                    status = "FAIL_FIXED"
                    break
        if self.progress_bar:
            pbar.close()
        return flowpipe_data, status


    def _integrate_local_flow(self, x0_tmv: TMVector, t_step: float, rem_guess: List[Interval], 
                              inv_scales: List[float], time_start: float, 
                              current_order: int, max_attempts: int = 2) -> dict:
        """
        pure integration logic (picard iteration + remainder verification)
        sub-step reused by both the naive single_step and left_right architecture
        
        Note: x0_tmv may or may not include time as the last component,
        depending on whether this is called from single_step or left_right mode.
        """
        time_interval = Interval(0, t_step)
        dimension = len(x0_tmv.tms)
        # determine if has time
        has_time_component = (dimension > self.state_dim)
        state_dim = self.state_dim if has_time_component else dimension
        
        local_rem_guess = rem_guess[:state_dim]
        local_inv_scales = inv_scales[:state_dim]

        # 1. POLYNOMIAL FLOW
        with self._measure("Integrate: Picard"):
            poly_flow_tmv = Picard.compute_polynomial_flowpipe(
                x0=x0_tmv,
                ode_rhs=self.ode_rhs,
                order=current_order,
                cutoff_threshold=self.cutoff_threshold,
            )

        if poly_flow_tmv is None:
            return {
                'success': False,
                'inflated': False,
                'reason': 'PICARD_COMPUTE_FAILED'
            }

        # 2. REMAINDER VERIFICATION LOOP
        # a. evaluate J over the "rough" enclosure of the flowpipe from (1)
        with self._measure("Integrate: Jacobian"):
            poly_bounds = poly_flow_tmv.bound()

            # inflate by largest remainder guess & construct boz that covers likely flow
            eval_box = []
            for i in range(state_dim):
                rad = local_rem_guess[i].radius()

                inflated_interval = poly_bounds[i] + Interval(-rad, rad)
                eval_box.append(inflated_interval)

            jacobian_matrix = self.jacobian_evaluator(eval_box)
            if jacobian_matrix is None:
                # if cannot bound jacobian (e.g singularity/undefined) fail gracefully so adaptive logic can retyr
                return {'success': False, 'inflated': True}


        current_guess_intervals = local_rem_guess
        inflated = False
        
        # extract only state dimensions for the center vector
        c_l = [float(tm.poly.poly.constant_coefficient()) for tm in x0_tmv.tms[:state_dim]]

        for attempt in range(max_attempts):
            with self._measure("Integrate: Remainder Loop"):
                # prepare guess tmv
                scaled_estimation = []
                min_normalized_rem = 1e-12 
                
                for i in range(state_dim):
                    interval = current_guess_intervals[i]
                    scaled_est_rad = interval.radius() * local_inv_scales[i]
                    effective_rad = max(scaled_est_rad, 1e-12)
                    scaled_estimation.append(Interval(-effective_rad, effective_rad))

                # extract state components only for remainder computation
                state_poly_flow_tms = poly_flow_tmv.tms[:state_dim]
                state_poly_flow_tmv = TMVector(state_poly_flow_tms)

                if len(scaled_estimation) != self.state_dim:
                    scaled_estimation = scaled_estimation[:self.state_dim]
                guess_tmv = Remainder.compute_initial_guess(state_poly_flow_tmv, scaled_estimation)
                
                # verify
                with self._measure("Integrate: Verify"):
                    success, verified_tmv, trunc_errors = Remainder.verify_remainder(
                        guess_tmv=guess_tmv, x0=x0_tmv, ode_rhs=self.ode_rhs, 
                        time_var=self.time_var, time_step=time_interval, time_start=time_start,
                        order=current_order, cutoff_threshold=self.cutoff_threshold
                    )

                if verified_tmv is None:
                    return {
                        'success': False,
                        'inflated': inflated,
                        'needs_smaller_step': True,
                        'reason': 'PICARD_EVAL_FAILED'
                    }

                if success:
                    # take snapshot of verified rems to propagate as next guess
                    verified_rems_snapshot = [
                        Interval(tm.remainder.lower, tm.remainder.upper)
                        for tm in verified_tmv.tms[:state_dim]
                    ]
                    # refine
                    with self._measure("Integrate: Refine"):
                        final_tmv = Remainder.refine_remainder(
                            verified_tmv=verified_tmv, x0=x0_tmv, ode_rhs=self.ode_rhs, 
                            time_var=self.time_var, time_step=time_interval, 
                            order=current_order, truncation_errors=trunc_errors, 
                            max_refinements=self.max_iterations, jacobian=jacobian_matrix
                        )
                    
                    return {
                        'success': True, 
                        'tmv': final_tmv, 
                        'verified_tmv': verified_tmv,
                        'verified_rems_snapshot': verified_rems_snapshot,
                        'c_l': c_l,
                        'inflated': inflated,
                        'jacobian': jacobian_matrix,
                        'order_used': current_order
                    }
                else:
                    # verififcation failed? computed error > guess -> 
                    # use the computed error from the failed check as the new guess

                    # make sure it hasnt exploded
                    if any(tm.remainder.is_nan for tm in verified_tmv.tms):
                        break 
                    
                    # convert computed remainder to global scale & multiply by safety factor
                    new_guess = []

                    MAX_SCALE_WARN = 1e20
                    SAFETY = 2.0
                    for i in range(state_dim):
                        norm_rad = verified_tmv.tms[i].remainder.radius()
                        raw_rad = norm_rad / local_inv_scales[i] if local_inv_scales[i] != 0 else float('inf')
                        # keep raw guess consistent with normalized minimum remainder radius (1e-12)
                        raw_min = 1e-12 / local_inv_scales[i] if local_inv_scales[i] != 0 else 1e-12
                        raw_rad = max(float(raw_rad), float(raw_min))
                        # propose new guess in normalized coordinates (same space as current_guess_intervals)
                        ng = Interval(-SAFETY * raw_rad, SAFETY * raw_rad)

                        S_val = 1.0 / local_inv_scales[i] if local_inv_scales[i] != 0 else float('inf')
                        if not math.isfinite(S_val) or abs(S_val) > MAX_SCALE_WARN:
                            return {'success': False, 'inflated': True}
                        new_guess.append(ng)
                    
                    current_guess_intervals = new_guess
                    inflated = True

        return {'success': False, 'inflated': inflated}

    def _compute_local_coordinates(self, target_tmv: TMVector, t_step: float, 
                                   step_idx: int, t_global_start: float,
                                   retrying: bool = False) -> dict:
        """
        calculates the local coordinate system based on a target tmVector
        encapsulate original single_step logic for x0_new construction 
        """
        dimension = len(target_tmv.tms)
        state_dimension = self.state_dim
        normalized_domain = [Interval(-1.0, 1.0) for _ in range(state_dimension)]

        # determine center & shift
        with self._measure("Coords: Shift"):
            tmv_deviation, center_c0 = Precondition.shift_to_origin(target_tmv)
        
        # slice if time already injected
        if dimension > state_dimension:
            state_deviation = TMVector(tmv_deviation.tms[:state_dimension])
            state_domain = target_tmv.domain[:state_dimension]
        else:
            state_deviation = tmv_deviation
            state_domain = target_tmv.domain

        # branch according to preconditioning (ID vs QR)
        x0_new = None
        scales = []
        inv_scales = []
        Q_matrix = np.eye(dimension)
        A_lin = None
        A_lin_inv = None

        if self.preconditioning == "QR":
            with self._measure("Coords: QR"):
                x0_new, scales, inv_scales, _, Q_matrix = Precondition.qr_preconditioning(
                    tmv_pre=target_tmv,
                    t_step=t_step,
                    current_domain=target_tmv.domain,
                    time_var=self.time_var,
                    var_names=self.var_names,
                    time_start=t_global_start,
                    state_dim=state_dimension
                )

            s_min = 1e-6
            scales = [max(s, s_min) for s in scales]
            inv_scales = [1.0/s for s in scales]

        elif self.preconditioning == "ID":
            update_enabled = not retrying
            range_of_x0 = Precondition.determine_magnitude(state_deviation, state_domain)
            desired = [max(float(iv.radius()), 1e-16) for iv in range_of_x0]

            # stagnation metrics (only used when id_preserve_coupling_on_stagnation is enabled)
            center_now = [float(v) for v in center_c0[:state_dimension]]
            scales_now = [float(s) for s in desired]

            dc_inf = None
            ds_rel = None
            offdiag_rel = None
            J_state = None
            diag_max = None
            offdiag_max = None

            if self._id_prev_center is not None:
                dc_inf = max(abs(c - pc) for c, pc in zip(center_now, self._id_prev_center))
            if self._id_prev_scales is not None:
                eps = 1e-16
                ds_rel = max(abs(s - ps) / max(abs(ps), eps) for s, ps in zip(scales_now, self._id_prev_scales))

            try:
                if getattr(self, "_hybrid_context", False):
                    time_iv = None
                    if hasattr(target_tmv, "domain") and target_tmv.domain is not None and len(target_tmv.domain) > state_dimension:
                        time_iv = target_tmv.domain[state_dimension]
                    J_state = self._hybrid_eval_tmv_jacobian_at_point(target_tmv, state_dimension, center_now, time_iv)
                    if J_state is None:
                        J_full = target_tmv.get_jacobian()
                        J_state = np.array(J_full[:state_dimension, :state_dimension], dtype=float)
                else:
                    J_full = target_tmv.get_jacobian()
                    J_state = np.array(J_full[:state_dimension, :state_dimension], dtype=float)
                diag = np.diag(J_state)
                od = J_state.copy()
                np.fill_diagonal(od, 0.0)
                offdiag_max = float(np.max(np.abs(od))) if od.size else 0.0
                diag_max = float(np.max(np.abs(diag))) if diag.size else 0.0
                offdiag_rel = offdiag_max / max(diag_max, 1e-16)
            except Exception:
                J_state = None
                offdiag_rel = None
                diag_max = None
                offdiag_max = None

            # init once
            if self._fixed_scales is None:
                scales = []
                inv_scales = []
                floor = 1e-16
                for s in desired:
                    if s <= floor:
                        scales.append(floor)
                        inv_scales.append(1.0)
                    else:
                        scales.append(s)
                        inv_scales.append(1.0 / s)
                self._fixed_scales = scales
                self._fixed_inv_scales = inv_scales

            else:
                # sticky rescaling trigger
                scales = list(self._fixed_scales)
                inv_scales = list(self._fixed_inv_scales)

                # only update if scale mismatch is severe
                # TODO change these to be user configurable
                ratio_hi = 3.0
                ratio_lo = 1.0/3.0
                cap = 1.5
                floor = 1e-16
                inv_floor_deadband = 1e-8

                if update_enabled:
                    # compute per-dim mismatch ratio: desired/current
                    ratios = []
                    for s_cur, s_des in zip(scales, desired):
                        s_cur = max(float(s_cur), floor)
                        s_des = max(float(s_des), floor)
                        r = s_des / s_cur
                        if s_cur <= inv_floor_deadband and s_des <= inv_floor_deadband:
                            r = 1.0

                        ratios.append(r)

                    # decide whether to rescale (any dim out of tolerance)
                    needs_update = any((r > ratio_hi) or (r < ratio_lo) for r in ratios)

                    if needs_update:
                        # bounded update: clamp each multiplicative change
                        new_scales = []
                        for s_cur, r in zip(scales, ratios):
                            r_clamped = min(max(r, 1.0 / cap), cap)
                            new_scales.append(max(float(s_cur) * r_clamped, floor))

                        scales = new_scales
                        inv_scales = []
                        for s in scales:
                            inv = 1.0 / max(s, inv_floor_deadband)
                            inv_scales.append(inv)

                        self._fixed_scales = scales
                        self._fixed_inv_scales = inv_scales

            # hybrid-only: treat scale changes as "stagnant" unless fixed scales actually update
            if getattr(self, "_hybrid_context", False) and (self._fixed_scales is not None):
                if self._id_prev_scales is not None:
                    eps = 1e-16
                    ds_rel = max(
                        abs(s - ps) / max(abs(ps), eps)
                        for s, ps in zip(self._fixed_scales, self._id_prev_scales)
                    )
                else:
                    ds_rel = None

            stagnating = False
            A_lin_pre = None
            A_lin_inv_pre = None
            use_full_linear = bool(self.hybrid_id_full_linear_on_stagnation)
            A_lin = None
            A_lin_inv = None
            J_off = None

            if (use_full_linear and self.id_preserve_coupling_on_stagnation and (J_state is not None)
                and (dc_inf is not None) and (ds_rel is not None) and (offdiag_rel is not None)):
                c_scale = max(1.0, max(abs(c) for c in center_now))
                c_tol = self.id_stag_c_tol * c_scale
                s_tol = self.id_stag_s_tol
                od_tol = self.id_stag_od_rel_tol
                stagnating = (dc_inf <= c_tol) and (ds_rel <= s_tol) and (offdiag_rel >= od_tol)

                if stagnating:
                    lam = max(0.0, min(1.0, float(self.id_stag_lambda)))
                    J_off = J_state.copy()
                    np.fill_diagonal(J_off, 0.0)
                    A_lin_pre = np.diag([float(s) for s in desired[:state_dimension]]) + lam * J_off
                    try:
                        A_lin_inv_pre = np.linalg.inv(A_lin_pre)
                    except np.linalg.LinAlgError:
                        try:
                            A_lin_inv_pre = np.linalg.inv(A_lin_pre + (1e-14 * np.eye(state_dimension)))
                        except np.linalg.LinAlgError:
                            A_lin_inv_pre = None

            # normalization switch: full linear inverse when available; otherwise diagonal path
            if use_full_linear and (A_lin_inv_pre is not None):

                state_deviation = Precondition.apply_linear_map_to_tmv(
                    state_deviation, A_lin_inv_pre, state_dimension
                )
                range_of_x0 = Precondition.determine_magnitude(state_deviation, state_domain)
                desired = [max(float(iv.radius()), 1e-16) for iv in range_of_x0]
                scales_now = [float(s) for s in desired]
            elif use_full_linear:
                # fallback: commit diagonal-mapped deviation
                inv_scales_diag = []
                for s in desired:
                    s = max(float(s), 1e-16)
                    inv_scales_diag.append(1.0 / s)

                state_deviation = Precondition.apply_diagonal_inv_scales_to_tmv(
                    state_deviation, inv_scales_diag, state_dimension
                )
                range_of_x0 = Precondition.determine_magnitude(state_deviation, state_domain)
                desired = [max(float(iv.radius()), 1e-16) for iv in range_of_x0]
                scales_now = [float(s) for s in desired]
            with self._measure("Coords: Construct ID"):
                time_interval = Interval(t_global_start, t_global_start + t_step)
                local_time_interval = Interval(0.0, t_step)
                x0_new = Precondition.construct_new_initial_vars(
                    state_dimension=state_dimension, 
                    scale_factors_S=scales, 
                    center_c0=center_c0[:state_dimension], 
                    max_order=self.order,
                    time_domain=local_time_interval,
                    var_names=self.var_names,
                    time_start=0.0
                )
                x0_new.domain = normalized_domain + [local_time_interval]

            # preserve off-diagonal coupling only under conservative stagnation trigger
            if use_full_linear and stagnating and (J_off is not None):
                lam = max(0.0, min(1.0, float(self.id_stag_lambda)))
                A_lin = np.diag([float(s) for s in scales[:state_dimension]]) + lam * J_off

                try:
                    A_lin_inv = np.linalg.inv(A_lin)
                except np.linalg.LinAlgError:
                    try:
                        A_lin_inv = np.linalg.inv(A_lin + (1e-14 * np.eye(state_dimension)))
                    except np.linalg.LinAlgError:
                        A_lin_inv = None
                        A_lin = None

                # inject A_lin into x0_new: x = c + A_lin * xi
                if A_lin is not None:
                    for i in range(state_dimension):
                        tm_i = x0_new.tms[i]
                        ring = tm_i.poly.ring
                        gens = ring.gens()
                        for j in range(state_dimension):
                            if i == j:
                                continue
                            aij = float(A_lin[i, j])
                            if abs(aij) == 0.0:
                                continue
                            tm_i.poly = Polynomial(
                                _poly=(tm_i.poly.poly + (RIF(aij) * gens[j])),
                                _ring=tm_i.poly.ring
                            )

            self._id_prev_center = center_now
            if getattr(self, "_hybrid_context", False):
                self._id_prev_scales = list(self._fixed_scales) if self._fixed_scales is not None else scales_now
            else:
                self._id_prev_scales = scales_now

        return {
            'x0_new': x0_new,
            'scales': scales,
            'inv_scales': inv_scales, 
            'Q_matrix': Q_matrix,
            'center': center_c0,
            'A_lin': A_lin,
            'A_lin_inv': A_lin_inv
        }

    
    def _advance_single_step(self, prev_tmv: TMVector, t_step: float, step_idx: int, 
                             rem_guess: List[Interval], time_start: float,
                             current_order: int, retrying: bool=False) -> dict:
        
        # 1. setup coordinates
        if retrying and self._coords_cache_step == step_idx and self._coords_cache is not None:
            coords = self._coords_cache
        
        else:
            with self._measure("Compute Local Coords"):
                coords = self._compute_local_coordinates(prev_tmv, t_step, step_idx, time_start, retrying=retrying)
                self._coords_cache_step = step_idx
                self._coords_cache = coords

        # 2. integrate
        with self._measure("Integrate Local Flow"):
            result = self._integrate_local_flow(
                x0_tmv=coords['x0_new'], 
                t_step=t_step, 
                rem_guess=rem_guess,
                inv_scales=coords['inv_scales'],
                # respect fixed step mode constraints for retries
                max_attempts=1 if self.fixed_step_mode else 2,
                time_start=time_start,
                current_order=current_order
            )
        
        # 3. package result
        if result['success']:
            # clear cache
            self._coords_cache_step = None
            self._coords_cache = None

            if self.preconditioning == "QR":
                A_l = coords['Q_matrix'] @ np.diag(coords['scales'])
            else:
                A_l = np.diag(coords['scales'])

            result['step_info'] = {
                'tmv': result['tmv'], 
                'c_l': result['c_l'], 
                'A_l': A_l,
                'global_center': coords['center'],
                'jacobian': result['jacobian']
            }

            # carry forward a physical/raw remainder guess for the next step
            try:
                new_guess = []
                inv_scales = coords['inv_scales'][:self.state_dim]
                SAFETY_NEXT = 1.8
                rems = result.get('verified_rems_snapshot', None)
                if rems is None:
                    guess_source_tmv = result.get('verified_tmv', None)
                    if guess_source_tmv is None:
                        guess_source_tmv = result['tmv']
                    rems = [tm.remainder for tm in guess_source_tmv.tms[:self.state_dim]]
                for i in range(self.state_dim):
                    norm_rad = rems[i].radius()
                    norm_rad = max(0.0, float(norm_rad))
                    raw_rad = float(norm_rad) / float(inv_scales[i]) if inv_scales[i] != 0 else float(norm_rad)
                    raw_rad = max(0.0, raw_rad)
                    new_guess.append(Interval(-SAFETY_NEXT * raw_rad, SAFETY_NEXT * raw_rad))
                self.current_remainder_guess = new_guess
            except Exception:
                pass
            
        return result
    
    def reach_left_right(self, initial_set: TMVector, initial_step: float, time_end: float, time_start: float) -> List[TMVector]:
        """
        left-right architecutre based reachability loop, following the structue
        in Florians preconditioning paper (Listing 1) and Makino/Berz's supression of the wrapping effect paper

        architecture:
        - Left Model (L): Local coordinate system (Affine Map) - Integrated at each step
        - Right Model (R): Accumulated flow dependency - Composed at each step
        """

        flowpipe_data = []
        status = "SUCCESS"

        t_current = time_start
        h = initial_step
        current_order = self.order

        # 0. initialization
        # construct L_0 and R_0 such that L_0(R_0(x)) = initial_set
        # R_0 = Identity on [-1, 1]
        # L_0 maps [-1, 1] to the physical initial set Y0
        bbox = initial_set.bound()
        m_0 = [iv.midpoint() for iv in bbox]
        r_0 = [iv.radius() for iv in bbox]
        
        time_int_0 = Interval(time_start, time_start)
        state_dim = self.state_dim

        # construct L_0
        left_tmv = Precondition.construct_affine_left_model(
            dimension=state_dim,
            Q=np.eye(state_dim),
            scale_factors_S=r_0,
            midpoints_m=[0.0] * state_dim,
            center_c0=m_0,
            max_order=self.order,
            domain=[Interval(-1.0, 1.0)] * state_dim + [Interval(0.0, 0.0)],  # local tau
            time_start=0.0,
            var_names=self.var_names,
        )
        
        # construct R_0
        right_tmv = Precondition.construct_new_initial_vars(
            state_dimension=state_dim,
            scale_factors_S=[1.0]*state_dim,
            center_c0=[0.0]*state_dim,
            max_order=self.order,
            time_domain=time_int_0,
            var_names=self.var_names,
            time_start=time_start
        )

        if self.progress_bar:
            pbar = tqdm(total=float(time_end), desc="L-R Reachability")

        while t_current < time_end:
            if t_current + h > time_end:
                h = time_end - t_current

            # perform the step
            with self._measure("Full Step (L-R)"):
                result = self._advance_left_right_step(left_tmv, right_tmv, h, time_start, current_order)

            if result['success']:
                # successful? update state for next step
                
                with self._measure("Vis Composition"):
                    composite_tmv = self._construct_visualization_tm(
                        L_integrated=result['L_integrated'],
                        L_next=result['L_next'],
                        R_prev=right_tmv,
                        h=h,
                        t_current=t_current
                    )

                step_info = {
                    'tmv': composite_tmv, 
                    'time_interval_abs': Interval(t_current, t_current + h),
                    'c_l': result['c_l'],
                    'A_l': result['A_l'],
                    'precondition_matrix': result['Q_matrix'],
                    'jacobian': result['jacobian'],
                    'order': current_order
                }
                flowpipe_data.append(step_info)

                left_tmv = result['L_next']
                right_tmv = result['R_next']

                # send data to observer classes
                with self._measure("Post Step Hook"):
                    self._post_step_hook(step_info, h)

                    if getattr(self, 'stop_integration', False):
                        status = "TERMINATED_BY_HOOK"
                        break

                t_current += h
                if self.progress_bar:
                    pbar.update(float(h))
                
                # control step size
                if not self.fixed_step_mode and not result.get('inflated', False):
                    h = min(h * 1.2, self.max_step)

                # control order
                if self.adaptive_order:
                    # check remainder of L_next model (the normalized coordinate system)
                    max_rem = max([tm.remainder.radius() for tm in left_tmv.tms])
                    if max_rem < self.cutoff_threshold * 0.1 and current_order > self.min_order:
                        current_order -= 1
            
            # FAILURE BLOCK
            else:
                if self.adaptive_order and current_order < self.max_order:
                    current_p += 1
                    continue
                if self.fixed_step_mode:
                    print(f"\nStep failed at t={t_current}. Fixed step mode abort.")
                    status = "FAIL_FIXED"
                    break
                
                h *= 0.5
                if h < self.min_step:
                    print(f"\nMin step reached at t={t_current}. Abort.")
                    status = "FAIL_MIN_STEP"
                    break
        if self.progress_bar:
            pbar.close()
        return flowpipe_data, status

    def _advance_left_right_step(self, L_prev: TMVector, R_prev: TMVector, h: float, t_start: float,
                                 current_order: int) -> dict:
        """
        executes one step of the left-right architecture's algorithm
        integrate -> decompose -> compose -> normalize
        """

        # 1. integrate the left model using picard
        state_dim = self.state_dim

        local_tol = 1e-2
        rem_guess = [Interval(-local_tol, local_tol)] * state_dim

        l_bounds = L_prev.bound()[:state_dim]
        scales = [b.radius() for b in l_bounds]
        inv_scales = [(1.0 if s < 1e-12 else 1.0 / s) for s in scales]

        L_integration = copy.deepcopy(L_prev)

        integration_time_domain = Interval(0, h)
        # adjust time domain for all tms
        for tm in L_integration.tms:
            tm.domain[-1] = integration_time_domain
            # make sure ref point is consistent
            rp = list(tm.ref_point)
            if len(rp) < state_dim + 1:
                rp = rp + [0.0] * ((state_dim + 1) - len(rp))
            rp[-1] = 0.0
            tm.ref_point = tuple(rp)
        
        with self._measure("Integrate Local Flow"):
            int_result = self._integrate_local_flow(
                x0_tmv=L_integration,
                t_step=h,
                rem_guess=rem_guess,
                inv_scales=inv_scales,
                time_start=0.0,
                max_attempts=3,
                current_order=current_order
            )

        if not int_result['success']:
            return {'success': False}
        
        L_integrated = int_result['tmv']
        inflated = bool(int_result.get("inflated", False))

        # 2. decompose the flow
        try:
            with self._measure("L-R: Decompose"):
                M_flow = Precondition.evaluate_at_t_end(L_integrated, float(h), self.time_var)
                # shift (extract center c0 and deviation M_centered = M_flow - c0)
                M_centered, center_c = Precondition.shift_to_origin(M_flow)

                A_full = M_centered.get_jacobian()
                A = A_full[:state_dim, :state_dim]

                Q = Precondition.compute_qr_matrix(A)
                Q_poly = Precondition.rotate_tmv(M_centered, Q)

        except Exception as e:
            print(f"DEBUG: Decomposition crashed: {e}")
            return {'success': False}

        # 3. perform shrink wrapping step w/ composition
        try:
            with self._measure("L-R: Composition"):
                Q_poly_state = TMVector(Q_poly.tms[:state_dim])

                time_tm = TaylorModel.from_constant(0.0, R_prev.tms[0])

                replacements = [copy.deepcopy(tm) for tm in R_prev.tms[:state_dim]] + [time_tm]
                for tm in replacements:
                    tm.domain = copy.deepcopy(R_prev.domain)
                    tm.ref_point = copy.deepcopy(R_prev.ref_point)

                T_state = Q_poly_state.compose(replacements)
                T_target = TMVector(list(T_state.tms) + [copy.deepcopy(R_prev.tms[-1])])

        except Exception as e:
            print(f"DEBUG: Composition crashed: {e}")
            return {'success': False}

        # 4. scale to normalized domain
        with self._measure("L-R: Normalize"):
            range_of_yr = T_state.bound()  # list[Interval] length = state_dim
            midpoints_m = [iv.midpoint() for iv in range_of_yr]
            scale_factors_S = []
            inv_scale_factors_S = []

            MIN_RAD = 1e-14
            for iv in range_of_yr:
                rad = iv.radius()
                rad = float(rad)
                if rad < MIN_RAD:
                    rad = MIN_RAD
                scale_factors_S.append(rad)
                inv_scale_factors_S.append(1.0 / rad)

            # 5. construct left/right models for next step
            R_next = Precondition.normalize_right_model(
                tm_target=T_target,
                midpoints_m=midpoints_m,
                inv_scale_factors_S=inv_scale_factors_S
            )

            next_time_domain_local = Interval(0.0, 0.0)
            norm_domain = [Interval(-1.0, 1.0)] * state_dim + [next_time_domain_local]

            L_next = Precondition.construct_affine_left_model(
                dimension=state_dim,
                Q=Q,
                scale_factors_S=scale_factors_S,
                midpoints_m=midpoints_m,
                center_c0=center_c,
                max_order=self.order,
                domain=norm_domain,
                time_start=0.0,
                var_names=self.var_names
            )

            try:
                t_abs_current = float(R_prev.domain[-1].lower)
            except Exception:
                t_abs_current = 0.0

            t_abs_next = t_abs_current + float(h)

            # enforce absolute time in R_next
            for tm in R_next.tms:
                dom = list(tm.domain)
                rp = list(tm.ref_point)
                if len(dom) < state_dim + 1:
                    dom = dom + [Interval(0.0, 0.0)] * ((state_dim + 1) - len(dom))
                if len(rp) < state_dim + 1:
                    rp = rp + [0.0] * ((state_dim + 1) - len(rp))
                dom[-1] = Interval(t_abs_next, t_abs_next)
                rp[-1] = t_abs_next
                tm.domain = dom
                tm.ref_point = tuple(rp)

            R_next.domain = copy.deepcopy(R_next.tms[0].domain)
            R_next.ref_point = copy.deepcopy(R_next.tms[0].ref_point)

            A_l = Q @ np.diag(scale_factors_S)
        
        return {
            'success': True,
            'L_next': L_next,
            'R_next': R_next,
            'c_l': center_c,
            'inflated': inflated,
            'Q_matrix': Q,
            'A_l' : A_l,
            'L_integrated': L_integrated,
            'jacobian': int_result['jacobian'],
            'order': current_order
        }

    def _construct_visualization_tm(self, L_integrated: TMVector, L_next: TMVector, 
                                  R_prev: TMVector, h: float, t_current: float) -> TMVector:
        """
        constructs the time-dependent TM for visualization/plotting Flow(t) = L_integrated( R_prev(x0), t )
        """
        state_dim = self.state_dim
        try:
            # prepare time polynomial
            proto_tm = L_integrated.tms[0]
            ring = proto_tm.poly.ring
            gens = ring.gens()
            
            # build local time tm w [0,h] domain and ref_point 0
            tau_domain = copy.deepcopy(R_prev.domain)
            tau_ref = list(copy.deepcopy(R_prev.ref_point))

            required_dim = proto_tm.poly.dim
            if len(tau_domain) < required_dim:
                tau_domain = tau_domain + [Interval(0.0, 0.0)] * (required_dim - len(tau_domain))
            if len(tau_ref) < required_dim:
                tau_ref = tau_ref + [0.0] * (required_dim - len(tau_ref))

            tau_domain[-1] = Interval(0.0, float(h))
            tau_ref[-1] = 0.0

            tau_poly = Polynomial(_poly=gens[-1], _ring=ring)
            tau_tm = TaylorModel(
                poly=tau_poly,
                rem=Interval(0.0),
                domain=tau_domain,
                ref_point=tuple(tau_ref),
                max_order=self.order
            )
            replacements = []
            for i in range(state_dim):
                tm_copy = copy.deepcopy(R_prev.tms[i])
                # force domain/ref_point compatibility with tau
                tm_copy.domain = copy.deepcopy(tau_domain)
                tm_copy.ref_point = tuple(tau_ref)
                replacements.append(tm_copy)     

            # append tau as time replacement
            replacements.append(tau_tm)           
            composed_full = L_integrated.compose(replacements)
            return TMVector(composed_full.tms[:state_dim])

        except Exception as e:
            self.logger.debug("Vis Composition failed at t=%s: %s", t_current, e)
            # fallback: Compose the affine next-step models (Snapshot at t=h)
            try:
                fallback = L_next.compose([copy.deepcopy(tm) for tm in R_prev.tms])
                return TMVector(fallback.tms[:state_dim])
            except Exception:
                # last resort: return state part of R_prev (should not happen in a correct run)
                return TMVector([copy.deepcopy(tm) for tm in R_prev.tms[:state_dim]])
        
    def _post_step_hook(self, step_info: dict, h: float):
        """ hook for subclasses to perform side-calculations like the stochastic extension
        without polluting the main continuous reachability engline's logic
        
        works like an observer (211) and this function exposes the intermediate values"""

        pass
