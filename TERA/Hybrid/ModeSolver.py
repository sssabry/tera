"""
Docstring for TERA.Hybrid.ModeSolver

child class of TMReach, to inject logic into the _post_step_hook()
acts as the mode-specific continuous reachability engine

enforces the mode invariants because flowpipes are only valid if compliant
"""
import copy
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

from TERA.TMFlow.TMReach import TMReach
from TERA.Hybrid.HybridModel import Mode, Condition, Transition
from TERA.Hybrid import Intersection

from TERA.TMCore.TMVector import TMVector
from TERA.TMCore.Interval import Interval

class ModeSolver(TMReach):
    """
    Docstring for ModeSolver
    inherits from TMReach (continuous reachability engine).
    primary goal: compute the continous flow while strictly enforcing invariants
    uses _post_step_hook() from TMReach to act as an observer and adjust flowpipes appropriately

    terminates integration once the invariants been violated 
    & exposes flowpipes for guard intersection checks
    """
    def __init__(self, mode: Mode, state_vars: List[Any], time_var: str = 't', config: Dict[str, Any] = None):
        """
        init solver by mapping hybrid config to continuous engine TMReach
        Initializes the solver by mapping the hybrid config to the continuous engine.
        """
        self.config = config if config is not None else {}
        
        super().__init__(
            ode_exprs=mode.ode_exprs,
            state_vars=state_vars,
            order=self.config.get('order', 4),
            cutoff_threshold=self.config.get('cutoff_threshold', 1e-10),
            max_iterations=self.config.get('max_iterations', 40),
            remainder_estimation=self.config.get('remainder_estimation'),
            time_var=time_var,
            precondition_setup=self.config.get('precondition_setup', 'QR'),
            min_step=self.config.get('min_step', 1e-6),
            max_step=self.config.get('max_step', 0.5),
            fixed_step_mode=self.config.get('fixed_step_mode', False),
            step_grow_width_cap=self.config.get('step_grow_width_cap', 1.0),
            id_preserve_coupling_on_stagnation=self.config.get('id_preserve_coupling_on_stagnation', True),
            id_stag_c_tol=self.config.get('id_stag_c_tol', 1e-12),
            id_stag_s_tol=self.config.get('id_stag_s_tol', 1e-10),
            id_stag_od_rel_tol=self.config.get('id_stag_od_rel_tol', 1e-6),
            id_stag_lambda=self.config.get('id_stag_lambda', 1.0),
            hybrid_id_full_linear_on_stagnation=self.config.get('hybrid_id_full_linear_on_stagnation', False)
        )
        self._hybrid_context = True
        
        self.state_vars = state_vars
        self.time_var = time_var
        self.current_mode = mode
        self.invariant = mode.invariant
        self.stop_integration = False
        self._constraint_cache = {}

    def propagate_mode_evolution(self, initial_set:TMVector, time_end: float, time_start: float) -> Tuple[List[Dict], str]:
        """
        computes the reachable set within the current mode for a bounded time horizzon
        equivalent to "computeFlowpipes()" in algorithm 12 of chen's thesis

        - initial_set: starting TMVector 'X_0'
        - time_horizon: maximum duriation to flow 'Delta'
        - max_jumps: used for tracking

        returns: 
            flowpipe_data: list of dicts containing flowpipe segments and metadata
            status: string indicating why integration stopped. options: TIME_LIMIT, INVARIANT_VIOLATED
        """

        # 1. reset flags
        self.stop_integration = False

        state_dim = len(self.state_vars)
        if len(initial_set.tms) != state_dim:
            raise ValueError(
                f"[ModeSolver] Contract violation: expected state-only TMVector dim={state_dim}, got dim={len(initial_set.tms)}"
            )
        if hasattr(initial_set, "domain") and initial_set.domain is not None and len(initial_set.domain) > state_dim:
            initial_set.domain = list(initial_set.domain[:state_dim])


        # 2. set initial step size 
        setting = self.config.get('setting', 'single_step')
        if setting != "single_step":
            print("Hybrid reachability is only currently available based on 'single_step' continuous reachability.")
        # 3. trigger TMReach
        flowpipes, status = self.reach(
            setting="single_step",
            initial_set=initial_set,
            initial_step=self.config.get('initial_step', self.min_step * 5),
            time_end=time_end,
            time_start=time_start
        )
        if flowpipes:
            final_time = float(flowpipes[-1]["time_interval_abs"].upper)
        else:
            final_time = float(time_start)

        print(
            f"DEBUG: Mode '{self.current_mode.name}' - Flow finished with status: {status}. "
            f"Segments: {len(flowpipes)}. Final t={final_time:.6g}"
        )
        return flowpipes, status
    
    def _advance_single_step(self, prev_tmv: TMVector, t_step: float, step_idx: int, 
                             rem_guess: List[Interval], time_start: float, current_order: int, retrying: bool) -> dict:
        """
        inject invariant aware step size modification into TMReach._advance_single_step
        if flowpipe crossing a boundary (UNKNOWN) reject the step and force shrink h and retry
        """
        result = super()._advance_single_step(prev_tmv, t_step, step_idx, rem_guess, time_start, current_order, retrying)
        # respect fixed step mode
        if self.fixed_step_mode:
            return result
        
        if result['success']:
            candidate_tmv = result['tmv']
            status = self._check_interval_satisfaction(candidate_tmv.bound(), self.invariant)
            
            # if boundary returns UNKNOWN
            # shrink the step and fake a failure to force the engine to decrease the size
            if status == "UNKNOWN" and t_step > (self.min_step * 2):
                return {'success': False, 'inflated': False}
            
        return result
    
    def _post_step_hook(self, step_info, h):
        """overrides TMReach._post_step_hook
        implements the filter logic of invariants from notion notes
        
        called immediately after a raw TM flowpipe is computed 
        but before its finalized in the flowpipe list (i.e. we can modify)
        
        - step_info: dict containing 'tmv' (computed flowpipe), 'time_interval', etc
        - h: step size used
        """

        # 1. extract computed flowpipe TMvector
        current_tmv = step_info['tmv']
        bounds = current_tmv.bound()

        quick_status = self._check_interval_satisfaction(bounds, self.invariant)

        if quick_status == "FULL":
            step_info['is_valid'] = True
        elif quick_status == "EMPTY":
            self.stop_integration = True 
            step_info['is_valid'] = False
        else:
            # reject step then try to contract
            step_info['is_valid'] = False
            if float(h) <= self.min_step * 2:
                contracted = Intersection.intersect_flowpipe_guard(
                    current_tmv,
                    self.invariant,
                    self.state_vars,
                    method="domain_contraction", 
                )

                if contracted is not None:
                    step_info['tmv'] = contracted
                    step_info['is_valid'] = True
                    self.stop_integration = False
            if contracted:
                step_info['tmv'] = contracted
                step_info['is_valid'] = True

            else:
                self.stop_integration = True
                step_info['is_valid'] = False

    def _check_interval_satisfaction(self, bounds: List[Interval], condition: Condition) -> str:
        """
        helper to evaluate constraints g(x) <= 0 over an interval box
        
        inputs:
            bounds: List of Intervals representing the bounding box of the flowpipe.
            condition: The Condition object (Invariant) containing constraints.
            
        returns:
            "EMPTY" if box is strictly outside (g(x) > 0 for some x).
            "FULL" if box is strictly inside (g(x) <= 0 for all x).
            "UNKNOWN" if overlapping boundary (requires contraction).
        """
        if not condition or not condition.constraints:
             return "FULL" # No invariant = always satisfied

        all_inside = True
        
        # iterate over every invariant g_i(x) <= 0
        for constraint in condition.constraints:
            # evaluate constriant over interval bounds
            val_interval = self._eval_constraint(constraint, bounds)
            
            # case 1: Strictly Violated
            # if the lower bound of g(x) is > 0, then g(x) is positive everywhere in the box.
            if val_interval.lower > 0:
                return "EMPTY"
            
            # case 2: Strictly Satisfied
            # if the upper bound of g(x) is <= 0, then g(x) is negative everywhere.
            if val_interval.upper <= 0:
                continue 
            
            # case 3: Overlap
            # lo <= 0 < hi. The boundary g(x)=0 passes through the box.
            all_inside = False
            
        if all_inside:
            return "FULL"
        else:
            return "UNKNOWN" # Mixed results, need contraction
        
    def _eval_constraint(self, expr: Any, bounds: List[Interval]) -> Interval:
        """
        evaluates a symbolic expression over a list of intervals
        similar compilation strategy to TMReach._generate_ode_evaluator
        """
        # context for evaluating symbolic expressions with intervals
        from sage.all import fast_callable, RIF
    
        try:
            key = (str(expr), tuple(str(v) for v in self.state_vars))
            f = self._constraint_cache.get(key)
            if f is None:
                f = fast_callable(expr, vars=self.state_vars, domain=RIF)
                self._constraint_cache[key] = f
            rif_bounds = [RIF(b.lower, b.upper) for b in bounds[:len(self.state_vars)]]
            res = f(*rif_bounds)
            return Interval(res.lower(), res.upper())
        except Exception as e:
            val = RIF(expr)
            return Interval(val.lower(), val.upper())
        
