"""
Docstring for TERA.Hybrid.HybridReach

orchestration layer of the hybrid systems reachability engine-extension

implements the worklist algorithm (algorithm 12) from chen's thesis
manages the global time horizon & handles the discrete switching logic

"""
from collections import deque
from typing import List, Dict, Tuple, Any
import copy
from sage.all import PolynomialRing, RIF

from TERA.Hybrid.HybridModel import HybridAutomaton, Mode
from TERA.Hybrid.ModeSolver import ModeSolver
from TERA.TMCore.TMVector import TMVector
from TERA.TMCore.Interval import Interval
from TERA.TMCore.Polynomial import Polynomial
from TERA.TMCore.TaylorModel import TaylorModel
from TERA.Hybrid import Intersection

class HybridReach:
    """Orchestrate hybrid reachability over a HybridAutomaton."""
    def __init__(self, automaton: HybridAutomaton, config: Dict[str, Any]):
        """
        Initializes the Hybrid Reachability Engine.
        - time_horizon (Delta): maximum global time elapsed
        - max_jumps (I): Maximum number of allowed discrete transitions
        - order: Taylor Model polynomial order (passed to ModeSolver).
        - cutoff_threshold: Magnitude for sweeping coefficients into remainder.
        - aggregation_method: "PCA" or "CRITICAL" for joining intersection sets
        - aggregation_threshold: Max number of segments to collect before forced aggregation.
        """
        self.automaton = automaton
        self.config = config
        self.state_vars = config.get('state_vars')
        self.time_var = config.get('time_var', 't')
        # store flowpipes as List of (ModeName, FlowpipeSegment)
        self.reachable_set = []
        self._visited_workitems = {} # Dict[Tuple[str,int], List[Tuple[float, List[Interval]]]]

    def compute_reachability(self) -> List[Tuple[str, TMVector]]:
        """
        implements the algorithm 12 of chen's thesis
        """
        queue = deque()
        total_horizon = self.config.get('time_horizon', 10.0)
        max_jumps = self.config.get('max_jumps', 5)
        urgent_jumps_mode = bool(self.config.get("urgent_jumps_mode", False))


        # 1. init queue, where each task (mode, initTMV, globalTime, remainingJumps)
        for init_mode_name, initial_tmv in self.automaton.initial_sets:
            mode_obj = self.automaton.get_mode(init_mode_name)
            prepared_tmv = self._prepare_initial_tmv(initial_tmv, 0.0)
            if not self._verify_containment(prepared_tmv, mode_obj.name, 0.0, max_jumps):
                self._mark_visited(prepared_tmv, mode_obj.name, 0.0, max_jumps)
                queue.append((mode_obj, prepared_tmv, 0.0, max_jumps))


        # 2. main processing loop (algorithm 12 lines 6-24)
        while queue:
            curr_mode, initial_tmv, t_start, j_rem = queue.popleft()
            
            # check new mode's start time doesnt exceed global horizon
            if t_start >= total_horizon:
                continue
            
            # a. compute continuous flow 
            # init ModeSolver for this mode
            solver = ModeSolver(
                mode=curr_mode, 
                state_vars=self.state_vars, 
                time_var=self.time_var,
                config=self.config
            )
            
            # flow within mode until time allowance ends
            flowpipes, status = solver.propagate_mode_evolution(
                initial_tmv, 
                time_end=total_horizon, 
                time_start=t_start
            )

            # if jump allowance available & transitions possible: analyze all flowpipes
            if j_rem > 0 and curr_mode.transitions and flowpipes:
                enabled, truncate_after_index = self.analyze_transitions(
                    curr_mode=curr_mode,
                    flowpipes=flowpipes,
                    j_rem=j_rem,
                    urgent_jumps_mode=urgent_jumps_mode,
                )
                # urgent: truncate continuous evolution after the first-hit segment
                if truncate_after_index is not None:
                    for k in range(truncate_after_index + 1, len(flowpipes)):
                        flowpipes[k]["is_valid"] = False

                # enqueue successors for all enabled transitions
                for (transition, intersections_to_aggregate, t_r) in enabled:
                    # NOTE: intersections_to_aggregate is already bucketed per transition.
                    agg_tmv = Intersection.aggregate_intersections(
                        intersection_list=intersections_to_aggregate,
                        state_vars=self.state_vars,
                        time_var_name=self.time_var,
                        t_r=t_r,
                        order=self.config.get("order", 4),
                        method=self.config.get("aggregation_method", "PCA"),
                        candidates=None,  # keep fast; see optional section below
                    )

                    next_set = Intersection.apply_reset_map(agg_tmv, transition.reset, self.state_vars)
                    next_set = self._prepare_initial_tmv(next_set, t_r)
                    next_mode_name = transition.target.name if hasattr(transition.target, "name") else transition.target

                    if not self._verify_containment(next_set, next_mode_name, t_r, j_rem - 1):
                        self._mark_visited(next_set, next_mode_name, t_r, j_rem - 1)
                        queue.append((transition.target, next_set, t_r, j_rem - 1))

            for segment in flowpipes:
                if segment.get('is_valid', True):
                    self.reachable_set.append({'mode': curr_mode.name, **segment})
        
        return self.reachable_set
    
    def _prepare_initial_tmv(self, tmv:TMVector, t_start: float) -> TMVector:
        state_dim = len(self.state_vars)
        new_tms = [copy.deepcopy(tmv.tms[i]) for i in range(min(state_dim, len(tmv.tms)))]
        out = TMVector(new_tms)
        out.domain = list(tmv.domain[:state_dim])

        # If this is a raw "identity" box (x_i over non-normalized domain),
        # Convert to affine normalized form so constant terms carry the true center.
        def _is_identity_box(vec: TMVector) -> bool:
            try:
                for i in range(state_dim):
                    poly_obj = vec.tms[i].poly.poly
                    gens = poly_obj.parent().gens()
                    if i >= len(gens):
                        return False
                    # exact match to generator (no constant or cross terms)
                    if not (poly_obj - gens[i]).is_zero():
                        return False
                return True
            except Exception:
                return False

        if _is_identity_box(out) and len(out.domain) == state_dim:
            var_names = tuple(str(v) for v in self.state_vars)
            ring = PolynomialRing(RIF, names=var_names)
            gens = ring.gens()
            dom = [Interval(-1.0, 1.0) for _ in range(state_dim)]
            ref = tuple([0.0] * state_dim)
            tms = []
            for i in range(state_dim):
                iv = out.domain[i]
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
                    max_order=out.tms[i].max_order
                )
                tm.sweep()
                tms.append(tm)
            out = TMVector(tms)

        # TODO: confirm whether this inflation is correct
        epsilon = 1e-9
        for tm in out.tms:
            tm.remainder = tm.remainder + Interval(-epsilon, epsilon)

        return out  

    def _verify_containment(self, new_set: TMVector, mode_name: str, t_start: float, j_rem: int) -> bool:
        """
        substep of algorithm 12 (line 16): check if a set has already been visited
        corrected logic: prunes exact/contained duplicates
        """
        key = (mode_name, j_rem)
        new_bbox = new_set.bound()
        
        t_tol = float(self.config.get("visited_time_tol", 1e-7))

        if key not in self._visited_workitems:
            return False
        # iterate through existing reachable segments for this mode
        for (t_prev, prev_bbox) in self._visited_workitems[key]:
            # must match the same discrete time slice
            if abs(float(t_prev) - float(t_start)) > t_tol:
                continue

            # enclosure check over all shared dims (state + time)
            min_dims = min(len(prev_bbox), len(new_bbox))
            enclosed = True
            for i in range(min_dims):
                if not prev_bbox[i].encloses(new_bbox[i]):
                    enclosed = False
                    break

            if enclosed:
                return True

        return False
    
    def _mark_visited(self, tmv: TMVector, mode_name: str, t_start: float, j_rem: int) -> None:
        key = (mode_name, j_rem)
        bbox = tmv.bound()
        self._visited_workitems.setdefault(key, []).append((float(t_start), bbox))

    def analyze_transitions(self, curr_mode: Mode, flowpipes: List[Dict[str, Any]], j_rem: int, urgent_jumps_mode: bool):
        """
        analyze all transitions against the produced flowpipes and return
        - enabled: list of (transition, intersections_list, t_r)
        - truncate_after_index: if urgent mode hit -> return index after which to invalidate segments, else None
        """

        if j_rem <= 0 or not curr_mode.transitions or not flowpipes:
            return [], None
        
        intersection_method = self.config.get('intersection_method', 'combined')
        aggregation_threshold = int(self.config.get("aggregation_threshold", 10))

        enabled = []
        truncate_after_index = None

        # URGENT MODE: consider ONLY 1st segment where ANY guard may intersect
        if urgent_jumps_mode:
            transitions = list(curr_mode.transitions)
            n_tr = len(transitions)
            urgent_validate = bool(self.config.get("urgent_validate", False))

            for i, segment in enumerate(flowpipes):
                if not segment.get("is_valid", True):
                    continue

                tmv_seg = segment["tmv"]
                seg_t = float(segment["time_interval_abs"].lower)

                if "bound" not in segment:
                    segment["bound"] = tmv_seg.bound()
                box = segment["bound"]

                if len(tmv_seg.domain) == len(self.state_vars) + 1:
                    time_iv = tmv_seg.domain[-1]
                else:
                    # treat as instantaneous
                    time_iv = Interval(0,0)

                # cheap pre-filter
                candidate_indices = []
                safe_indices = []
                for tr_idx, tr in enumerate(transitions):
                    q = Intersection.quick_guard_check_on_box(
                        tr.guard, box, self.state_vars, time_iv, self.time_var
                    )
                    if q == "VIOLATED":
                        continue
                    if q == "SAFE":
                        safe_indices.append(tr_idx)
                    else:
                        candidate_indices.append(tr_idx)

                if not safe_indices and not candidate_indices:
                    continue

                # expensive intersection ONLY for candidates
                per_transition = [[] for _ in range(n_tr)]
                for tr_idx in candidate_indices:
                    tr = transitions[tr_idx]
                    f_g = Intersection.intersect_flowpipe_guard(
                        tmv_seg, tr.guard, self.state_vars, method=intersection_method
                    )
                    if f_g is not None:
                        per_transition[tr_idx].append(f_g)

                # SAFE guards are enabled with the full segment
                for tr_idx in safe_indices:
                    per_transition[tr_idx].append(tmv_seg)

                if any(per_transition):
                    if urgent_validate:
                        # minimal sanity validation for urgent logic
                        assert len(per_transition) == n_tr, "urgent_validate: per_transition size mismatch"
                        assert all(
                            (idx < n_tr and idx >= 0) for idx in (safe_indices + candidate_indices)
                        ), "urgent_validate: guard indices out of range"
                    truncate_after_index = i
                    for tr_idx, fg_list in enumerate(per_transition):
                        if fg_list:
                            enabled.append((transitions[tr_idx], fg_list, seg_t))
                    break # urgent: stop scanning later segments

            return enabled, truncate_after_index
        # NON URGENT MODE: collect all segments
        transitions = list(curr_mode.transitions)
        n_tr = len(transitions)
        buckets = [[] for _ in range(n_tr)]
        t_r_map = [None for _ in range(n_tr)]  # earliest time low @ which there's an intersection
        for segment in flowpipes:
            if not segment.get("is_valid", True):
                continue

            tmv_seg = segment["tmv"]
            seg_t = float(segment["time_interval_abs"].lower)
            if "bound" not in segment:
                segment["bound"] = tmv_seg.bound()
            box = segment["bound"]

            if len(tmv_seg.domain) == len(self.state_vars) + 1:
                time_iv = tmv_seg.domain[-1]
            else:
                # treat as instantaneous
                time_iv = Interval(0,0)

            for tr_idx, tr in enumerate(transitions):
                # cheap pre-filter
                q = Intersection.quick_guard_check_on_box(tr.guard, box, self.state_vars, time_iv, self.time_var)

                if q == "VIOLATED":
                    continue

                if q == "SAFE":
                    f_g = tmv_seg  # no intersection needed
                else:
                    f_g = Intersection.intersect_flowpipe_guard(
                        tmv_seg, tr.guard, self.state_vars, method=intersection_method
                    )
                    if f_g is None:
                        continue

                buckets[tr_idx].append(f_g)
                if t_r_map[tr_idx] is None or seg_t < t_r_map[tr_idx]:
                    t_r_map[tr_idx] = seg_t

                # aggregate to prevent unbounded growth
                if len(buckets[tr_idx]) >= aggregation_threshold:
                    t_r = t_r_map[tr_idx]
                    assert t_r is not None
                    agg = Intersection.aggregate_intersections(
                        intersection_list=buckets[tr_idx],
                        state_vars=self.state_vars,
                        time_var_name=self.time_var,
                        t_r=t_r,
                        order=self.config.get("order", 4),
                        method=self.config.get("aggregation_method", "PCA"),
                        candidates=None,
                    )
                    buckets[tr_idx] = [agg]  # replace bucket with aggregated single TMV

        # finalize per-transition results
        for tr_idx, lst in enumerate(buckets):
            if not lst:
                continue
            t_r = t_r_map[tr_idx]
            assert t_r is not None
            enabled.append((transitions[tr_idx], lst, t_r))

        return enabled, None
