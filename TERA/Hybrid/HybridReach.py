"""
Docstring for TERA.Hybrid.HybridReach

orchestration layer of the hybrid systems reachability engine-extension

implements the worklist algorithm (algorithm 12) from chen's thesis
manages the global time horizon & handles the discrete switching logic

"""
from collections import deque
from typing import List, Dict, Tuple, Any, Optional
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
        self.remainder_contraction = bool(config.get("remainder_contraction", False))

        # store flowpipes as List of (ModeName, FlowpipeSegment)
        self.reachable_set = []
        self._visited_workitems = {} # Dict[Tuple[str,int], List[Tuple[float, List[Interval]]]]
        self.horizon_tol = float(config.get("horizon_tol", 1e-9))

    def compute_reachability(self) -> List[Tuple[str, TMVector]]:
        """
        implements the algorithm 12 of chen's thesis
        """
        queue = deque()
        total_horizon = self.config.get('time_horizon', 10.0)
        max_jumps = self.config.get('max_jumps', 5)
        urgent_jumps_mode = bool(self.config.get("urgent_jumps_mode", False))
        progress_bar = bool(self.config.get('progress_bar', False))
        t_progress = 0.0

        if progress_bar:
            from tqdm.auto import tqdm
            pbar = tqdm(total=float(total_horizon), desc="Hybrid Reachability Analysis")

        # 1. init queue, where each task is (mode, initTMV, globalTime, remainingJumps)
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
            solver_cfg = dict(self.config)
            solver_cfg["mode_start_abs"] = float(t_start)

            solver = ModeSolver(
                mode=curr_mode,
                state_vars=self.state_vars,
                time_var=self.time_var,
                config=solver_cfg
            )
        
            # flow within mode until time allowance ends
            flowpipes, status = solver.propagate_mode_evolution(
                initial_tmv,
                time_end=total_horizon,
                time_start=t_start
            )

            # handle boundary events appropriately
            boundary_step = solver.boundary_step_info if getattr(solver, "boundary_event", False) else None
            transition_flowpipes = list(flowpipes) if flowpipes is not None else []
            if boundary_step is not None:
                boundary_for_analysis = dict(boundary_step)
                if "time_interval_abs" not in boundary_for_analysis and boundary_for_analysis.get("time_interval") is not None:
                    t_local = boundary_for_analysis["time_interval"]
                    boundary_for_analysis["time_interval_abs"] = Interval(
                        float(t_start) + float(t_local.lower), float(t_start) + float(t_local.upper)
                    )
                if "time_interval" not in boundary_for_analysis and boundary_for_analysis.get("time_interval_abs") is not None:
                    t_abs = boundary_for_analysis["time_interval_abs"]
                    boundary_for_analysis["time_interval"] = Interval(
                        float(t_abs.lower) - float(t_start), float(t_abs.upper) - float(t_start)
                    )
                transition_flowpipes.append(boundary_for_analysis)

            observed_t_end = float(t_start)
            if flowpipes:
                try:
                    observed_t_end = max(
                        float(seg["time_interval_abs"].upper)
                        for seg in flowpipes
                        if isinstance(seg, dict) and seg.get("time_interval_abs") is not None
                    )
                except Exception:
                    observed_t_end = float(t_start)

            if boundary_step is not None and boundary_step.get("time_interval_abs") is not None:
                boundary_t = float(boundary_step["time_interval_abs"].upper)
                if boundary_t > observed_t_end:
                    observed_t_end = boundary_t

            # update progress bar with new max time if activated
            if progress_bar:
                t_end = observed_t_end
                if t_end > t_progress:
                    t_diff = t_end - t_progress
                    t_progress = t_end
                    pbar.update(float(t_diff))

            enabled = []
            truncate_after_index = None
            accepted_successor_count = 0

            # if jump allowance available & transitions possible: analyze all flowpipes
            if j_rem > 0 and curr_mode.transitions and transition_flowpipes:

                if urgent_jumps_mode:
                    analysis_offset = 0

                    while analysis_offset < len(transition_flowpipes):
                        candidate_flowpipes = transition_flowpipes[analysis_offset:]

                        enabled, local_truncate_after_index = self.analyze_transitions(
                            curr_mode=curr_mode,
                            flowpipes=candidate_flowpipes,
                            j_rem=j_rem,
                            urgent_jumps_mode=True,
                        )

                        global_truncate_after_index = None
                        if local_truncate_after_index is not None:
                            global_truncate_after_index = analysis_offset + local_truncate_after_index

                        if not enabled:
                            break

                        accepted_this_round = 0

                        for (transition, intersections_to_aggregate, t_r) in enabled:
                            if len(intersections_to_aggregate) == 1:
                                agg_tmv = intersections_to_aggregate[0]
                            else:
                                agg_tmv = Intersection.aggregate_intersections(
                                    intersection_list=intersections_to_aggregate,
                                    state_vars=self.state_vars,
                                    time_var_name=self.time_var,
                                    t_r=t_r,
                                    order=self.config.get("order", 4),
                                    method=self.config.get("aggregation_method", "PCA"),
                                    candidates=None,
                                    sample_mode=self.config.get("aggregation_sample_mode", "midpoint"),
                                )

                            reset_input = self._boxify_state_tmv(agg_tmv)
                            next_set = Intersection.apply_reset_map(reset_input, transition.reset, self.state_vars)
                            next_set = self._boxify_state_tmv(next_set)

                            next_mode_name = transition.target.name if hasattr(transition.target, "name") else transition.target

                            tgt_mode_obj = transition.target if hasattr(transition.target, "invariant") else self.automaton.get_mode(next_mode_name)
                            tgt_inv = getattr(tgt_mode_obj, "invariant", None)
                            if tgt_inv is not None and getattr(tgt_inv, "constraints", None):
                                clipped_next = Intersection.intersect_flowpipe_guard(
                                    next_set,
                                    tgt_inv,
                                    self.state_vars,
                                    method=self.config.get("intersection_method", "combined"),
                                    remainder_contraction=self.remainder_contraction,
                                )

                                if clipped_next is None:
                                    continue

                                next_set = self._boxify_state_tmv(clipped_next)

                                tgt_cls, _tgt_vals = Intersection._classify_condition_on_box(
                                    tgt_inv,
                                    next_set.bound(),
                                    self.state_vars,
                                    time_iv=(next_set.domain[-1] if len(next_set.domain) > len(self.state_vars) else Interval(float(t_r), float(t_r))),
                                    time_var_name=self.time_var,
                                )

                                if tgt_cls != "FULL":
                                    continue

                            if not self._verify_containment(next_set, next_mode_name, t_r, j_rem - 1):
                                self._mark_visited(next_set, next_mode_name, t_r, j_rem - 1)
                                queue.append((transition.target, next_set, t_r, j_rem - 1))
                                accepted_this_round += 1
                                accepted_successor_count += 1

                        if accepted_this_round > 0:
                            truncate_after_index = global_truncate_after_index
                            break

                        if global_truncate_after_index is None:
                            break

                        analysis_offset = global_truncate_after_index + 1

                else:
                    enabled, truncate_after_index = self.analyze_transitions(
                        curr_mode=curr_mode,
                        flowpipes=transition_flowpipes,
                        j_rem=j_rem,
                        urgent_jumps_mode=False,
                    )

                    for (transition, intersections_to_aggregate, t_r) in enabled:
                        if len(intersections_to_aggregate) == 1:
                            agg_tmv = intersections_to_aggregate[0]
                        else:
                            agg_tmv = Intersection.aggregate_intersections(
                                intersection_list=intersections_to_aggregate,
                                state_vars=self.state_vars,
                                time_var_name=self.time_var,
                                t_r=t_r,
                                order=self.config.get("order", 4),
                                method=self.config.get("aggregation_method", "PCA"),
                                candidates=None,
                                sample_mode=self.config.get("aggregation_sample_mode", "midpoint"),
                            )

                        reset_input = self._boxify_state_tmv(agg_tmv)
                        next_set = Intersection.apply_reset_map(reset_input, transition.reset, self.state_vars)
                        next_set = self._boxify_state_tmv(next_set)

                        next_mode_name = transition.target.name if hasattr(transition.target, "name") else transition.target

                        tgt_mode_obj = transition.target if hasattr(transition.target, "invariant") else self.automaton.get_mode(next_mode_name)
                        tgt_inv = getattr(tgt_mode_obj, "invariant", None)
                        if tgt_inv is not None and getattr(tgt_inv, "constraints", None):
                            clipped_next = Intersection.intersect_flowpipe_guard(
                                next_set,
                                tgt_inv,
                                self.state_vars,
                                method=self.config.get("intersection_method", "combined"),
                                remainder_contraction=self.remainder_contraction,
                            )

                            if clipped_next is None:
                                continue

                            next_set = self._boxify_state_tmv(clipped_next)

                            tgt_cls, _tgt_vals = Intersection._classify_condition_on_box(
                                tgt_inv,
                                next_set.bound(),
                                self.state_vars,
                                time_iv=(next_set.domain[-1] if len(next_set.domain) > len(self.state_vars) else Interval(float(t_r), float(t_r))),
                                time_var_name=self.time_var,
                            )

                            if tgt_cls != "FULL":
                                continue

                        if not self._verify_containment(next_set, next_mode_name, t_r, j_rem - 1):
                            self._mark_visited(next_set, next_mode_name, t_r, j_rem - 1)
                            queue.append((transition.target, next_set, t_r, j_rem - 1))
                            accepted_successor_count += 1

            if urgent_jumps_mode and accepted_successor_count > 0 and truncate_after_index is not None:
                for k in range(truncate_after_index + 1, len(flowpipes)):
                    flowpipes[k]["is_valid"] = False

            for segment in (flowpipes or []):
                if segment.get('is_valid', True):
                    out_seg = {'mode': curr_mode.name, **segment}
                    self.reachable_set.append(out_seg)

        if progress_bar:
            pbar.close()

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

        return out  
    
    def _boxify_state_tmv(self, tmv: TMVector, max_order: Optional[int] = None) -> TMVector:
        """
        convert TMV into state-only box over state var ring! used for resets, so post-reset
        successors dont carry source flowpipe's local-time poly structure into next mode
        """
        state_dim = len(self.state_vars)
        bounds = tmv.bound()[:state_dim]

        var_names = tuple(str(v) for v in self.state_vars)
        ring = PolynomialRing(RIF, names=var_names)
        gens = ring.gens()

        dom = [Interval(-1.0, 1.0) for _ in range(state_dim)]
        ref = tuple([0.0] * state_dim)

        if max_order is None:
            try:
                max_order = tmv.tms[0].max_order
            except Exception:
                max_order = int(self.config.get("order", 4))

        tms = []
        for i in range(state_dim):
            iv = bounds[i]
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
                max_order=max_order,
            )
            tm.sweep()
            tms.append(tm)

        out = TMVector(tms)
        out.domain = list(dom)
        return out
    
    def _extract_transition_time(self,tmv: TMVector, fallback_t: float, seg_abs_lower: Optional[float] = None,
        seg_local_time_iv: Optional[Interval] = None,) -> float:
        """
        extract an absolute transition tiem from an intersected TMV (uses abs time NOT local time)
        """
        try:
            dom = getattr(tmv, "domain", None)
            if dom is not None and len(dom) > len(self.state_vars):
                t_local_iv = dom[-1]
                t_local_upper = float(t_local_iv.upper)

                if seg_abs_lower is not None:
                    seg_local_lower = 0.0
                    if seg_local_time_iv is not None:
                        seg_local_lower = float(seg_local_time_iv.lower)

                    return float(seg_abs_lower) + (t_local_upper - seg_local_lower)
        except Exception:
            pass

        return float(fallback_t)

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
                if not isinstance(segment, dict):
                    continue
                force_jump = bool(segment.get("force_jump", False))
                if not segment.get("is_valid", True) and (not force_jump):
                    continue
                if "tmv" not in segment or segment.get("tmv") is None:
                    continue

                tmv_seg = segment["tmv"]

                if "bound" not in segment:
                    segment["bound"] = tmv_seg.bound()
                box = segment["bound"]

                if len(tmv_seg.domain) == len(self.state_vars) + 1:
                    time_iv = tmv_seg.domain[-1]
                else:
                    time_iv = Interval(0, 0)

                if "time_interval_abs" in segment:
                    seg_abs_lo = float(segment["time_interval_abs"].lower)
                    seg_abs_hi = float(segment["time_interval_abs"].upper)
                    seg_t = seg_abs_hi
                else:
                    seg_abs_lo = None
                    seg_abs_hi = float(time_iv.upper)
                    seg_t = seg_abs_hi

                candidate_indices = []
                safe_indices = []
                for tr_idx, tr in enumerate(transitions):
                    q = Intersection.quick_guard_check(
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

                per_transition = [[] for _ in range(n_tr)]
                per_transition_t = [None for _ in range(n_tr)]

                for tr_idx in candidate_indices:
                    tr = transitions[tr_idx]
                    f_g = Intersection.intersect_flowpipe_guard(
                        tmv_seg, tr.guard, self.state_vars, method=intersection_method,
                        remainder_contraction=self.remainder_contraction,
                    )
                    if f_g is not None:
                        per_transition[tr_idx].append(f_g)
                        per_transition_t[tr_idx] = self._extract_transition_time(f_g,fallback_t=seg_t, seg_abs_lower=seg_abs_lo, seg_local_time_iv=time_iv)

                for tr_idx in safe_indices:
                    per_transition[tr_idx].append(tmv_seg)
                    if per_transition_t[tr_idx] is None:
                        # if whole segment is guard-safe, use earliest absolute time in segment
                        per_transition_t[tr_idx] = seg_abs_lo if seg_abs_lo is not None else seg_t

                if any(per_transition):
                    if urgent_validate:
                        assert len(per_transition) == n_tr, "urgent_validate: per_transition size mismatch"
                        assert all(
                            (idx < n_tr and idx >= 0) for idx in (safe_indices + candidate_indices)
                        ), "urgent_validate: guard indices out of range"

                    truncate_after_index = i
                    for tr_idx, fg_list in enumerate(per_transition):
                        if fg_list:
                            t_hit = per_transition_t[tr_idx] if per_transition_t[tr_idx] is not None else seg_t
                            enabled.append((transitions[tr_idx], fg_list, t_hit))
                    break

            return enabled, truncate_after_index

        # NON URGENT MODE: collect all segments
        transitions = list(curr_mode.transitions)
        n_tr = len(transitions)
        buckets = [[] for _ in range(n_tr)]
        buckets_bounds = [[] for _ in range(n_tr)]
        t_r_map = [None for _ in range(n_tr)]

        for seg_idx, segment in enumerate(flowpipes):
            if not isinstance(segment, dict):
                continue
            force_jump = bool(segment.get("force_jump", False))
            if (not segment.get("is_valid", True)) and (not force_jump):
                continue
            if "tmv" not in segment or segment.get("tmv") is None:
                continue

            tmv_seg = segment["tmv"]
            if "bound" not in segment:
                segment["bound"] = tmv_seg.bound()
            box = segment["bound"]

            if len(tmv_seg.domain) == len(self.state_vars) + 1:
                time_iv = tmv_seg.domain[-1]
            else:
                time_iv = Interval(0, 0)

            if "time_interval_abs" in segment:
                seg_abs_lo = float(segment["time_interval_abs"].lower)
                seg_abs_hi = float(segment["time_interval_abs"].upper)
                seg_t = seg_abs_hi
            else:
                seg_abs_lo = None
                seg_abs_hi = float(time_iv.upper)
                seg_t = seg_abs_hi

            for tr_idx, tr in enumerate(transitions):
                q = Intersection.quick_guard_check(tr.guard, box, self.state_vars, time_iv, self.time_var)

                if q == "VIOLATED":
                    continue

                if q == "SAFE":
                    f_g = tmv_seg
                    t_hit = seg_abs_lo if seg_abs_lo is not None else seg_t
                else:
                    f_g = Intersection.intersect_flowpipe_guard(
                        tmv_seg, tr.guard, self.state_vars, method=intersection_method,
                        remainder_contraction=self.remainder_contraction,
                    )
                    if f_g is None:
                        continue
                    t_hit = self._extract_transition_time(f_g, fallback_t=seg_t, seg_abs_lower=seg_abs_lo, seg_local_time_iv=time_iv)

                buckets[tr_idx].append(f_g)
                if f_g is tmv_seg:
                    buckets_bounds[tr_idx].append(box[:len(self.state_vars)])
                else:
                    buckets_bounds[tr_idx].append(f_g.bound()[:len(self.state_vars)])

                if t_r_map[tr_idx] is None or t_hit < t_r_map[tr_idx]:
                    t_r_map[tr_idx] = t_hit

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
                        sample_mode=self.config.get("aggregation_sample_mode", "midpoint"),
                        precomputed_bounds=buckets_bounds[tr_idx],
                    )
                    buckets[tr_idx] = [agg]
                    buckets_bounds[tr_idx] = [agg.bound()[:len(self.state_vars)]]

        for tr_idx, lst in enumerate(buckets):
            if not lst:
                continue
            t_r = t_r_map[tr_idx]
            assert t_r is not None
            enabled.append((transitions[tr_idx], lst, t_r))

        return enabled, None