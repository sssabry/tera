"""
Implements Flowpipe-Guard Intersections (Section 4.3) and Intersection Aggregation (Section 4.4).
Based on the Hybrid Systems Framework by Xin Chen.
"""

import numpy as np
from sage.all import RIF, SR, fast_callable, var, jacobian, vector
import copy
from collections import OrderedDict

from typing import List, Tuple, Optional, Any
from TERA.TMCore.TMVector import TMVector
from TERA.TMCore.TaylorModel import TaylorModel
from TERA.TMCore.Interval import Interval
from TERA.Hybrid.HybridModel import Condition, ResetMap, Transition
from TERA.TMCore.TMComputer import init_taylor_model

_FAST_CALLABLE_CACHE = {}
_EVAL_CACHE = OrderedDict()
_EVAL_CACHE_MAX = 20000

def apply_reset_map(aggregated_tmv: TMVector, reset_map: ResetMap, state_vars: List[Any]) -> TMVector:
    """
    implements the discrete jump reset map x' = r(x)
    according to chen's framework. if a var is missng from the mapping: treat as identity
    uses TM composition/symbolic eval to maintain rigor
    """
    state_dim = len(state_vars)
    reset_tms = []
    
    for i in range(state_dim):
        var_name = str(state_vars[i])
        expr = reset_map.mapping.get(var_name, var_name)
        reset_tms.append(aggregated_tmv.evaluate_symbolic(expr, state_vars))
        
    if len(aggregated_tmv.tms) > state_dim:
        reset_tms.append(aggregated_tmv.tms[-1])
        
    return TMVector(reset_tms)
    

# SECTION 4.3: FLOWPIPE/GUARD INTERSECTIONS 

def intersect_flowpipe_guard(tmv: TMVector, guard: Condition, state_vars: List[Any], 
                            time_var_name: str = 't', method: str = "combined", 
                            remainder_contraction: bool = False,
                            **kwargs) -> Optional[TMVector]:
    """
    top-level directing funciton for flowpipe-guard intersection
    currently implemented = domain contraction
    """
    # common parameters
    threshold = kwargs.get("threshold", 1e-4)
    epsilon = kwargs.get("epsilon", 1e-6)
    order = kwargs.get("order", 1)

    if method == "domain_contraction":
        return domain_contraction(tmv, guard, state_vars, time_var_name, threshold=threshold, epsilon=epsilon)
    
    elif method == "range_over_approximation":
        return range_over_approximation(tmv, guard, state_vars, time_var_name, 
                                        epsilon=epsilon, order=order)
        
    elif method == "combined":
        tight_tmv = domain_contraction(
            tmv, guard, state_vars, time_var_name, threshold=threshold, 
            epsilon=epsilon, remainder_contraction=remainder_contraction
        )
        
        if tight_tmv is None:
            return None #proven empty intersection early
            
        return range_over_approximation(tight_tmv, guard, state_vars, time_var_name, 
                                        epsilon=epsilon, order=order)
    else:
        raise ValueError(f"Unknown intersection method: {method}")

def _classify_condition_on_box(condition: Condition, box: List[Interval], state_vars: List[Any],
                               time_iv: Interval = None, time_var_name: str = "t") -> Tuple[str, List[Interval]]:
    """classify condition g_i(x) <= 0 (or < 0 if strict=True) over an interval box"""
    if condition is None or not getattr(condition, "constraints", None):
        return "FULL", []

    tiv = time_iv if time_iv is not None else Interval(0, 0)
    cond_strict = bool(getattr(condition, "strict", False))
    vals: List[Interval] = []
    all_full = True

    for expr in condition.constraints:
        v = _eval_constraint_on_box(expr, box, state_vars, tiv, time_var_name)
        vals.append(v)

        if cond_strict:
            if v.lower >= 0:
                return "EMPTY", vals
            if v.upper >= 0:
                all_full = False
        else:
            if v.lower > 0:
                return "EMPTY", vals
            if v.upper > 0:
                all_full = False

    return ("FULL" if all_full else "UNKNOWN"), vals
    
def domain_contraction(tmv: TMVector, condition: Condition, state_vars: List[Any], 
                        time_var_name: str, threshold: float = 1e-4, 
                        epsilon: float = 1e-6, remainder_contraction: bool = False) -> Optional[TMVector]:
    """
    implements algorithm 14 from chen's thesis - efficient iterative domain refinement
    goal: find a conservative interval box D_c = X_0' * [t_h, t_l] such that all solutions satisfying
    the invariant g(p(x0, t) + y) <= 0 are contained
    """
    # 1. init current domain and remainder
    curr_domain = [Interval(i.lower, i.upper) for i in tmv.domain]
    curr_remainder = [Interval(tm.remainder.lower, tm.remainder.upper) for tm in tmv.tms]
    prev_vol = _calc_box_volume(curr_domain, curr_remainder)
    local_cache = {}

    # 2. iterate to refine: until reductoin is negligble
    # prevent infinite loops
    max_iter = 20 
    for i in range(max_iter):
        
        # a. refine domain vars (x0 and t)
        for j in range(len(curr_domain)):
            is_domain = True
            lo = contract_variable_boundary(tmv, curr_domain, curr_remainder, condition, j, state_vars, time_var_name, epsilon, is_domain, True, cache=local_cache)
            up = contract_variable_boundary(tmv, curr_domain, curr_remainder, condition, j, state_vars, time_var_name, epsilon, is_domain, False, cache=local_cache)
            if lo > up: 
                return None
            curr_domain[j] = Interval(lo, up)
            
        # b. refine remainder variables (y) - not t dimension
        if remainder_contraction:
            for k in range(len(curr_remainder) - 1):
                is_domain = False
                lo = contract_variable_boundary(tmv, curr_domain, curr_remainder, condition, k, state_vars, time_var_name, epsilon, is_domain, True, cache=local_cache)
                up = contract_variable_boundary(tmv, curr_domain, curr_remainder, condition, k, state_vars, time_var_name, epsilon, is_domain, False, cache=local_cache)
                if lo > up: 
                    return None # proven empty intersection
                curr_remainder[k] = Interval(lo, up)
        
        # 3. check convergence
        curr_vol = _calc_box_volume(curr_domain, curr_remainder)
        if (prev_vol - curr_vol) < threshold * prev_vol:
            break
        prev_vol = curr_vol
    
    # 4. construct tightened TMV 
    new_tms = [tm.copy() for tm in tmv.tms]
    new_tmv = TMVector(new_tms)
    for i, tm in enumerate(new_tmv.tms):
        tm.domain = curr_domain
        if i < len(curr_remainder):
            tm.remainder = curr_remainder[i]
    
    return new_tmv


def contract_variable_boundary(tmv: TMVector, Dc: List[Interval], Rc: List[Interval], 
                             condition: Condition, var_idx: int, state_vars: List[Any],
                             time_var_name: str, epsilon: float, is_domain: bool,
                             is_lower: bool, cache: Optional[dict] = None) -> float:
    """generalized boundary search combingin algorithm 15 and 16 from chen's thesis"""
    # init alpha/beta from existing domain or remainder
    box = Dc if is_domain else Rc
    alpha, beta = box[var_idx].lower, box[var_idx].upper
    local_eps = max(epsilon, (beta - alpha) * 1e-4)
    local_cache = {} if cache is None else cache

    while (beta - alpha) >= local_eps:
        gamma = (alpha + beta) / 2.0
        test_Dc, test_Rc = list(Dc), list(Rc)
        
        # define half-interval based on search direction
        if is_lower:
            test_box = test_Dc if is_domain else test_Rc
            test_box[var_idx] = Interval(alpha, gamma)
        else:
            test_box = test_Dc if is_domain else test_Rc
            test_box[var_idx] = Interval(gamma, beta)

        if _possibly_contains_solution(tmv, test_Dc, test_Rc, condition, state_vars, time_var_name, cache=local_cache):
            if is_lower: beta = gamma   # Solution in lower half, move beta down
            else: alpha = gamma        # Solution in upper half, move alpha up
        else:
            if is_lower: alpha = gamma  # No solution in lower half, move alpha up
            else: beta = gamma         # No solution in upper half, move beta down
            
    return alpha

def contract_range_box(range_box: List[Interval], guard: Condition, 
                       state_vars: List[Any], time_interval: Interval, 
                       time_var_name: str, epsilon: float = 1e-6) -> Optional[List[Interval]]:
    """
    contracts a range-enclosure according to the guard constraints reusing the ICP
    boundary search logic but on the range space
    """
    state_dim = len(state_vars)
    curr_range = [Interval(i.lower, i.upper) for i in range_box[:state_dim]]
    
    range_search_domain = curr_range + [time_interval]
    var_names = [str(v) for v in state_vars] + [str(time_var_name)]
    # treat range vars as domain for search w/ dummy remainder
    dummy_remainder = [Interval(0, 0) for _ in range(state_dim + 1)]

    identity_tms = []
    for i in range(state_dim):
        tm_ident = init_taylor_model(SR(var_names[i]), var_names, range_search_domain, order=1)
        identity_tms.append(tm_ident)
    
    time_tm = init_taylor_model(SR(time_var_name), var_names, range_search_domain, order=1)
    identity_tms.append(time_tm)
    identity_tmv = TMVector(identity_tms)

    for j in range(state_dim):
        lo = contract_variable_boundary(identity_tmv, range_search_domain, 
                                       dummy_remainder, guard, j, state_vars, 
                                       time_var_name, epsilon, is_domain=True, is_lower=True)
        up = contract_variable_boundary(identity_tmv, range_search_domain, 
                                       dummy_remainder, guard, j, state_vars, 
                                       time_var_name, epsilon, is_domain=True, is_lower=False)
        
        if lo > up: return None 
        curr_range[j] = Interval(lo, up)
        range_search_domain[j] = Interval(lo, up)
        
    return curr_range

def range_over_approximation(tmv: TMVector, guard: Condition, state_vars: List[Any], 
                             time_var_name: str, **kwargs) -> Optional[TMVector]:
    """
    technique 2 for flowpipe-guard intersection: rnage over approximation
    """
    epsilon = kwargs.get("epsilon", 1e-6)
    order = kwargs.get("order", 1)

    # compute initial range box
    initial_range = tmv.bound()    
    time_interval = tmv.domain[-1]

    # clip range box against guard constraints
    contracted_range = contract_range_box(initial_range, guard, state_vars, 
                                         time_interval, time_var_name, epsilon)
    
    if contracted_range is None:
        return None

    center = np.array([I.midpoint() for I in contracted_range])
    radii = np.array([(I.upper - I.lower) / 2.0 for I in contracted_range])
    
    Mg = np.diag(radii)
    t_r = time_interval.midpoint()
    
    return construct_parallelotope_tm(center, Mg, state_vars, 
                                    time_var_name, t_r, order)

def _possibly_contains_solution(tmv: TMVector, domain: List[Interval], 
                               remainder: List[Interval], condition: Condition, 
                               state_vars: List[Any], time_var_name: str,
                               cache: Optional[dict] = None) -> bool:
    """
    helper function to check constraint satisfaction g(x) <= 0 over a sub-domain
    follows chen's inclusion-isotonic requirement strictly
    """
    if cache is not None:
        domain_sig = tuple((iv.lower, iv.upper) for iv in domain)
        rem_sig = tuple((iv.lower, iv.upper) for iv in remainder)
        cache_key = (domain_sig, rem_sig)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    # mutate tmv in-place to avoid deepcopy overhead, then restore
    tms = tmv.tms
    saved_domains = [tm.domain for tm in tms]
    saved_ref_points = [tm.ref_point for tm in tms]
    saved_remainders = [tm.remainder for tm in tms]
    new_ref = tuple(float(iv.midpoint()) for iv in domain)

    try:
        for i, tm in enumerate(tms):
            tm.domain = domain
            tm.ref_point = new_ref
            if i < len(remainder):
                tm.remainder = remainder[i]

        range_box = tmv.bound()
        time_interval = domain[-1]

        # iterate through inequalities g_i(x) <= 0 (or strict < 0)
        cond_strict = bool(getattr(condition, "strict", False))
        for expr in condition.constraints:
            # if lower bound of g(x) is strictly positive: rangebox cant satisfy <= 0
            val_interval = _eval_constraint_on_box(expr, range_box, state_vars, time_interval, time_var_name)

            if cond_strict:
                if val_interval.lower >= 0:
                    if cache is not None:
                        cache[cache_key] = False
                    return False
            elif val_interval.lower > 0:
                if cache is not None:
                    cache[cache_key] = False
                return False
        # overlap or fully enclosed
        if cache is not None:
            cache[cache_key] = True
        return True
    finally:
        for i, tm in enumerate(tms):
            tm.domain = saved_domains[i]
            tm.ref_point = saved_ref_points[i]
            tm.remainder = saved_remainders[i]

def _eval_constraint_on_box(expr, range_box, state_vars, time_interval, time_var_name):
    """
    Evaluates a symbolic constraint g(x) rigorously over an interval box.
    uses caching to avoid recompiling fast_callable repeatedly.
    """
    try:
        t_sym = var(time_var_name)
        all_vars = list(state_vars) + [t_sym]

        expr_key = str(expr)
        var_key = tuple(str(v) for v in all_vars)
        range_sig = tuple((b.lower, b.upper) for b in range_box)
        time_sig = (time_interval.lower, time_interval.upper)
        eval_key = (expr_key, var_key, range_sig, time_sig)

        cached = _EVAL_CACHE.get(eval_key)
        if cached is not None:
            return cached

        # cache key: expression + variable ordering
        key = (expr_key, var_key)
        f = _FAST_CALLABLE_CACHE.get(key)
        if f is None:
            f = fast_callable(expr, vars=all_vars, domain=RIF)
            _FAST_CALLABLE_CACHE[key] = f

        # range_box should contain (state + time) intervals; if not, append time_interval
        args = [RIF(b.lower, b.upper) for b in range_box]
        if len(args) < len(all_vars):
            args.append(RIF(time_interval.lower, time_interval.upper))
        elif len(args) > len(all_vars):
            args = args[:len(all_vars)]

        res_rif = f(*args)
        result = Interval(res_rif.lower(), res_rif.upper())
        _EVAL_CACHE[eval_key] = result
        if len(_EVAL_CACHE) > _EVAL_CACHE_MAX:
            _EVAL_CACHE.popitem(last=False)
        return result

    except Exception as e:
        try:
            val = RIF(expr)
            result = Interval(val.lower(), val.upper())
            _EVAL_CACHE[eval_key] = result
            if len(_EVAL_CACHE) > _EVAL_CACHE_MAX:
                _EVAL_CACHE.popitem(last=False)
            return result
        except:
            raise RuntimeError(f"Rigorous evaluation failed for {expr}: {e}")

def _calc_box_volume(domain: List[Interval], remainder: List[Interval]) -> float:
    """helper to calculate the combined volume for convergence checking"""
    vol = 1.0
    for i in domain + remainder:
        vol *= max(0, (i.upper - i.lower))
    return vol

# SECTION 4.4: INTERSECTION AGGREGATION

def aggregate_intersections(intersection_list: List[TMVector], state_vars: List[Any], 
                            order: int, time_var_name: str, t_r: float, method: str = "PCA",
                            candidates: Optional[List[np.ndarray]] = None,
                            sample_mode: str = "midpoint",
                            precomputed_bounds: Optional[List[List[Interval]]] = None) -> TMVector:
    """
    implements the general flowpipe aggregation framework from section 4.4. of chen's thesis

    goal: cluster multiple TM flowpipe segemnets into a single aggregate set to prevent explosion
    of flowpipes after jumps
    """
    if not intersection_list:
        raise ValueError("Cannot aggregate an empty list of intersections.")
    
    # 1. collect samples by evaluating TMs at domain midpoints & corners
    all_samples = []
    state_dim = len(state_vars)

    for idx, tmv in enumerate(intersection_list):
        # a. midpoint sample
        mid_domain = [I.midpoint() for I in tmv.domain]
        mid_pt = tmv.evaluate(mid_domain) # Returns List[Interval]
        all_samples.append([I.midpoint() for I in mid_pt[:state_dim]])

        # midpoint-only mode must add facet centers from a range enclosure
        if sample_mode == "midpoint":
            if precomputed_bounds is not None and idx < len(precomputed_bounds) and precomputed_bounds[idx] is not None:
                box = precomputed_bounds[idx]
            else:
                box = tmv.bound()  # List[Interval], conservative enclosure in state space
            c = [I.midpoint() for I in box[:state_dim]]
            for i in range(state_dim):
                p_lo = list(c); p_lo[i] = float(box[i].lower)
                p_hi = list(c); p_hi[i] = float(box[i].upper)
                all_samples.append(p_lo)
                all_samples.append(p_hi)
            continue

        if sample_mode in {"facet", "full"}:
            # b. boundary samples (simplified - min/max of each domain dim)
            for i in range(len(tmv.domain)):
                for val in [tmv.domain[i].lower, tmv.domain[i].upper]:
                    sample_domain = list(mid_domain)
                    sample_domain[i] = val
                    pt = tmv.evaluate(sample_domain)
                    all_samples.append([I.midpoint() for I in pt[:state_dim]])

    # convert to (n x m) matrix for linear algebra
    samples_matrix = np.array(all_samples).T 

    # 2. select orientation with user-preffered method
    if method.upper() == "PCA":
        vectors = compute_pca_orientation(samples_matrix)
        L = vectors.T
    else:
        selected_normals = select_critical_directions(samples_matrix, candidates)
        L = np.array(selected_normals)

    # 3. compute the support functions by projecting samples onto normals
    projections = L @ samples_matrix
    a = np.max(projections, axis=1)
    minus_b = np.min(projections, axis=1)

    # find center and radii (c/lamgbda)
    h_center = (a + minus_b) / 2.0
    h_radii = (a - minus_b) / 2.0

    # convert h-rep to g-rep (generators Mg)
    # l * x = h_center + diag(h_radii) * sigma
    # so: x = L^-1 * h_center + L^-1 * diag(h_radii) * sigma
    try:
        L_inv = np.linalg.inv(L)
    except np.linalg.LinAlgError:
        L_inv = np.linalg.pinv(L)
    
    center_point = L_inv @ h_center
    Mg = L_inv @ np.diag(h_radii)

    return construct_parallelotope_tm(center_point, Mg, state_vars, time_var_name, t_r, order)  

def compute_pca_orientation(samples: np.ndarray) -> np.ndarray:
    """Technique A: Aggregation by Oriented Rectangular Hull"""
    n, m = samples.shape

    # edge case where not enough data to determine orientation
    if m <= 1:
        return np.eye(n)
    
    # 1. calculate the mean s_bar
    mean_vec = np.mean(samples, axis=1, keepdims=True)
    
    # 2. subtract mean from every sample to center around origin
    ms_centered = samples - mean_vec

    # 3. get covariance matrix using:
    # m_cov = (1 / m-1) * ms_centered * ms_centered.T
    m_cov = (1.0 /(m - 1)) * np.dot(ms_centered, ms_centered.T)

    # 4. decompose m_cov to find the eigenvectors
    # the single value decomp columns of U are the eigenvectors
    u, s, vh = np.linalg.svd(m_cov)

    # u = orthogonal,representation of oriented hull's axes
    return u

def select_critical_directions(samples: np.ndarray, candidates: List[np.ndarray]) -> List[np.ndarray]:
    """
    Technique B: Aggregation by Parallelotope (Critical Directions)
    matches algorithm 17 in chen's thesis
    """
    n = samples.shape[0]
    selected_vectors = []

    # 0. normalize all candidate vectors
    working_pool = []
    for v in candidates:
        v_np = np.array(v, dtype=float).flatten()
        norm = np.linalg.norm(v_np)
        if norm > 1e-12:
            working_pool.append(v_np / norm)
    
    # if pool is empty/insufficient: seed it with axis aligned unit vectors
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        working_pool.append(e_i)

    # 1. loop to select
    for i in range(n):

        # fallback: should not happen
        if not working_pool:
            break

        candidate_scores = []
        for v in working_pool:
            # 2. compute the orthogonality score 
            # mu(v) = product of (1 - |cos(theta)|)
            mu = 1.0
            for l_prev in selected_vectors:
                mu *= (1.0 - abs(np.dot(l_prev, v)))
            candidate_scores.append(mu)
        
        # 3. keep candidates with best scores
        max_mu = max(candidate_scores)
        
        # if max_mu near 0: must pick unit vector that wasnt chosen
        if max_mu < 1e-6:
            # find the first unit vector not yet represented
            for j in range(n):
                e_j = np.zeros(n); e_j[j] = 1.0
                if not any(abs(np.dot(e_j, s)) > 0.99 for s in selected_vectors):
                    best_v = e_j
                    break
        else:
            # else? get vectors w/ top 1% of scores
            best_indices = [idx for idx, m in enumerate(candidate_scores) if m >= max_mu * 0.99]
            
            # 4. find the candidate that minimizes projected width
            best_v = None
            min_width = float('inf')
            for idx in best_indices:
                v_cand = working_pool[idx]
                projections = np.dot(v_cand, samples)
                width = np.max(projections) - np.min(projections)
                if width < min_width:
                    min_width = width
                    best_v = v_cand

        # 5. remove any vector from pool that's parallel to best_v
        # prevents picking in the same direction or its negation in future iterations
        selected_vectors.append(best_v)
        working_pool = [v for v in working_pool if abs(np.dot(v, best_v)) < 0.99]

    return selected_vectors


def construct_parallelotope_tm(center: np.ndarray, Mg: np.ndarray, state_vars: List[Any], 
                               time_var_name: str, t_r: float, order: int) -> TMVector:
    """Translates an H-represented Parallelotope into an Order-1 TM """
    state_dim = Mg.shape[0]
    
    unit_domain = [Interval(-1, 1) for _ in range(state_dim)] + [Interval(0,0)]
    sigma_names = [str(v) for v in state_vars] + [str(time_var_name)]

    s_syms = [var(name) for name in sigma_names]
    normalized_ref_point = tuple([0.0] * (state_dim +1))
    
    base_tms = []
    for i in range(state_dim):
        tm = init_taylor_model(s_syms[i], sigma_names, unit_domain, order, ref_point=normalized_ref_point)
        base_tms.append(tm)
    
    transformed_tms = []
    for i in range(state_dim):
        # start with constant center point for this dimension
        c_val = float(np.array(center).flatten()[i])
        row_tm = TaylorModel.from_constant(c_val, base_tms[0])

        # add the contribution of each generator variable s_j
        for j in range(state_dim):
            coeff = float(Mg[i, j])
            # scale by matrix coeff & add to row
            row_tm = row_tm + (base_tms[j] * coeff)
            
        transformed_tms.append(row_tm)

    time_tm = init_taylor_model(
        my_func=float(t_r), 
        var_names=sigma_names, 
        domains=unit_domain, 
        order=order, 
        ref_point=normalized_ref_point,
        expand_function=False 
    )
    transformed_tms.append(time_tm)
    
    return TMVector(transformed_tms)


def quick_guard_check(guard: Condition, box: List[Interval], state_vars: List[Any],
                      time_interval: Interval, time_var_name: str = 't') -> str:
    """cheap pre-filter for g(x) <= 0 over an interval enclosure
    returns:
        "SAFE" if all constraints are satisfied over the whole box
        "VIOLATED" if any constraint is strictly violated over the whole box
        "UNKNOWN" otherwise (boundary overlap)
    """
    if guard is None or not getattr(guard, "constraints", None):
        return "SAFE"

    guard_strict = bool(getattr(guard, "strict", False))
    all_inside = True
    for expr in guard.constraints:
        val = _eval_constraint_on_box(expr, box, state_vars, time_interval, time_var_name)

        # strictly violated everywhere
        if guard_strict:
            if val.lower >= 0:
                return "VIOLATED"
        elif val.lower > 0:
            return "VIOLATED"

        # strictly satisfied everywhere
        if guard_strict:
            if val.upper < 0:
                continue
        elif val.upper <= 0:
            continue

        # overlaps boundary: lo <= 0 < hi
        all_inside = False

    return "SAFE" if all_inside else "UNKNOWN"
