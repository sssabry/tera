"""Preconditioning helpers for Taylor model flowpipes."""

import copy
from typing import List, Tuple, Optional
from sage.all import RIF, PolynomialRing
import numpy as np

from TERA.TMCore.TMVector import TMVector
from TERA.TMCore.TaylorModel import TaylorModel
from TERA.TMCore.Interval import Interval
from TERA.TMCore.Polynomial import Polynomial

# Right-model invariant check (L/R)
def check_right_invariant(R: TMVector, state_dim: int, slack: float = 1e-12, time_var: Optional[str] = None, verbose: bool = False):
    """
    Check that the right model maps the standard normalized box B=[-1,1]^n into itself
    required by the L/R compositional integration!!
    """
    bounds = []
    slack_f = float(slack)
    accept = Interval(-1.0 - slack_f, 1.0 + slack_f)

    ring = R.tms[0].poly.ring
    dim = ring.ngens()
    n_comp = len(R.tms)
    var_names = ring.variable_names()
    gens = ring.gens()
    box = Interval(-1.0, 1.0)

    valid_shape = (n_comp == state_dim and dim == state_dim) or (n_comp == state_dim + 1 and dim == state_dim + 1)
    if not valid_shape:
        if verbose:
            print(
                f"L/R invariant check: invalid shape. "
                f"n_comp={n_comp}, state_dim={state_dim}, ring_dim={dim}, vars={var_names}"
            )
        return False, bounds, "SHAPE_MISMATCH", accept

    tgen = None
    time_index = None
    if time_var is not None and time_var in var_names:
        time_index = var_names.index(time_var)
        tgen = gens[time_index]

    def _depends_on_var(poly_obj, gen) -> bool:
        try:
            return poly_obj.degree(gen) > 0
        except Exception:
            pass
        try:
            return gen in poly_obj.variables()
        except Exception:
            pass
        try:
            sub0 = poly_obj.subs({gen: 0})
            sub1 = poly_obj.subs({gen: 1})
            return sub0 != sub1
        except Exception:
            return True

    # right model is expected to be time-free; enforce or fail fast.
    if tgen is not None:
        for i in range(state_dim):
            poly_obj = R.tms[i].poly.poly
            if _depends_on_var(poly_obj, tgen):
                if verbose:
                    print(
                        f"L/R invariant check: right model depends on time var '{time_var}' "
                        f"at component {i}. vars={var_names}"
                    )
                return False, bounds, "TIME_DEP", accept

    for i in range(state_dim):
        tm = R.tms[i]
        if len(tm.domain) != dim:
            if verbose:
                print(
                    f"L/R invariant check: domain dim mismatch at component {i}. "
                    f"domain={tm.domain}, expected_dim={dim}, vars={var_names}"
                )
            return False, bounds, "DOMAIN_DIM", accept
        domain_eval = list(tm.domain)
        for j in range(state_dim):
            domain_eval[j] = box
        if time_index is not None:
            domain_eval[time_index] = tm.domain[time_index]
        bound = tm.poly.range_evaluate(tuple(domain_eval)) + tm.remainder
        bounds.append(bound)

    if not bounds:
        return False, bounds, "EMPTY_BOUNDS", accept

    ok = all(accept.encloses(b) for b in bounds)
    if not ok:
        return False, bounds, "BOUND_VIOLATION", accept
    return True, bounds, None, accept

# Bünger corrected shrink wrapping (domain inflation p(qB))
def _remainder_radii(T_state: TMVector) -> List[RIF]:
    rads = []
    for i, tm in enumerate(T_state.tms):
        if not tm.remainder.encloses(Interval(0)):
            raise ValueError(f"Remainder does not contain 0 at component {i}: {tm.remainder}")
        lo = RIF(tm.remainder.lower)
        hi = RIF(tm.remainder.upper)
        rads.append(max(abs(lo), abs(hi)))
    return rads


def _bound_jacobian_poly_over_box(T_state: TMVector, domain_box: List[Interval]) -> List[List[Interval]]:
    n = len(T_state.tms)
    ring = T_state.tms[0].poly.ring
    dim = ring.ngens()
    gens = ring.gens()
    M = [[Interval(0) for _ in range(n)] for _ in range(n)]
    if len(domain_box) < dim:
        domain_box = list(domain_box) + [Interval(0.0, 0.0)] * (dim - len(domain_box))
    for i in range(n):
        p_i = T_state.tms[i].poly.poly
        for j in range(n):
            dpi = p_i.derivative(gens[j])
            if hasattr(dpi, "is_zero") and dpi.is_zero():
                M[i][j] = Interval(0)
                continue
            deriv_wrapper = Polynomial(_poly=dpi, _ring=dpi.parent())
            M[i][j] = deriv_wrapper.range_evaluate(tuple(domain_box))
    return M

def _choose_preconditioner_R(T_state: TMVector) -> Optional[np.ndarray]:
    try:
        A0 = T_state.get_jacobian()
        R = np.linalg.inv(A0)
        return R
    except Exception:
        return None

def _bound_abs_gprime_over_qB(T_state: TMVector, q: List[RIF], R: Optional[np.ndarray]) -> List[List[RIF]]:
    q_box = [Interval(-qi, qi) for qi in q]
    M_p_int = _bound_jacobian_poly_over_box(T_state, q_box)
    n = len(M_p_int)

    M_f_int = [[Interval(0) for _ in range(n)] for _ in range(n)]
    if R is None:
        M_f_int = M_p_int
    else:
        for i in range(n):
            for j in range(n):
                acc = Interval(0)
                for k in range(n):
                    acc += Interval(R[i, k]) * M_p_int[k][j]
                M_f_int[i][j] = acc

    M_g = [[RIF(0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                g_int = M_f_int[i][j] - Interval(1)
                M_g[i][j] = max(abs(RIF(g_int.lower)), abs(RIF(g_int.upper)))
            else:
                f_int = M_f_int[i][j]
                M_g[i][j] = max(abs(RIF(f_int.lower)), abs(RIF(f_int.upper)))
    return M_g

def shrink_wrap_corrected(T_state: TMVector, time_var: Optional[str] = None, slack_q: float = 1e-12,
    max_iter: int = 10, q_cap: float = 1.2, use_preconditioning: bool = True, verbose: bool = False, 
    strict_sanity: bool = False) -> dict:
    """Matching Florian Bunger's corrected shrink wrapping method (based on Berz/Makino classic theory)"""

    n = len(T_state.tms)
    ring = T_state.tms[0].poly.ring
    dim = ring.ngens()
    var_names = ring.variable_names()
    gens = ring.gens()

    valid_shape = (dim == n) or (dim == n + 1 and time_var is not None and time_var in var_names)
    if not valid_shape:
        return {'success': False, 'reason': 'DIM_MISMATCH'}

    if time_var is not None and time_var in var_names:
        tgen = gens[var_names.index(time_var)]
        for i in range(n):
            p_i = T_state.tms[i].poly.poly
            depends = False
            try:
                depends = p_i.degree(tgen) > 0
            except Exception:
                try:
                    depends = tgen in p_i.variables()
                except Exception:
                    try:
                        depends = p_i.subs({tgen: 0}) != p_i.subs({tgen: 1})
                    except Exception:
                        depends = True
            if depends:
                return {'success': False, 'reason': 'TIME_DEP'}

    r = _remainder_radii(T_state)
    q = [RIF(1) + ri for ri in r]
    R = _choose_preconditioner_R(T_state) if use_preconditioning else None
    eps = RIF(slack_q)

    for _ in range(max_iter):
        M_g = _bound_abs_gprime_over_qB(T_state, q, R)
        dq = [qi - RIF(1) for qi in q]
        s = []
        for i in range(n):
            acc = RIF(0)
            for j in range(n):
                acc += M_g[i][j] * dq[j]
            s.append(acc)
        q_new = [RIF(1) + r[i] + s[i] for i in range(n)]

        if verbose and n == 1:
            q_box = [Interval(-q[0], q[0])]
            M_p_int = _bound_jacobian_poly_over_box(T_state, q_box)
            f_int = M_p_int[0][0]
            if R is not None:
                f_int = Interval(R[0, 0]) * f_int
            g_int = f_int - Interval(1)
            g_abs = max(abs(RIF(g_int.lower)), abs(RIF(g_int.upper)))
            # optional debug info for 1D
            print(f"[shrink_wrap] r={r[0]}, q={q[0]}, p'={f_int}, g'={g_int}, |g'|={g_abs}")

        if any(float(qi) > q_cap for qi in q_new):
            return {'success': False, 'reason': 'Q_TOO_LARGE', 'q': q_new}

        if all(q_new[i] <= q[i] + eps for i in range(n)):
            q = q_new
            break
        q = q_new
    else:
        return {'success': False, 'reason': 'NO_CONVERGENCE', 'q': q}

    subs_map = {gens[j]: q[j] * gens[j] for j in range(n)}
    new_tms = []
    for i in range(n):
        tm = T_state.tms[i]
        p_i = tm.poly.poly
        p_i_sw = ring(p_i.subs(subs_map))
        poly_sw = Polynomial(_poly=p_i_sw, _ring=ring)
        new_tm = TaylorModel(
            poly=poly_sw,
            rem=Interval(0),
            domain=tm.domain,
            ref_point=tm.ref_point,
            max_order=tm.max_order
        )
        new_tms.append(new_tm)

    T_sw = TMVector(new_tms)

    try:
        orig_bounds = T_state.bound()
        sw_bounds = T_sw.bound()
        for i in range(n):
            if not sw_bounds[i].encloses(orig_bounds[i]):
                if strict_sanity:
                    return {'success': False, 'reason': 'SANITY_FAIL'}
                return {'success': True, 'T_sw': T_sw, 'q': q, 'reason': 'OK', 'warn': 'SANITY_FAIL'}
    except Exception:
        if strict_sanity:
            return {'success': False, 'reason': 'SANITY_FAIL'}
        return {'success': True, 'T_sw': T_sw, 'q': q, 'reason': 'OK', 'warn': 'SANITY_FAIL'}

    return {'success': True, 'T_sw': T_sw, 'q': q, 'reason': 'OK'}

# Shared helper methods
def evaluate_at_time(tmv_prev: TMVector, tau: float, time_var_name: str="t") -> TMVector:
    """Evaluate a TMVector at a local time value."""
    if tmv_prev is None or not getattr(tmv_prev, "tms", None):
        raise ValueError("evaluate_at_time: empty TMVector")

    # get the parent ring from TMV
    try:
        ring = tmv_prev.tms[0].poly.ring
        gens = ring.gens()
        gen_names = [str(g) for g in gens]
    except Exception as e:
        raise RuntimeError(f"evaluate_at_time: failed to access ring/gens: {e}")

    # identify time generator using time_var_name
    t_name = str(time_var_name)

    if t_name in gen_names:
        t_gen = gens[gen_names.index(t_name)]
    else:
        raise RuntimeError(f"evaluate_at_time: cannot resolve time generator for time var name {time_var_name}; gens={gen_names}")

    # substitute with exact interval field element
    tau_val = RIF(tau)
    out = tmv_prev.substitute({t_gen: tau_val})

    return out


def evaluate_at_t_end(tmv_prev: TMVector, t_end: float, time_var_name: str="t") -> TMVector:
    """Evaluate a TMVector at the end of a step."""
    return evaluate_at_time(tmv_prev, t_end, time_var_name)

def shift_to_origin(tmv_of_x0: TMVector):
    """Shift a TMVector so its center is at the origin."""
    dimension = len(tmv_of_x0.tms)
    c_0 = []

    # create a centered version (x_deviation)
    tmv_deviation = copy.deepcopy(tmv_of_x0)

    for i in range(dimension):
        # extract constant part
        constant_val = tmv_of_x0.tms[i].poly.poly.constant_coefficient()
        
        # add the midpoint of the remainder to the center
        rem_mid = tmv_of_x0.tms[i].remainder.midpoint()
        total_center = float(constant_val) + float(rem_mid)

        c_0.append(total_center)
    
        # subtract constant from each polynomial
        val_rif = RIF(c_0[i])
        tmv_deviation.tms[i].poly = tmv_deviation.tms[i].poly - val_rif

        # center the remainder in the deviation by remoing midpoint
        rem_rad = tmv_of_x0.tms[i].remainder.radius()
        tmv_deviation.tms[i].rem = Interval(-rem_rad, rem_rad)
    
    return tmv_deviation, c_0


def determine_magnitude(tmv_deviation: TMVector, domain: List[Interval]) -> List[Interval]:
    """Compute the deviation magnitude over the current domain."""
    # call range_evaluate(domain) on each TMVector
    tmv_deviation.domain = domain
    range_of_x0 = tmv_deviation.bound()
    return range_of_x0


def apply_linear_map_to_tmv(tmv: TMVector, M: np.ndarray, state_dim: int) -> TMVector:
    """Apply a linear map to state components of a TMVector."""
    out = copy.deepcopy(tmv)
    ring = out.tms[0].poly.ring

    new_tms = []
    for i in range(state_dim):
        acc_poly = ring(0)
        acc_rem = Interval(0, 0)
        for j in range(state_dim):
            a = float(M[i, j])
            if a == 0.0:
                continue
            acc_poly = acc_poly + RIF(a) * out.tms[j].poly.poly
            acc_rem = acc_rem + (Interval(a, a) * out.tms[j].remainder)

        tm_i = copy.deepcopy(out.tms[i])
        tm_i.poly = Polynomial(_poly=acc_poly, _ring=ring)
        tm_i.remainder = acc_rem
        new_tms.append(tm_i)

    trailing = [copy.deepcopy(tm) for tm in out.tms[state_dim:]]
    return TMVector(new_tms + trailing)

def apply_diagonal_inv_scales_to_tmv(tmv: TMVector, inv_scales: List[float], state_dim: int) -> TMVector:
    """Apply diagonal inverse scales to state components of a TMVector."""
    D = np.diag([float(inv_scales[i]) for i in range(state_dim)])
    return apply_linear_map_to_tmv(tmv, D, state_dim)


def compute_transformation(range_of_x0: List[Interval]):
    """Compute scale factors to normalize the domain to [-1, 1]."""
    scale_factors_S = []
    inv_scale_factors_S = []

    for interval in range_of_x0:
        # calculate sup magnitude: max(|inf|, |sup|)
        if hasattr(interval, 'radius'):
            rad = interval.radius()
        else:
            rad = (interval.upper - interval.lower) / 2.0
        
        # handle singular/point intervals to avoid div by 0
        if rad == 0:
            scale_factors_S.append(0.0)
            inv_scale_factors_S.append(1.0)
        else:
            scale_factors_S.append(rad)
            inv_scale_factors_S.append(1.0 / rad)
            
    return scale_factors_S, inv_scale_factors_S
    

def construct_new_initial_vars(state_dimension: int, scale_factors_S: List[float], center_c0: List[float], max_order: int, var_names: tuple, time_start: float, time_domain: Interval) -> TMVector:
    """Construct a normalized initial TMVector for the next step."""
    # 1. Get variables for the new normalized domain
    normalized_domain = [Interval(-1, 1)] * state_dimension + [time_domain]
    normalized_ref_point = tuple([0.0] * (state_dimension +1))
    ring = PolynomialRing(RIF, names=var_names)
    gens = ring.gens()

    new_tms = []
    
    for i in range(state_dimension):
        scale = scale_factors_S[i]
        
        # Create linear polynomial using string"scale * xi"
        # Add center constant: P_i = c0[i] + S[i] * x_i
        poly_sage = RIF(center_c0[i]) + RIF(scale) * gens[i]
        
        poly = Polynomial(_poly=poly_sage, _ring=ring)
        
        tm = TaylorModel(
            poly=poly, 
            rem=Interval(0), 
            domain=normalized_domain, 
            ref_point=normalized_ref_point, 
            order_bounds=[], 
            max_order=max_order
        )
        
        new_tms.append(tm)
    
    # create a separate proper time entry with proper start & end time
    time_poly_sage = gens[-1]
    
    time_poly = Polynomial(_poly=time_poly_sage, _ring=ring)
    time_tm = TaylorModel(
        poly=time_poly,
        rem=Interval(0),
        domain=normalized_domain,
        ref_point=normalized_ref_point,
        max_order=max_order
    )
    
    new_tms.append(time_tm)
    new_x0 = TMVector(new_tms)

    return new_x0


def compute_qr_matrix(A: np.ndarray) -> np.ndarray:
    """Compute a permuted-QR preconditioning matrix."""
    # compute the euclidean norm for every column in A
    A_norms = np.linalg.norm(A, axis=0)

    # sort norm vectors in descending order for 'sorting trick' (reduce overestimation)
    # get indices that would sort it
    desc_indices = np.argsort(A_norms)[::-1]

    # sort accordingly
    A_sorted = A[:, desc_indices]

    # perform standard QR decomposition on sorted A
    Q, R = np.linalg.qr(A_sorted)

    # normalize the signs to ensure R has non-negative diagonal
    diag_R = np.diag(R)
    d = np.sign(diag_R)
    d[d == 0] = 1.0

    # perform element-wise multiplication of the diagonal on Q
    Q = Q * d

    return Q

def rotate_tmv(tmv_deviation: TMVector, Q: np.ndarray) -> TMVector:
    """Rotate a centered TMVector into the Q coordinate system."""
    state_dim = Q.shape[0]
    tmv_rotated = copy.deepcopy(tmv_deviation)

    polys = [tm.poly for tm in tmv_deviation.tms]
    new_polys = []

    # compute the inverse of Q which its transpose
    Q_inv = Q.T

    # transform the polynomials by Q_Inv (ignore time)
    for i in range(state_dim):
        # p_new_i = sum(Q_inv[i][j] * p_j)
        # calculate row i (dot product of Q_inv[i,:] and polys[:])
        row_sum = polys[0] * 0.0 
        
        for j in range(state_dim):
            coeff = Q_inv[i, j]
            if coeff != 0:
                # multiply polynomial j by scalar matrix element
                term = polys[j] * coeff
                row_sum = row_sum + term
        
        new_polys.append(row_sum)

    # update the TMvector w the rotated polys (ignore time)
    for i in range(state_dim):
        tmv_rotated.tms[i].poly = new_polys[i]
        # ignore the remainders because we recalculate it next step

    old_rem_rads = np.array([tm.remainder.radius() for tm in tmv_deviation.tms[:state_dim]])
    abs_Q_inv = np.abs(Q_inv)
    new_rem_rads = abs_Q_inv @ old_rem_rads

    for i in range(state_dim):
        rad = float(new_rem_rads[i])
        tmv_rotated.tms[i].rem = Interval(-rad, rad)

    return tmv_rotated

def construct_qr_vars(dimension: int, state_dim: int, Q: np.ndarray, scale_factors_S: List[float], midpoints_m: List[float], 
                    center_c0: List[float], max_order: int, var_names: tuple, time_start: float, time_domain: Interval) -> TMVector:
    """Construct QR-preconditioned initial variables."""
    local_time_domain = Interval(0.0, float(time_domain.upper - time_domain.lower))
    normalized_domain = [Interval(-1, 1)] * state_dim + [local_time_domain]
    normalized_ref_point = tuple([0.0] * (state_dim + 1))
    
    ring = PolynomialRing(RIF, names=var_names)
    gens = ring.gens() # [x0, x1, ..., xn, t]
    
    # create the inner vector (the un-rotated, scaled box)
    # v_i = m[i] + S[i] * x_i
    V_polys = []
    for i in range(state_dim):
        # m + s * xi
        poly_sage = RIF(midpoints_m[i]) + RIF(scale_factors_S[i]) * gens[i]
        V_polys.append(poly_sage)
        
    # aply the preconditioner: W = Q * V
    W_polys = []
    for i in range(state_dim):
        # calculate row i: sum(Q[i,j] * V[j])
        row_poly = ring(0)
        for j in range(state_dim):
            q_val = RIF(Q[i, j])
            if q_val != 0:
                row_poly += q_val * V_polys[j]
        W_polys.append(row_poly)
        
    # add the centers
    final_tms = []
    for i in range(state_dim):
        # add c[i]
        final_poly_sage = W_polys[i] + RIF(center_c0[i])
        poly_wrapper = Polynomial(_poly=final_poly_sage, _ring=ring)
        
        tm = TaylorModel(
            poly=poly_wrapper,
            rem=Interval(0),
            domain=normalized_domain,
            ref_point=normalized_ref_point,
            order_bounds=[],
            max_order=max_order
        )
        final_tms.append(tm)

    time_poly_sage = gens[-1]
    time_poly = Polynomial(_poly=time_poly_sage, _ring=ring)
    
    time_tm = TaylorModel(
        poly=time_poly,
        rem=Interval(0),
        domain=normalized_domain,
        ref_point=normalized_ref_point,
        max_order=max_order
    )
    final_tms.append(time_tm)
        
    return TMVector(final_tms)

def construct_affine_left_model(dimension: int, Q: np.ndarray, scale_factors_S: List[float], midpoints_m: List[float],
                                center_c0: List[float], max_order:int, var_names: tuple, domain: List[Interval], time_start: float):
    """Construct the affine left model for QR preconditioning."""
    ref_point = tuple([0.0] * (dimension +1))

    corrected_domain = list(domain)
    corrected_domain[-1] = Interval(0.0, 0.0)

    ring = PolynomialRing(RIF, names=var_names)
    gens = ring.gens()

    # construct the inner vector V = m + S*x
    V_polys = []
    for i in range(dimension):
        # s_i * x_i + m_i
        poly_sage = RIF(scale_factors_S[i]) * gens[i] + RIF(midpoints_m[i])
        V_polys.append(poly_sage)

    # apply Q's matrix multiplication
    # w_i = sum(Q_ij * V_j)
    W_polys = []
    for i in range(dimension):
        row_poly = ring(0)
        for j in range(dimension):
            q_val = RIF(Q[i, j])
            if q_val != 0:
                row_poly += q_val * V_polys[j]
        W_polys.append(row_poly)

    # add center and create tmvector
    final_tms = []
    for i in range(dimension):
        # fiinal = W_i + c_i
        final_poly_sage = W_polys[i] + RIF(center_c0[i])
        
        poly_wrapper = Polynomial(_poly=final_poly_sage, _ring=ring)

        tm = TaylorModel(
            poly=poly_wrapper,
            rem=Interval(0), 
            domain=corrected_domain,
            ref_point=ref_point,
            max_order=max_order
        )
        final_tms.append(tm)
    
    time_poly_sage = gens[-1]
    time_poly = Polynomial(_poly=time_poly_sage, _ring=ring)

    time_tm = TaylorModel(
        poly=time_poly,
        rem=Interval(0),
        domain=corrected_domain,
        ref_point=ref_point,
        max_order=max_order
    )
    final_tms.append(time_tm)
    assert len(final_tms) == dimension + 1
        
    return TMVector(final_tms)

def normalize_right_model(tm_target:TMVector, midpoints_m:List[float], inv_scale_factors_S: List[float]) -> TMVector:
    """Normalize the right model into the [-1, 1] domain."""

    state_dim = len(midpoints_m)
    in_dim = len(tm_target.tms)

    has_time = (in_dim == state_dim + 1)
    if not has_time:
        raise AssertionError(
            f"normalize_right_model: expected tm_target dim {state_dim+1} (state+time) "
            f"but got {in_dim}"
        )
    
    new_tms = []
    for i in range(state_dim):
        tm = tm_target.tms[i]
        
        # 1. shift: (TM - m)
        m_val = RIF(midpoints_m[i])
        shifted_poly = tm.poly - m_val
        
        # 2. scale: inv_S * shifted
        inv_s = RIF(inv_scale_factors_S[i])
        scaled_poly = shifted_poly * inv_s
        
        # 3. scale remainder: new_rem = old_rem * inv_s
        scaled_rem = tm.remainder * float(inv_scale_factors_S[i])
        
        new_tm = TaylorModel(
            poly=scaled_poly,
            rem=scaled_rem,
            domain=tm.domain,
            ref_point=tm.ref_point,
            max_order=tm.max_order
        )

        new_tms.append(new_tm)

    time_tm = tm_target.tms[-1]
    new_tms.append(copy.deepcopy(time_tm))

    result = TMVector(new_tms)
    result.dimension = state_dim + 1

    # hard invariants
    assert len(result.tms) == state_dim + 1
    assert result.dimension == state_dim + 1
    assert result.tms[0].poly.dim == state_dim + 1
    assert len(result.tms[0].domain) == state_dim + 1
    assert len(result.tms[0].ref_point) == state_dim + 1

    return result

def decompose_flow(tmv_integrated:TMVector, t_step:float, time_var:str):
    """Decompose flow for left/right model factorization (unused)."""
    # 1. extract spatial map at t=h
    M_flow = evaluate_at_t_end(tmv_integrated, t_step, time_var)
    
    # 2. shift to origin and extract c
    M_centered, center_c = shift_to_origin(M_flow)

    # 3. compute QR matrix with sorting trick (G)
    A = M_centered.get_jacobian()
    Q = compute_qr_matrix(A)

    # 4. rotate (compute G_inv * M_centered) -> Q_poly
    Q_poly = rotate_tmv(M_centered, Q)
    
    return M_centered, center_c, Q, Q_poly

## preconditioning strategies

def qr_preconditioning(tmv_pre: TMVector, t_step: float, current_domain: List[Interval], 
                       time_var: str, time_start: float, var_names: tuple, state_dim: int):
    """Apply permuted-QR preconditioning for the next step."""
    tmv_at_t_end = evaluate_at_t_end(tmv_pre, t_step, time_var)

    # 0. jacobian: extract the linear coefficients from p to form A
    A_matrix = tmv_at_t_end.get_jacobian()
    if A_matrix.shape[0] > state_dim:
        A_matrix = A_matrix[:state_dim, :state_dim]

    # 1 - 4 compute QR with the sorting trick
    Q_matrix = compute_qr_matrix(A_matrix)

    # 5. shift the TM to origin
    tmv_deviation, center_c0 = shift_to_origin(tmv_at_t_end)

    # 6. transform centered polynomial to new coordinate system
    tmv_rotated = rotate_tmv(tmv_deviation, Q_matrix)

    # 7. determine bounding box of rotated set to nromalize domain to [-1,1]
    range_of_x0 = determine_magnitude(
        TMVector(tmv_rotated.tms[:state_dim]), 
        current_domain[:state_dim]
    )
    # 8. compute the transformation scales
    scales, inv_scales = compute_transformation(range_of_x0)
    midpoints_m = [i.midpoint() for i in range_of_x0]

    # 8. construct the new initial set for the next step
    dimension = len(tmv_pre.tms)
    max_order = tmv_pre.max_order
    time_interval = Interval(time_start, time_start + t_step)
    
    new_x0 = construct_qr_vars(
        dimension=dimension,
        state_dim=state_dim,
        Q=Q_matrix,
        scale_factors_S=scales,
        midpoints_m=midpoints_m,
        center_c0=center_c0,
        max_order=max_order,
        var_names=var_names,
        time_domain=time_interval,
        time_start=time_start
    )
    
    # standard normalized domain for the next integration step
    normalized_domain = [Interval(-1.0, 1.0) for _ in range(state_dim)] + [time_interval]
    
    return new_x0, scales, inv_scales, normalized_domain, Q_matrix
