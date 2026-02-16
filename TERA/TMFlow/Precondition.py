"""Preconditioning helpers for Taylor model flowpipes."""

import copy
from typing import List, Tuple
from sage.all import RIF, PolynomialRing
import numpy as np

from TERA.TMCore.TMVector import TMVector
from TERA.TMCore.TaylorModel import TaylorModel
from TERA.TMCore.Interval import Interval
from TERA.TMCore.Polynomial import Polynomial

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
