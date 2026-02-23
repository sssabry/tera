"""Picard iteration helpers for Taylor model flowpipes."""

from typing import Callable

from TERA.TMCore.TMVector import TMVector
from TERA.TMCore.TaylorModel import TaylorModel
from TERA.TMCore.Interval import Interval

def compute_polynomial_flowpipe(x0: TMVector, ode_rhs: Callable[[TMVector], TMVector], 
                                order: int, cutoff_threshold: float,) -> TMVector:
    """Compute the polynomial flowpipe using Picard iteration."""
    x_current = x0.copy()

    # iteratively build higher order terms
    x0_tms = x0.tms
    x0_polys = [tm.poly for tm in x0_tms]
    x0_domains = [tm.domain for tm in x0_tms]
    x0_refs = [tm.ref_point for tm in x0_tms]

    ring = x0.tms[0].poly.ring
    gens = ring.gens()
    time_gen = gens[-1]
    time_idx = x0.tms[0].poly.dim - 1

    for k in range(1, order+1):
        # substitute f(P(t))
        velocity_vec = ode_rhs(x_current)
        if velocity_vec is None:
            return None   # signal verification failure

        new_tms = []
        for i in range(len(velocity_vec)):
            velocity_tm = velocity_vec[i]
            integrated_poly = velocity_tm.poly.integrate_truncated_by_var_index(
                int_var_index=time_idx,
                max_degree=k-1,
                start_expr=0.0
            )

            # add initial condition: x(t) = x0 + integral
            # x0[i] is the constant term (initial set)
            x0_poly = x0_polys[i]
            
            # construct the new TM (ignore remainder for now)
            new_poly = x0_poly + integrated_poly

            new_tm = TaylorModel(
                poly=new_poly,
                rem=Interval(0),
                domain=x0_domains[i],
                ref_point=x0_refs[i],
                max_order=order
            )
            
            # sweep small coefficients
            new_tm.sweep(cutoff_threshold)
            new_tms.append(new_tm)

        # add original t_tm back from x0
        if len(new_tms) < len(x0):
             for j in range(len(new_tms), len(x0)):
                 new_tms.append(x0[j])

        x_current = TMVector(new_tms)

    return x_current

def compute_verified_step(x_poly: TMVector, x0: TMVector, ode_rhs: Callable[[TMVector], TMVector], time_var: str, time_step: Interval,
                           time_start: float, order: int, cutoff_threshold: float,) -> TMVector:
    """Propagate remainder bounds using Picard iteration."""

    # evaluate vector field using IA P_out + R_out = f(P_in + R_in)
    velocity_vec = ode_rhs(x_poly)
    if velocity_vec is None:
        return None   # signal verification failure

    verified_tms = []
    time_var_idx = x0.tms[0].poly.dim - 1
    time_gen = x0.tms[0].poly.vars[time_var_idx]
    h = float(time_step.upper)  # assumes time_step = [0, h]
    x0_tms = x0.tms
    x0_polys = [tm.poly for tm in x0_tms]
    x0_rems = [tm.remainder for tm in x0_tms]
    x0_domains = [tm.domain for tm in x0_tms]
    x0_refs = [tm.ref_point for tm in x0_tms]
    
    for i in range(len(velocity_vec)):
        vel_tm = velocity_vec[i]
        
        # integrate polynomial part
        integrated_poly = vel_tm.poly.definite_integral(
            int_var=time_gen,
            dummy_var=None,
            start_expr=0.0,
            end_expr=time_gen
        )
        
        # integrate remainder part (symmetric radius bound)
        vel_rem = vel_tm.remainder
        vel_rad = float(vel_rem.abs().upper)
        integrated_rem = Interval(-vel_rad * h, vel_rad * h)
        
        # add initial condition x0 + Integral
        final_poly = x0_polys[i] + integrated_poly
        
        # combine remainders (symmetric convention)
        x0_rem = x0_rems[i]
        x0_rad = max(abs(float(x0_rem.lower)), abs(float(x0_rem.upper)))
        final_rem = Interval(-x0_rad, x0_rad) + integrated_rem
        
        # construct TM
        result_tm = TaylorModel(
            poly=final_poly,
            rem=final_rem,
            domain=x0_domains[i],
            ref_point=x0_refs[i],
            max_order=order
        )
        
        # sweep (automatically pushes to remainder)
        result_tm.sweep(cutoff_threshold)
        
        verified_tms.append(result_tm)
        
    # preserve time_tm 
    if len(verified_tms) < len(x_poly):
        for j in range(len(verified_tms), len(x_poly)):
            verified_tms.append(x_poly[j])
            
    return TMVector(verified_tms)
