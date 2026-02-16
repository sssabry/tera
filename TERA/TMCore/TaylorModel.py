"""Taylor model data structure and operations."""

from typing import List, Tuple
from sage.all import SR, PolynomialRing, RIF
from sage.rings.integer import Integer
import numbers
import math
from math import factorial

from TERA.TMCore.Polynomial import Polynomial, bound_monomial
from TERA.TMCore.Interval import Interval
EPS_C = 1e-20  # coefficient cutoff threshold


class TaylorModel:
    """Represent a Taylor model with polynomial and remainder."""

    def __init__(self, poly: Polynomial, rem: Interval,  domain: List[Interval], ref_point: Tuple, order_bounds: List[Interval] = [], max_order: int = None):
        """Initialize a Taylor model."""

        if not hasattr(poly, 'poly') or not hasattr(poly, 'ring'):
            raise TypeError("poly must be a Polynomial object (or act like one).")
        if ref_point is None:
            raise ValueError("TM must have a reference point for squaring (for consistency).")
        if domain is None:
            raise ValueError("TM must have a domain for rigorous bounds.")

        dim = poly.dim
        if len(domain) != dim:
            raise ValueError(f"Domain dimension ({len(domain)}) must match polynomial dimension ({dim}).")
        if len(ref_point) != dim:
            raise ValueError(f"Reference point dimension ({len(ref_point)}) must match polynomial dimension ({dim}).")

        # check if reference point is within domain
        for i, point_i in enumerate(ref_point):
            if not domain[i].contains(point_i):
                raise ValueError(f"Reference point {ref_point} is not inside the domain {domain}.")

        self.poly = poly
        self.remainder = rem
        self.order_bounds = order_bounds
        self.domain = domain
        self.max_order = max_order
        self.dimension = poly.dim
        self.ref_point = ref_point
        self._bound_cache = None
        self._bound_cache_key = None

    def evaluate(self, point: Tuple) -> Interval:
        """Evaluate the Taylor model at a point."""
        poly_val = self.poly.evaluate(point)
        # convert the  point evaluation to an interval and add the remainder
        return Interval(poly_val) + self.remainder

    def bound(self) -> Interval:
        """Return a cached bound of the Taylor model over its domain."""
        normalized_domain = tuple(self.domain)  # (Interval(-1, 1),) * self.dimension
        domain_sig = tuple((iv.lower, iv.upper) for iv in normalized_domain)
        rem_sig = (self.remainder.lower, self.remainder.upper)
        sage_poly_id = id(self.poly.poly)
        ring_id = id(self.poly.ring)
        cache_key = (sage_poly_id, ring_id, domain_sig, rem_sig)

        if self._bound_cache_key == cache_key and self._bound_cache is not None:
            return self._bound_cache

        poly_bound = self.poly.range_evaluate(normalized_domain)
        result = poly_bound + self.remainder
        self._bound_cache_key = cache_key
        self._bound_cache = result
        return result

    def truncate(self, new_order: int):

        extra_rem = Interval(0)
        normalized_domain = tuple(self.domain)  # (Interval(-1, 1),) * self.dimension

        new_sage_poly = self.poly.ring.zero()

        if self.dimension == 1:
            # univariate: keys are integers
            for exponent, coeff in self.poly.poly.dict().items():
                exp_tuple = (exponent,) if exponent != () else (0,)

                if sum(exp_tuple) <= new_order:
                    new_sage_poly += coeff * self.poly.ring.monomial(*exp_tuple)
                else:
                    extra_rem += bound_monomial(coeff, exp_tuple, normalized_domain)
        else:
            # multivariate: keys are tuples
            for exp_tuple, coeff in self.poly.poly.dict().items():
                if not exp_tuple:  # constant term
                    exp_tuple = (0,) * self.dimension

                if sum(exp_tuple) <= new_order:
                    new_sage_poly += coeff * self.poly.ring.monomial(*exp_tuple)
                else:
                    extra_rem += bound_monomial(coeff, exp_tuple, normalized_domain)

        new_remainder = self.remainder + extra_rem
        
        # update order bounds if they exist
        new_poly_obj = Polynomial(_poly=new_sage_poly, _ring=self.poly.ring)

        return TaylorModel(
            poly=new_poly_obj,
            rem=new_remainder,
            domain=self.domain,
            ref_point=self.ref_point,
            max_order=new_order
        )
    
    def sweep(self, EPS_c: float = EPS_C):
        """
        Iterate through polynomial coefficients:
        if |coef| < EPS_c: bound the monomial and add to remainder.
        CHANGE: after updating coefficients to be intervals,
        it now checks if magnitude of coefficient itnerval < EPS_C
        """

        rem_extra = Interval(0, 0)
        new_coeffs = {}
        normalized_domain = tuple(self.domain)  # (Interval(-1, 1),) * self.dimension
        try:
            cutoff = self.cutoff_threshold
        except Exception:
            cutoff = EPS_c

        if self.dimension == 1:
            for exponent, coeff in self.poly.poly.dict().items():
                exp_tuple = (exponent,) if exponent != () else (0,)
                if coeff.abs().upper() < cutoff and sum(exp_tuple) > 0:
                    term_bound = bound_monomial(coeff, exp_tuple, normalized_domain)
                    rem_extra += term_bound
                else:
                    new_coeffs[exponent] = coeff
        else:
            for exp_tuple, coeff in self.poly.poly.dict().items():
                if not exp_tuple:  # Handle constant term
                    new_coeffs[exp_tuple] = coeff
                    continue

                if coeff.abs().upper() < cutoff:
                    term_bound = bound_monomial(coeff, exp_tuple, normalized_domain)
                    rem_extra += term_bound
                else:
                    new_coeffs[exp_tuple] = coeff

        new_sage_poly = self.poly.ring(new_coeffs)
        self.poly = Polynomial(_poly=new_sage_poly)
        self.remainder += rem_extra

    def substitute(self, substitutions: dict) -> 'TaylorModel':
        """substitues variables int he polynomial, according to substitutions dict
        which maps a sage variable to a constant or expression"""

        new_poly = self.poly.substitute(substitutions)
        new_rem = Interval(self.remainder)

        return TaylorModel(
            poly=new_poly,
            rem=new_rem,
            domain=list(self.domain),
            ref_point=tuple(self.ref_point),
            max_order=self.max_order
        )

    # Util
    def __repr__(self):
        return f"TaylorModel(P={self.poly.poly}, R={self.remainder}, order={self.max_order}, dim={self.dimension})"

    def __str__(self):
        return f"P(x) = {self.poly.poly}\nRemainder = {self.remainder}"

    def copy(self) -> 'TaylorModel':
        """util for creating safe full copies of TMs"""
        new_poly = Polynomial(_poly=self.poly.poly)
        new_rem = Interval(self.remainder._interval)
        try:
            return TaylorModel(
                poly=new_poly,
                rem=new_rem,
                domain=list(self.domain),
                ref_point=tuple(self.ref_point),
                order_bounds=list(self.order_bounds),
                max_order=self.max_order
            )
        except ValueError:
            # preserve behavior of deepcopy (no ref_point validation)
            tm = TaylorModel.__new__(TaylorModel)
            tm.poly = new_poly
            tm.remainder = new_rem
            tm.order_bounds = list(self.order_bounds)
            tm.domain = list(self.domain)
            tm.max_order = self.max_order
            tm.dimension = self.dimension
            tm.ref_point = tuple(self.ref_point)
            tm._bound_cache = None
            tm._bound_cache_key = None
            return tm
    
    def get_constant_part(self) -> Interval:
        """extracts the constant term c from the polynomial P(x)"""
        return self.poly.get_constant_part()

    def _prepare_binary_op(self, other, tol: float = 1e-12):
        """helper function that verified compatibility for binary operation
            and returns standardized copies of self and other"""
        if not isinstance(other, TaylorModel):
            raise TypeError(f"Unsupported operand type for binary operation: {type(other)}")

        # check dimension
        if self.dimension != other.dimension:
            raise ValueError(f"Dimension mismatch: TM1 has {self.dimension}, TM2 has {other.dimension}.")

        # check reference points
        if len(self.ref_point) != len(other.ref_point):
            raise ValueError("Reference point dimensions do not match.")
        for p1, p2 in zip(self.ref_point, other.ref_point):
            if abs(p1 - p2) > tol:
                raise ValueError(f"Reference point mismatch: {self.ref_point} vs {other.ref_point}.")

        # check domains (within tolerance)
        if len(self.domain) != len(other.domain):
            raise ValueError("Domain dimensions do not match.")
        for d1, d2 in zip(self.domain, other.domain):
            if not (abs(d1.lower - d2.lower) < tol and abs(d1.upper - d2.upper) < tol):
                raise ValueError(f"Domain mismatch: {self.domain} vs {other.domain}.")

        # standardize order by truncating the higher-order TM
        op1 = self.copy()
        op2 = other.copy()

        if op1.max_order > op2.max_order:
            op1 = op1.truncate(op2.max_order)
        elif op2.max_order > op1.max_order:
            op2 = op2.truncate(op1.max_order)

        return op1, op2

    # Arithmetic operator overloading
    def __neg__(self):
        """negation of a taylor model -(p,I) = (-p, -I)"""
        new_poly = -self.poly
        new_rem = self.remainder * -1

        return TaylorModel(
            poly=new_poly,
            rem=new_rem,
            order_bounds=[],
            domain=list(self.domain),
            ref_point=tuple(self.ref_point),
            max_order=self.max_order
        )

    def __add__(self, other):

        if isinstance(other, (numbers.Number, int, float, Interval, Integer)):
            # scalar = create a constant TM from 'other'
            const_poly = Polynomial(_poly=self.poly.ring(Interval(other)._interval))
            const_tm = TaylorModel(
                poly=const_poly,
                rem=Interval(0),
                domain=list(self.domain),
                ref_point=tuple(self.ref_point),
                max_order=self.max_order
            )
            # recurse with (self + const_tm)
            return self.__add__(const_tm)

        # handle TM + TM
        if hasattr(other, 'poly') and hasattr(other, 'remainder'):
            op1, op2 = self._prepare_binary_op(other)

            sum_poly = op1.poly + op2.poly
            sum_rem = op1.remainder + op2.remainder

            return TaylorModel(
                poly=sum_poly,
                rem=sum_rem,
                domain=op1.domain,
                ref_point=op1.ref_point,
                max_order=op1.max_order
            )

        raise ValueError(f"Cannot add TaylorModel to unsupported type {type(other)}")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # handle TM - constant
        if isinstance(other, (numbers.Number, int, float, Interval, Integer)):
            const_poly = Polynomial(_poly=self.poly.ring(Interval(other)._interval))
            const_tm = TaylorModel(
                poly=const_poly,
                rem=Interval(0),
                domain=list(self.domain),
                ref_point=tuple(self.ref_point),
                max_order=self.max_order
            )
            return self.__sub__(const_tm)

        # handle TM - TM
        if hasattr(other, 'poly') and hasattr(other, 'remainder'):
            op1, op2 = self._prepare_binary_op(other)

            diff_poly = op1.poly - op2.poly
            diff_rem = op1.remainder - op2.remainder

            return TaylorModel(
                poly=diff_poly,
                rem=diff_rem,
                domain=op1.domain,
                ref_point=op1.ref_point,
                max_order=op1.max_order
            )

        raise ValueError(f"Cannot subtract unsupported type {type(other)} from TaylorModel")

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        # CASE 1: TaylorModel * TaylorModel
        if hasattr(other, 'poly') and hasattr(other, 'remainder'):
            op1, op2 = self._prepare_binary_op(other)
            max_order = op1.max_order

            prod_poly_sage = op1.poly.poly * op2.poly.poly

            kept_coeffs = {}
            trunc_rem = Interval(0)

            normalized_domain = tuple(self.domain)  # (Interval(-1, 1),) * op1.dimension
            is_univariate = op1.dimension == 1

            for exponents, coeff in prod_poly_sage.dict().items():
                if is_univariate:
                    exp_tuple = (exponents,) if exponents != () else (0,)
                else:
                    exp_tuple = exponents if exponents else (0,) * op1.dimension

                if sum(exp_tuple) <= max_order:
                    kept_coeffs[exponents] = coeff
                else:
                    trunc_rem += bound_monomial(coeff, exp_tuple, normalized_domain)

            new_poly_sage = op1.poly.ring(kept_coeffs)
            new_poly = Polynomial(_poly=new_poly_sage)

            bound_p1 = op1.poly.range_evaluate(tuple(self.domain))
            bound_p2 = op2.poly.range_evaluate(tuple(self.domain))

            rem_p1_r2 = bound_p1 * op2.remainder
            rem_p2_r1 = bound_p2 * op1.remainder
            rem_r1_r2 = op1.remainder * op2.remainder

            new_rem = trunc_rem + rem_p1_r2 + rem_p2_r1 + rem_r1_r2

            result = TaylorModel(
                poly=new_poly,
                rem=new_rem,
                domain=op1.domain,
                ref_point=op1.ref_point,
                max_order=max_order
            )
            result.sweep()
            return result

        # CASE 2: TaylorModel * Scalar
        elif isinstance(other, (numbers.Number, int, float, Interval, Integer)):
            if isinstance(other, Interval):
                scalar_interval = other
            else:
                scalar_interval = Interval(other)

            scaled_poly = self.poly * scalar_interval
            scaled_rem = self.remainder * scalar_interval

            return TaylorModel(
                poly=scaled_poly,
                rem=scaled_rem,
                domain=list(self.domain),
                ref_point=tuple(self.ref_point),
                max_order=self.max_order
            )

        raise ValueError(f"Cannot multiply TaylorModel by unsupported type {type(other)}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        if not isinstance(other, (int, Integer)) or other < 0:
            raise ValueError("Power must be a non-negative integer")

        if other == 0:
            # Return a TM for the constant function 1.0
            one_poly = Polynomial(expr="1.0", variables=self.poly.ring.variable_names())
            return TaylorModel(
                poly=one_poly,
                rem=Interval(0),
                domain=list(self.domain),
                ref_point=tuple(self.ref_point),
                max_order=self.max_order
            )

        result = self.copy()
        for _ in range(other - 1):
            result = result * self
        return result

    @classmethod
    def from_constant(cls, value, prototype: 'TaylorModel') -> 'TaylorModel':
        """
        factory method -> creates a constant TM matching the strucutral properties of a prototype TM
        
        inputs:
        - value: scalar value to convert to constant TM
        - prototype: existing TM to mimic structural attributes from
        """
        # hanlde input type/unwrap if necessary
        if isinstance(value, Interval):
            val_rif = value._interval
        else:
            val_rif = value

        # create constant polynomial in prototype's ring (has vars)
        sage_poly = prototype.poly.ring(val_rif)
        
        # wrap it in polynomial class
        const_poly_wrapper = Polynomial(_poly=sage_poly, _ring=prototype.poly.ring)

        # create and return TM
        return cls(
            poly=const_poly_wrapper,
            rem=Interval(0),  # constant = 0 remainder
            domain=list(prototype.domain),
            ref_point=tuple(prototype.ref_point),
            order_bounds=list(prototype.order_bounds) if prototype.order_bounds else [],
            max_order=prototype.max_order
        )

    def reciprocal(self) -> 'TaylorModel':
        """rigorous TM reciprocal matching the Makino-Berz intrinsic function framework
        from "TAYLOR MODELS AND OTHER VALIDATED FUNCTIONAL INCLUSION METHODS" (2003)
        
        compute 1/self as: 1/(c0 + u) = (1/c0) * sum_{i=0}^k (-u/c0)^i  +  R_k
        where c0 is a scalar center, u = self - c0
        - ensure 0 not in bound(self)
        - remainder: given x = u/c0 and Tail for 1/(1+x) after order k is r = (-x)^(k+1) / (1 + θ x)^(k+2),  θ in (0,1)
            so: |r| ≤ |x|^(k+1) / min_{θ∈[0,1]} |1 + θ x|^(k+2) and scale by |1/c0|
        """
        order = self.max_order

        # ensure safety constraint
        full_bound = self.bound()
        if full_bound.contains(0):
            raise ZeroDivisionError(f"Reciprocal undefined: 0 is contained in the TM bound {full_bound}.")

        # choose scalar center c0 as midpoint
        c0 = float(full_bound.midpoint())
        if c0 == 0.0:
            # fallback: pick a safe nonzero point. conservative choice: endpoint w/ larger magnitude
            lo = float(full_bound.lower)
            hi = float(full_bound.upper)
            c0 = lo if abs(lo) >= abs(hi) else hi
            if c0 == 0.0:
                raise ZeroDivisionError(f"Reciprocal failed: could not pick a nonzero scalar center from bound {full_bound}.")

        inv_c0 = 1.0 / c0

        # u = self - c0
        c_tm = TaylorModel.from_constant(c0, self)
        u_tm = self - c_tm

        # x = u / c0 
        x_tm = u_tm * inv_c0

        # polynomial: sum_{i=0}^n (-x)^i via Horner-like recurrence
        one_tm = TaylorModel.from_constant(1.0, self)
        poly = TaylorModel.from_constant(1.0, self)
        for _ in range(order, 0, -1):
            poly = -(poly * x_tm) + one_tm

        # multiply by 1/c0
        result = poly * inv_c0

        # compute remainder bound 
        x_bound = x_tm.bound()
        denom = Interval(1.0) + (x_bound * Interval(0.0, 1.0))

        if denom.contains(0):
            raise ZeroDivisionError(
                f"Reciprocal error bound failed: (1 + x) contains 0. "
                f"x_bound={x_bound}, denom_bound={denom}"
            )

        # bound |(-x)^(n+1)| by |x|^(n+1)
        abs_x = x_bound.abs()
        abs_x_pow = abs_x ** (order + 1)  
        
        denom_abs = denom.abs()
        min_abs_denom = denom_abs.lower

        min_abs_denom_iv = Interval(min_abs_denom)
        denom_pow = min_abs_denom_iv ** (order + 2)
        inv_denom_pow = Interval(1.0) / denom_pow

        # tail magnitude bound for 1/(1+x): 
        abs_inv_c0 = Interval(abs(inv_c0))
        tail_mag = abs_x_pow * inv_denom_pow * abs_inv_c0

        # add symmetric remainder
        result.remainder += tail_mag * Interval(-1.0, 1.0)
        return result


    def __truediv__(self, other):
        if isinstance(other, (int, float, numbers.Number, Interval)):
            # simply multiply by 1/scalar
            if isinstance(other, Interval):
                if other.contains(0):
                    raise ZeroDivisionError("Division by interval containing zero.")
                return self * (1.0 / other)
            
            if other == 0:
                raise ZeroDivisionError("Division by zero scalar.")
            return self * (1.0 / other)
            
        # TM/TM
        if isinstance(other, TaylorModel):
            # xin chen method
            # compute reciprocal of denominator then multiply
            try:
                recip_other = other.reciprocal()
            except ZeroDivisionError as e:
                ob = other.bound()
                oc = other.get_constant_part()
                raise ZeroDivisionError(f"{e} | denom_bound={ob} constant_part={oc}")
            return self * recip_other
            
        return NotImplemented
    
    def __rtruediv__(self, other):

        if isinstance(other, (int, float, numbers.Number, Interval)):
            recip_self = self.reciprocal()
            return recip_self * other
            
        return NotImplemented
    
    def compose(self, replacements: List['TaylorModel']) -> 'TaylorModel':
        """
        computes the composition of g(f(x)) where:
        - g (self) is the outer Taylor Model
        - f (replacements) is a list of inner Taylor Models (one for each variable of g).
        
        optimized implementation with fallback to manual version
        """
        if len(replacements) != self.dimension:
            raise ValueError(f"Composition mismatch: Outer TM dim {self.dimension} vs {len(replacements)} replacements.")

        # prepare symbolic substitution mpa
        subs_map = {var: r.poly.poly for var, r in zip(self.poly.vars, replacements)}

        # perform fast symbolic composition
        try:
            composed_poly_raw = self.poly.poly.subs(subs_map)
        except Exception:
            # fallback for complex rings or mismatches
            return self._compose_horner_fallback(replacements)
        
        # force it to stay in the right ring
        new_ring = replacements[0].poly.ring
        try:
            composed_poly_high = new_ring(composed_poly_raw)
        except Exception:
            return self._compose_horner_fallback(replacements)
        
        target_ring = replacements[0].poly.ring
        target_dim = len(replacements[0].poly.vars)
        new_coeffs = {}
        truncation_rem = Interval(0)
        
        target_domain = replacements[0].domain
        domain_intervals = tuple(target_domain)
        max_order = self.max_order
        
        raw_dict = composed_poly_raw.dict()
        
        # manually reconstruct the dictionary of coeffs
        for exponents, coeff in raw_dict.items():
            if isinstance(exponents, (int, numbers.Integral)):
                try:
                    if target_dim == 1:
                        term_key = (exponents,)
                    else:
                        if exponents == 0:
                            term_key = (0,) * target_dim
                        else:
                            return self._compose_horner_fallback(replacements)
                except:
                    return self._compose_horner_fallback(replacements)
            else:
                term_key = exponents

            # calculate order
            current_order = sum(term_key)
            
            if current_order <= max_order:
                # accumulate coefficients because multiple source terms might map to same dest term
                if term_key in new_coeffs:
                    new_coeffs[term_key] += coeff
                else:
                    new_coeffs[term_key] = coeff
            else:
                # bound high-order term
                term_bound = Interval(coeff)
                for i, exp in enumerate(term_key):
                    if exp > 0:
                        term_bound *= (domain_intervals[i] ** exp)
                truncation_rem += term_bound

        try:
            new_sage_poly = target_ring(new_coeffs)
        except Exception:
             return self._compose_horner_fallback(replacements)
             
        new_poly_wrapper = Polynomial(_poly=new_sage_poly, _ring=target_ring)
        
        # propagate remainder
        # r_new = R_outer + Trunc_Error + (Grad(P) @ Range(f)) * R_inner
        inner_bounds = [r.bound() for r in replacements]
        inner_rems = [r.remainder for r in replacements]
        propagated_rem = Interval(0)

        # enforce remainder must be an additive uncertainty set (0 in remainder)
        for i, Ri in enumerate(inner_rems):
            if not Ri.contains(0):
                raise ValueError(
                    f"compose(): replacement[{i}] remainder does not contain 0 "
                    f"(invalid TM decomposition for validated composition). remainder={Ri}"
                )
        
        for i, var in enumerate(self.poly.vars):
            derivative_poly = self.poly.poly.derivative(var)

            if derivative_poly.is_zero():
                continue

            # evaluate gradient over inner bounds (validated; with conservative fallbacks)
            try:
                deriv_wrapper = Polynomial(_poly=derivative_poly, _ring=derivative_poly.parent())
                grad_bound = deriv_wrapper.range_evaluate(tuple(inner_bounds))
            except Exception:
                try:
                    deriv_wrapper = Polynomial(_poly=derivative_poly, _ring=derivative_poly.parent())
                    grad_bound = deriv_wrapper._naive_range_evaluate(tuple(inner_bounds))
                except Exception:
                    var_intervals = {str(v): inner_bounds[j] for j, v in enumerate(self.poly.vars)}
                    grad_bound = Interval.bound_function(SR(derivative_poly), var_intervals)

            propagated_rem += grad_bound * inner_rems[i]

        total_rem = self.remainder + truncation_rem + propagated_rem
        
        result = TaylorModel(
            poly=new_poly_wrapper,
            rem=total_rem,
            domain=list(target_domain),
            ref_point=replacements[0].ref_point,
            max_order=max_order
        )
        
        result.sweep()
        return result
    
    def _compose_horner_fallback(self, replacements: List['TaylorModel']) -> 'TaylorModel':
        """
        includes optimizations to speed of the left-right architecture issue:
        - recursive horner to minimize TM multiplications and remainder accumulation
        - adds a final sweep to remove any tiny coefficients
        """
        # use first replacement as prototype for creating the constant TMs
        prototype = replacements[0]
        poly_dict = self.poly.poly.dict()
        
        def recursive_horner(coeffs: dict, var_idx: int) -> 'TaylorModel':
            """
            recursively evaluates polynomial using horner's rule on variable at var_idx
            P(x0, x1...) = sum( x0^k * Q_k(x1...) )
                         = (...(Q_n * x0 + Q_{n-1}) * x0 + ... + Q_0)
            """
            # base case: no more variables
            if var_idx == self.dimension:
                # coeffs should contain a single entry with key () -> constant
                c_val = sum(coeffs.values())
                
                # convert scalar/interval coefficient to constant TM
                if hasattr(c_val, 'midpoint'):
                    mid = float(c_val.midpoint())
                    rad = float(c_val.radius())
                    tm = TaylorModel.from_constant(mid, prototype)
                    tm.remainder = Interval(-rad, rad)
                else:
                    tm = TaylorModel.from_constant(float(c_val), prototype)
                return tm

            # recursive step: group terms by power of current var
            # buckets[power] = { (e_{idx+1}, ...): coeff }
            buckets = {}
            for exps, val in coeffs.items():
                p = exps[0] if exps else 0
                remaining_exps = exps[1:]
                
                if p not in buckets:
                    buckets[p] = {}
                buckets[p][remaining_exps] = val
            
            # identify powers present for this var
            powers = sorted(buckets.keys(), reverse=True)
            
            # none? return 0 as a TM
            if not powers:
                return TaylorModel.from_constant(0.0, prototype)
                
            # init horner's schme: start w coeff of the highest power
            max_p = powers[0]
            current_var_tm = replacements[var_idx]
            
            # evaluate its coeff poly
            result_tm = recursive_horner(buckets[max_p], var_idx + 1)
            
            # multiply by x and add next lower coefficient (horner loop)
            # result = (...((coeff_max * x + coeff_{max-1}) * x + ...) ... + ceff_0)
            for p in range(max_p - 1, -1, -1):
                # multiply by current variable
                result_tm = result_tm * current_var_tm
                
                # add polynomial coefficient for power p
                if p in buckets:
                    term_tm = recursive_horner(buckets[p], var_idx + 1)
                    result_tm = result_tm + term_tm
            
            return result_tm

        # recursive to execute
        result = recursive_horner(poly_dict, 0)
        
        # add outer remainder 
        result.remainder += self.remainder
        # do final sweep 
        result.sweep()
        
        return result

    def exp(self) -> 'TaylorModel':
        """applies intrinsic operation exp(TM) === e^TM 
        according to methodology described in Kyoko Makino's thesis"""
        order = self.max_order

        # decompose TM's P(X) into c + f(x)
        # extract constant part
        c_int = self.get_constant_part()

        # create TM from constant
        c_tm = TaylorModel.from_constant(c_int.midpoint(), self)
        f_bar = self - c_tm

        # expand the polynomial using the normalized horner scheme
        result = TaylorModel.from_constant(1.0, self)
        one_tm = TaylorModel.from_constant(1.0, self)

        # loop backwards from order down to 1
        for k in range(order, 0, -1):
            # divide by k as scalar
            scale = 1.0 / k
            result = result * scale 
            
            # multiply by variable part
            result = result * f_bar 
            
            # add 1
            result = result + one_tm
            
        # scale by exp(c)
        exp_c = c_int.exp()
        result = result * exp_c
        
        # compute remainder - exp(c) * (B^(n+1) / (n+1)!) * exp([0,1] * B)
        # b is bound of f_bar (above)
        bound_B = f_bar.bound()
        
        # (n+1)!
        fact_n_plus_1 = factorial(order + 1)
        
        # b^(n+1)
        pow_val = bound_B ** (order + 1)
        
        # exp([0,1] * B)
        rem_exp_part = (bound_B * Interval(0, 1)).exp()
        
        lagrange_rem = (pow_val / fact_n_plus_1) * rem_exp_part
        
        # scale remainder by exp(c) and add to result
        final_rem_term = exp_c * lagrange_rem
        result.remainder += final_rem_term
        
        return result
    
    def sin(self) -> 'TaylorModel':
        order = self.max_order
        
        # decomposition
        c_int = self.get_constant_part()
        c0 = float(c_int.midpoint())

        c0_tm = TaylorModel.from_constant(c0, self)
        u_tm = self - c0_tm

        # sin(c0), cos(c0) as point intervals
        sin_c0_iv = Interval(c0).sin()
        cos_c0_iv = Interval(c0).cos()
        
        # initialize TM with constant sin(c0) (centered properly)
        sin_c0 = float(sin_c0_iv.midpoint())
        result = TaylorModel.from_constant(sin_c0, self)
        # carry any tiny enclosure width (due to rigorous interval sin) into remainder
        result.remainder = Interval(sin_c0_iv.lower - sin_c0, sin_c0_iv.upper - sin_c0)

        # power accumulator u^i
        u_pow = TaylorModel.from_constant(1.0, self)
        
        # derivative cycle for sin at c0:
        # i=1:  cos(c0)
        # i=2: -sin(c0)
        # i=3: -cos(c0)
        # i=4:  sin(c0)
        for i in range(1, order + 1):
            u_pow = u_pow * u_tm

            cycle = i % 4
            if cycle == 1:
                coeff = cos_c0_iv
            elif cycle == 2:
                coeff = -sin_c0_iv
            elif cycle == 3:
                coeff = -cos_c0_iv
            else:  # cycle == 0
                coeff = sin_c0_iv

            term_scalar = coeff / math.factorial(i)   # Interval / int -> Interval
            result = result + (u_pow * term_scalar)

        # remainder: |u|^(k+1)/(k+1)! * [-1,1]
        u_rng = u_tm.bound().abs()
        rem_mag = (u_rng ** (order + 1)) / math.factorial(order + 1)
        result.remainder += rem_mag * Interval(-1.0, 1.0)
        return result
    
    def cos(self) -> 'TaylorModel':
        order = self.max_order
        c_int = self.get_constant_part()
        c0 = float(c_int.midpoint())

        c0_tm = TaylorModel.from_constant(c0, self)
        u_tm = self - c0_tm

        sin_c0_iv = Interval(c0).sin()
        cos_c0_iv = Interval(c0).cos()

        # Initialize with cos(c0), properly centered
        cos_c0 = float(cos_c0_iv.midpoint())
        result = TaylorModel.from_constant(cos_c0, self)
        result.remainder = Interval(cos_c0_iv.lower - cos_c0, cos_c0_iv.upper - cos_c0)

        u_pow = TaylorModel.from_constant(1.0, self)

        # Derivative cycle for cos at c0:
        # i=1: -sin(c0)
        # i=2: -cos(c0)
        # i=3:  sin(c0)
        # i=4:  cos(c0)
        for i in range(1, order + 1):
            u_pow = u_pow * u_tm

            cycle = i % 4
            if cycle == 1:
                coeff = -sin_c0_iv
            elif cycle == 2:
                coeff = -cos_c0_iv
            elif cycle == 3:
                coeff = sin_c0_iv
            else:  # cycle == 0
                coeff = cos_c0_iv

            term_scalar = coeff / math.factorial(i)
            result = result + (u_pow * term_scalar)

        u_rng = u_tm.bound().abs()
        rem_mag = (u_rng ** (order + 1)) / math.factorial(order + 1)

        result.remainder += rem_mag * Interval(-1.0, 1.0)

        return result
