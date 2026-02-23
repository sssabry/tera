"""Polynomial utilities used in Taylor models."""

from sage.all import sage_eval, PolynomialRing, RIF, SR
from sage.rings.integer import Integer
from sage.rings.rational import Rational
import numbers

from typing import Tuple, Dict, Union, List
from TERA.TMCore.Interval import Interval


class Polynomial:
    """Polynomial wrapper with interval-aware evaluation."""

    def __init__(self, expr: str = None, variables: Tuple[str] = None, _poly=None, _ring=None):
        """Initialize a polynomial from expression or Sage objects."""
        if _poly is not None:
            # construct from an existing Sage polynomial
            if _ring is not None:
                try:
                    self.poly = _ring(_poly)  # Coerce the expression/poly into the ring
                    self.ring = self.poly.parent()
                except Exception as e:
                    raise TypeError(f"Failed to coerce _poly ({_poly}) into _ring ({_ring}): {e}")
            elif hasattr(_poly, 'parent'):
                self.poly = _poly
                self.ring = _poly.parent()
            else:
                if variables is not None:
                    self.ring = PolynomialRing(RIF, names=variables)
                    try:
                        self.poly = self.ring(_poly)
                    except Exception as e:
                        raise TypeError(f"Failed to create polynomial from {_poly} with variables {variables}: {e}")
                else:
                    raise TypeError("Internal constructor with _poly requires a valid Sage polynomial or a _ring to coerce into.")
        elif expr is not None and variables is not None:
            # constructor from a string expression + variables
            if not variables:
                raise ValueError("Must provide at least one variable.")

            if _ring is None:
                # define the polynomial ring over RIF ring
                self.ring = PolynomialRing(RIF, names=variables)
            else:
                self.ring = _ring  # to allow for interval coefficients

            # safely evaluate the expression within the context of the ring's variables
            try:
                self.poly = sage_eval(expr, locals=self.ring.gens_dict())
            except Exception as e:
                raise ValueError(f"Failed to parse expression '{expr}': {e}")
        else:
            raise ValueError("Must provide a string expression and a tuple of variables.")
        
        # various caches to optimize computation speed for repeated operations
        # cache for derivative polynomials used in centered form evaluation
        self._deriv_cache = None
        # cache for monomial terms used in naive range evaluation
        self._monomials_cache = None
        self._monomials_poly_id = id(self.poly)
        # cache for full range evaluation by domain
        self._range_cache_key = None
        self._range_cache_value = None
        # small cache for range evaluation of differences
        self._diff_range_cache = {}
        self._diff_range_cache_order = []
        # cache for max exponents per variable
        self._max_exp_cache = {}
        # cache for truncate_by_var_index results
        self._truncate_var_cache = {}
        # cache of monomial exponents per variable
        self._var_exp_cache = {}
        # cache for constant detection
        self._const_cache_key = None
        self._const_cache_val = None

    # Properties
    @property
    def vars(self) -> Tuple:
        """Return the generator variables of the polynomial ring."""
        return self.ring.gens()

    @property
    def dim(self) -> int:
        """Return the number of variables."""
        return self.ring.ngens()

    @property
    def coeffs(self) -> Dict:
        """Return the sparse coefficient dictionary."""
        return self.poly.dict()

    # Evaluation
    def evaluate(self, point: Tuple) -> Interval:
        """Evaluate the polynomial at a point."""

        if len(point) != self.dim:
            raise ValueError(f"Point dimension {len(point)} doesn't match polynomial dimension {self.dim}")

        safe_point = []
        for p in point:
            if hasattr(p, 'lower') and hasattr(p, 'upper'):
                # convert my Interval to Sage RIF
                safe_point.append(p._interval)
            else:
                # standard float or int to RIF
                safe_point.append(RIF(p))

        return self.poly(*safe_point)

    def _centered_form_evaluate(self, domain: Tuple[Interval, ...]) -> Interval:
        """Bound the polynomial over a box using a centered form."""
        if len(domain) != self.dim:
            if self.dim == 0:
                return Interval(self.poly.constant_coefficient())
            raise ValueError(f"Domain dimension ({len(domain)}) does not match polynomial dimension ({self.dim})")

        if self.dim == 0:
            # Polynomial is just a constant
            return Interval(self.poly.constant_coefficient())

        # 1. Find the center of the domain (c)
        centers = [iv.midpoint() for iv in domain]

        # 2. Find the centered intervals (y_i - c_i)
        centered_intervals = [domain[i] - centers[i] for i in range(self.dim)]

        # 3. Evaluate the polynomial at the center point P(c)
        sage_gens = self.ring.gens()
        center_dict = {sage_gens[i]: RIF(centers[i]) for i in range(self.dim)}
        p_at_c = self.poly.subs(center_dict)  # This is a float

        total_bound = Interval(p_at_c)  # Start with P(c)

        # 4. Cache derivative polynomials (structure-only, domain varies)
        if self._deriv_cache is None or len(self._deriv_cache) != self.dim:
            deriv_cache = []
            for i in range(self.dim):
                var = sage_gens[i]
                sage_deriv = self.poly.derivative(var)
                deriv_cache.append(Polynomial(_poly=sage_deriv, _ring=sage_deriv.parent()))
            self._deriv_cache = deriv_cache

        # 5. Sum the partial derivative terms
        for i in range(self.dim):
            deriv_poly = self._deriv_cache[i]
            deriv_bound = deriv_poly._naive_range_evaluate(domain)
            total_bound += centered_intervals[i] * deriv_bound

        return total_bound

    def _naive_range_evaluate(self, domain: Tuple[Interval, ...]) -> Interval:
        """Bound the polynomial over a box using naive evaluation."""
        if len(domain) != self.dim:
            raise ValueError("Domain dimension does not match polynomial dimension.")

        total_bound = Interval(0)

        # go over each monomial (exponent: coefficient)
        if self._monomials_cache is None or id(self.poly) != self._monomials_poly_id:
            self._monomials_cache = list(self.poly.dict().items())
            self._monomials_poly_id = id(self.poly)

        if self.dim == 1:
            # univariate: keys are integers (exponents)
            for exponent, coeff in self._monomials_cache:
                if exponent == ():  # Handle constant term (exponent is empty tuple)
                    exp_tuple = (0,)
                else:
                    exp_tuple = (exponent,)  # Create 1-tuple for bound_monomial
                term_bound = bound_monomial(coeff, exp_tuple, domain)
                total_bound += term_bound
        else:
            # multivariate: keys are tuples of exponents
            for exp_tuple, coeff in self._monomials_cache:
                if not exp_tuple:  # Handle constant term (empty tuple)
                    exp_tuple = (0,) * self.dim
                term_bound = bound_monomial(coeff, exp_tuple, domain)
                total_bound += term_bound

        return total_bound

    def range_evaluate(self, domain: Tuple[Interval, ...]) -> Interval:
        """
        Rigorously bounds the polynomial's range over a box domain by
        computing *both* a naive bound and a centered-form bound,
        and returning their intersection.
        """
        domain_sig = tuple((iv.lower, iv.upper) for iv in domain)
        cache_key = (self._monomials_poly_id, domain_sig)
        if self._range_cache_key == cache_key and self._range_cache_value is not None:
            return self._range_cache_value

        # 1. Compute the naive, term-by-term bound (better for simple)
        bound_naive = self._naive_range_evaluate(domain)

        # 2. Compute the centered-form (MVT) bound (better for dependencies)
        bound_centered = self._centered_form_evaluate(domain)

        # 3. intersection is tighetest bound
        result = bound_naive.intersection(bound_centered)
        self._range_cache_key = cache_key
        self._range_cache_value = result
        return result

    def substitute_variable(self, var_name_to_sub: str, sub_value: float) -> 'Polynomial':
        """substitutes a specific variable with constant value using Sage,
        returns a new smaller polynomial that no longer depends on the substituted variable"""

        current_var_names = [str(v) for v in self.vars]

        var_to_sub_index = -1
        try:
            var_to_sub_index = current_var_names.index(var_name_to_sub)
        except ValueError:
            # Var isn't in this poly, return a copy
            from copy import deepcopy
            return deepcopy(self)

        sage_sub_value = RIF(sub_value)

        new_var_names = [v_str for v_str in current_var_names if v_str != var_name_to_sub]

        if not new_var_names:
            # result will a constant
            new_ring = PolynomialRing(RIF, names=[])

            if self.dim != 1:
                pass  

            substituted_sage_poly = self.poly.subs({self.vars[var_to_sub_index]: sage_sub_value})
            final_poly = new_ring(substituted_sage_poly)
            return Polynomial(_poly=final_poly, _ring=new_ring)

        # Result is a poly in (dim-1) variables
        new_ring = PolynomialRing(RIF, names=new_var_names)
        new_poly_coeffs = {}

        for exp_tuple, coeff in self.poly.dict().items():
            if not exp_tuple:
                exp_tuple = (0,) * self.dim

            k = exp_tuple[var_to_sub_index]

            if k == 0:
                # Monomial doesnt depend on var_to_sub
                new_exp_list = list(exp_tuple)
                new_exp_list.pop(var_to_sub_index)
                new_exp_tuple = tuple(new_exp_list)

                new_poly_coeffs[new_exp_tuple] = new_poly_coeffs.get(new_exp_tuple, 0) + coeff
            else:
                # monomial does depend on var to sub
                # P = c * (x_sub^k) * (other_vars)
                # Sub(P) = c * (val^k) * (other_vars)

                sub_coeff = coeff * (sage_sub_value ** k)

                new_exp_list = list(exp_tuple)
                new_exp_list.pop(var_to_sub_index)
                new_exp_tuple = tuple(new_exp_list)

                new_poly_coeffs[new_exp_tuple] = new_poly_coeffs.get(new_exp_tuple, 0) + sub_coeff

        final_poly = new_ring(new_poly_coeffs)

        return Polynomial(_poly=final_poly, _ring=new_ring)

    # Core arithmetic and operator overloading
    def __add__(self, other):
        if hasattr(other, 'poly'):  # isinstance(other, Polynomial):
            if self.ring != other.ring:
                try:
                    return Polynomial(_poly=self.poly + other.poly)
                except TypeError:
                    # Fallback: Coerce self to other's ring
                    P_other = other.ring
                    return Polynomial(_poly=P_other(self.poly) + other.poly)
            return Polynomial(_poly=self.poly + other.poly)

        elif isinstance(other, (int, float, Integer, Rational)):
            # Sage handles scalar addition automatically
            try:
                val = self.ring.base_ring()(other)
                return Polynomial(_poly=self.poly + val)
            except:
                # Fallback if coercion fails
                return Polynomial(_poly=self.poly + other)
        else:
            raise TypeError(f"Couldn't perform addition on Polynomial instance and {other} of type {type(other)}")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if hasattr(other, 'poly'): 
            if hasattr(other, 'ring') and self.ring != other.ring:
                raise ValueError("Cannot subtract polynomials from different rings.")
            return Polynomial(_poly=self.poly - other.poly)
        
        if isinstance(other, (int, float, Integer, Rational)):
            try:
                val = self.ring.base_ring()(other)
                return Polynomial(_poly=self.poly - val)
            except:
                # Fallback
                return Polynomial(_poly=self.poly - other)
        return Polynomial(_poly=self.poly - other)

    def __rsub__(self, other):
        return Polynomial(_poly=other - self.poly)

    def __mul__(self, other):
        if isinstance(other, Polynomial):
            if self.ring != other.ring:
                raise ValueError("Cannot multiply polynomials from different rings.")
            return Polynomial(_poly=self.poly * other.poly)

        if isinstance(other, Interval):
            # multiply the polynomial by the underlying sage RIF object
            return Polynomial(_poly=self.poly * other._interval)
        
        if isinstance(other, (int, float, Integer, Rational)):
            try:
                val = self.ring.base_ring()(other)
                return Polynomial(_poly=self.poly * val)
            except:
                pass
        return Polynomial(_poly=self.poly * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return Polynomial(_poly=-self.poly)

    def derivative(self, var) -> 'Polynomial':
        """compute partial derivative with respect to given variable "var" (string name or generator)"""
        if isinstance(var, str):
            if var not in self.ring.variable_names():
                raise ValueError(f"Variable '{var}' not in polynomial ring.")
            var_index = self.ring.variable_names().index(var)
            var = self.vars[var_index]

        return Polynomial(_poly=self.poly.derivative(var))

    def nth_derivative(self, var_index: int, n: int) -> 'Polynomial':
        """ computes the n^th partial derivative with respect to the variable at var_index"""
        if not (0 <= var_index < self.dim):
            raise ValueError(f"Variable index {var_index} out of bounds for dimension {self.dim}.")
        if n < 0:
            raise ValueError("Derivative order n must be non-negative.")

        var = self.vars[var_index]
        return Polynomial(_poly=self.poly.derivative(var, n), _ring=self.ring)

    def substitute(self, substitutions: Dict) -> 'Polynomial':
        """ substitutes variables w/ constants or other polynomials
        {t:s} substitution from the input dictionary"""

        safe_subs = {}
        base_ring = self.ring.base_ring()
        
        for k, v in substitutions.items():
            # Coerce standard python numbers/numpy types to RIF
            if isinstance(v, (int, float, numbers.Number, Integer, Rational)):
                try:
                    safe_subs[k] = base_ring(v)
                except Exception:
                    # Fallback if direct cast fails
                    safe_subs[k] = RIF(v)
            else:
                safe_subs[k] = v
                
        return Polynomial(_poly=self.poly.subs(safe_subs))

    def definite_integral(self, int_var, dummy_var, start_expr, end_expr) -> 'Polynomial':
        """computes the definite integral of a polynomial (used for _picard_iteration)
        - int_var -> sage variable to integrate with respect to
        - start_expr -> start of integration (always 0 for continuous, often not for hybrid)
        - end_expr -> end of integration (t)
        """
        if end_expr != int_var:
            raise NotImplementedError("Definite integral currently requires 'end_expr' to be the integration variable.")

        if isinstance(int_var, str):
            if int_var not in self.ring.variable_names():
                raise ValueError(f"Integration variable '{int_var}' not in polynomial ring variables: {self.ring.variable_names()}")
            int_var_index = self.ring.variable_names().index(int_var)
        else:
            try:
                int_var_index = self.vars.index(int_var)
            except ValueError:
                raise ValueError(f"Integration variable {int_var} not in polynomial ring: {self.vars}")

        ring = self.ring
        base_ring = ring.base_ring()
        IntervalOne = base_ring(1)

        # assume start_expr = number, convert to RIF
        t_start = RIF(start_expr)

        # start with zero polynomial in the same ring
        new_poly_sage = ring.zero()

        for exp_tuple, coeff in self.poly.dict().items():

            if len(exp_tuple) < len(self.vars):
                exp_tuple = exp_tuple + (0,) * (len(self.vars) - len(exp_tuple))

            k = exp_tuple[int_var_index]

            # 1. new_coeff = coeff / (k + 1)
            # use interval arithmetic for this
            new_coeff = coeff / (IntervalOne * (k + 1))

            # 2. variable part Q(t)
            new_exp_list = list(exp_tuple)
            new_exp_list[int_var_index] = k + 1
            new_poly_sage += new_coeff * ring.monomial(*tuple(new_exp_list))

            # 3. definite part -Q(t_start)
            # & set exponent of t to 0
            const_exp_list = list(exp_tuple)
            const_exp_list[int_var_index] = 0

            # 4. add to new polynomial
            shift_val = new_coeff * (t_start ** (k + 1))
            new_poly_sage -= shift_val * ring.monomial(*tuple(const_exp_list))

        return Polynomial(_poly=new_poly_sage, _ring=ring)

    def integrate_truncated_by_var_index(self, int_var_index: int, max_degree: int, start_expr: float = 0.0) -> 'Polynomial':
        """
        Integrate the polynomial w.r.t. variable at int_var_index, but only include
        terms whose exponent in that variable is <= max_degree (Picard-in-time).
        Equivalent to truncate_by_var_index(int_var_index, max_degree).definite_integral(...),
        but avoids constructing the intermediate truncated polynomial.
        """
        if not (0 <= int_var_index < self.dim):
            raise ValueError(f"Variable index {int_var_index} out of bounds for dimension {self.dim}.")
        if max_degree < 0:
            return Polynomial(_poly=self.ring.zero(), _ring=self.ring)

        ring = self.ring
        base_ring = ring.base_ring()
        IntervalOne = base_ring(1)
        t_start = RIF(start_expr)

        new_poly_sage = ring.zero()
        for exp_tuple, coeff in self.poly.dict().items():
            if len(exp_tuple) < len(self.vars):
                exp_tuple = exp_tuple + (0,) * (len(self.vars) - len(exp_tuple))
            k = exp_tuple[int_var_index]
            if k > max_degree:
                continue

            new_coeff = coeff / (IntervalOne * (k + 1))
            new_exp_list = list(exp_tuple)
            new_exp_list[int_var_index] = k + 1
            new_poly_sage += new_coeff * ring.monomial(*tuple(new_exp_list))

            const_exp_list = list(exp_tuple)
            const_exp_list[int_var_index] = 0
            shift_val = new_coeff * (t_start ** (k + 1))
            new_poly_sage -= shift_val * ring.monomial(*tuple(const_exp_list))

        return Polynomial(_poly=new_poly_sage, _ring=ring)

    def truncate(self, max_order: int) -> 'Polynomial':
        """ truncates the polynomial. DISCARDDS higher-order terms
        used in picard operation - not for rigorous truncation"""
        new_sage_poly = self.ring.zero()

        if self.dim == 1:
            for exponent, coeff in self.poly.dict().items():
                exp_tuple = (exponent,) if exponent != () else (0,)
                if sum(exp_tuple) <= max_order:
                    new_sage_poly += coeff * self.ring.monomial(*exp_tuple)

        else:
            for exp_tuple, coeff in self.poly.dict().items():
                if not exp_tuple:
                    exp_tuple = (0,) * self.dim
                if sum(exp_tuple) <= max_order:
                    new_sage_poly += coeff * self.ring.monomial(*exp_tuple)

        return Polynomial(_poly=new_sage_poly)

    def truncate_by_var_index(self, var_index: int, max_degree: int) -> 'Polynomial':
        """
        Return a new Polynomial containing only terms whose exponent in the selected variable
        (given by var_index) is <= max_degree. Other variable exponents are unrestricted.
        This is used for Picard-in-time truncation (truncate by time degree only)!!
        """
        if not (0 <= var_index < self.dim):
            raise ValueError(f"Variable index {var_index} out of bounds for dimension {self.dim}.")
        if max_degree < 0:
            return Polynomial(_poly=self.ring.zero(), _ring=self.ring)

        # cached truncated polynomial
        cache_key = (id(self.poly), var_index, max_degree)
        cached = self._truncate_var_cache.get(cache_key)
        if cached is not None:
            return cached

        d = self.poly.dict()
        if not d:
            # zero poly: returning self is fine (immutable semantics assumed)
            self._truncate_var_cache[cache_key] = self
            return self

        # univariate case (var_index must be 0)
        if self.dim == 1:
            if var_index != 0:
                raise ValueError("dim==1 implies var_index must be 0.")
            # keys are ints or () for constant
            # detect whether truncation actually happens, without extra structures
            max_exp = 0
            new_dict = {}
            has_trunc = False
            for k, coeff in d.items():
                e = 0 if k == () else k
                if e > max_exp:
                    max_exp = e
                if e <= max_degree:
                    new_dict[k] = coeff
                else:
                    has_trunc = True

            if not has_trunc:
                self._truncate_var_cache[cache_key] = self
                self._max_exp_cache[var_index] = max_exp
                return self

            res = Polynomial(_poly=self.ring(new_dict), _ring=self.ring)

        # multivariate case
        else:
            # keys are tuples length dim, or () for constant
            has_trunc = False
            max_exp = 0
            new_dict = {}
            for exp, coeff in d.items():
                if exp == ():
                    e = 0
                else:
                    # exp is tuple length dim
                    e = exp[var_index]
                if e > max_exp:
                    max_exp = e
                if e <= max_degree:
                    new_dict[exp] = coeff
                else:
                    has_trunc = True

            if not has_trunc:
                self._truncate_var_cache[cache_key] = self
                self._max_exp_cache[var_index] = max_exp
                return self

            res = Polynomial(_poly=self.ring(new_dict), _ring=self.ring)

        if len(self._truncate_var_cache) < 16:
            self._truncate_var_cache[cache_key] = res
        self._max_exp_cache[var_index] = max_exp
        return res

    # Utils
    def __str__(self):
        return str(self.poly)

    def __repr__(self):
        return f"Polynomial('{self.poly}', vars={self.ring.variable_names()})"

    def degree(self):
        return self.poly.total_degree()

    def to_symbolic(self):
        return self.poly
    
    def get_constant_part(self) -> Interval:
        """Return the constant term as an interval."""
        const_val = self.poly.constant_coefficient()
        
        return Interval(const_val)

    def constant_coeff_if_constant(self):
        """Return constant coefficient if polynomial is constant, else None."""
        key = id(self.poly)
        if self._const_cache_key == key:
            return self._const_cache_val
        d = self.poly.dict()
        if len(d) != 1:
            self._const_cache_key = key
            self._const_cache_val = None
            return None
        (k, coeff) = next(iter(d.items()))
        # determine exponent tuple safely
        try:
            exps = tuple(k)
        except Exception:
            try:
                exps = tuple(k.list())
            except Exception:
                try:
                    exps = (int(k),)
                except Exception:
                    exps = ()

        if self.dim == 1:
            if (len(exps) == 0) or (len(exps) == 1 and exps[0] == 0):
                self._const_cache_key = key
                self._const_cache_val = coeff
                return coeff
        else:
            if len(exps) == 0 or all(e == 0 for e in exps):
                self._const_cache_key = key
                self._const_cache_val = coeff
                return coeff
        self._const_cache_key = key
        self._const_cache_val = None
        return None


def bound_monomial(coeff, exponents, domain) -> Interval:
    """Bound a monomial over a box domain."""
    term_bound = Interval(coeff)
    for i, exp in enumerate(exponents):
        if exp > 0:
            term_bound *= domain[i] ** exp
    return term_bound
