"""
Collection of functions used to compute subsets of and the entire Taylor Model
"""
from typing import List, Tuple, Dict
from sage.all import SR, PolynomialRing, factorial, prod, RIF, N
from sage.rings.real_mpfr import RealLiteral, RealNumber
from sage.combinat.integer_vector import IntegerVectors
from sage.symbolic.expression import Expression
from sage.rings.rational import Rational
from sage.rings.integer import Integer

import numbers
from TERA.TMCore.TaylorModel import TaylorModel
from TERA.TMCore.Polynomial import Polynomial, bound_monomial
from TERA.TMCore.Interval import Interval, RIF_TYPE


# Helpers
def _make_expression_rigorous(expr):
    """recursively rebuilds a sage symbolic expression to make sure all numeric values are RIF intervals"""
    # case 1: already an RIF interval
    if isinstance(expr, RIF_TYPE):
        return expr

    # case 2: basic numeric types
    if isinstance(expr, (int, float, str)):
        try:
            return RIF(expr)
        except (TypeError, ValueError):
            return expr

    # case 3: non-sage experssion
    if not hasattr(expr, 'is_numeric'):
        try:
            return RIF(expr)
        except (TypeError, ValueError):
            return expr

    # case 4: special symbolic constants
    if str(expr) in ['pi', 'e']:
        return RIF(N(expr))

    # case 5: numeric symbolic expression
    if expr.is_numeric():
        try:
            return RIF(N(expr))
        except (TypeError, ValueError):
            return expr

    # case 6: variable or other leaf node
    op = expr.operator()
    if op is None:
        return expr

    # case 7: compound expression: break down & recurse
    rigorous_operands = [_make_expression_rigorous(o) for o in expr.operands()]

    # case 8: division
    if str(op) == '/' and len(rigorous_operands) == 2:
        num, den = rigorous_operands
        try:
            den_val = RIF(N(den)) if hasattr(den, 'is_numeric') else den
            return num / den_val
        except (TypeError, ValueError):
            pass

    # try applying the operator with rigorous operands
    try:
        result = op(*rigorous_operands)
        # if result is numeric, convert to RIF
        if hasattr(result, 'is_numeric') and result.is_numeric():
            return RIF(N(result))
        return result
    except Exception:
        return expr


def validate_and_prepare_inputs(var_names: Tuple[str, ...], domains: List[Interval],
                                order: int, ref_point: Tuple = None) -> Tuple[int, Tuple]:
    """
    validate inputs for compute_taylor_model and determine the reference point
    returns (dimension, ref_point_vals)

    """
    dimension = len(var_names)

    if len(domains) != dimension:
        raise ValueError("Number of domains must match number of variable names provided")
    if not isinstance(order, numbers.Integral) or order < 0:
        raise ValueError("Order must be a non-negative integer.")

    if ref_point is None:
        ref_point_vals = tuple(d.midpoint() for d in domains)
    else:
        if len(ref_point) != dimension:
            raise ValueError("Reference point dimension must match variable dimension.")
        ref_point_vals = ref_point

    return dimension, ref_point_vals


def compute_polynomial_terms(func: 'sage.symbolic.expression.Expression', var_names: Tuple[str, ...],
                             domains: List[Interval], ref_point_vals: Tuple, order: int, dimension: int) -> Tuple[PolynomialRing, Tuple, Tuple[str, ...]]:
    """ compute the normalized taylor polynomial, manually instead of using sagemath .taylor() for more control
        .taylor() uses Maxima to compute it, fails if func has RIF constants and produces centered coordinates
        like COSY -> goal is to store the polynomial on a normalized domain

        reimplemented to do things manually to work with RIF coefficients & constants, and handle TypeErrors
        CORA (2018 paper, sec 2.2) linear transformation of interval into a TM = 0.5(upper + lower) + 0.5(upper - lower)x'
        - 0.5(a + b) -> midpoint (ref_point)
        - 0.5(b - a) -> radius
        - x' -> normalized variable on domain [-1,1] e.g. y0, y1
    """
    # setup variables (og & normalized)
    vars_list = SR.var(' '.join(var_names))
    if not isinstance(vars_list, tuple):
        vars_list = (vars_list,)

    y_names = var_names # tuple(f'y{i}' for i in range(dimension))

    # setup polynomial ring with RIF (interval) coefficients
    tm_ring = PolynomialRing(RIF, names=y_names)

    # create a substitution dictionary for ref_point
    ref_point_dict = {var: RIF(val) for var, val in zip(vars_list, ref_point_vals)}

    # compute the radius (upper-lower) for each domain
    domain_rads = tuple(d.radius() for d in domains)

    # begin building the polynomial
    total_poly = tm_ring(0)

    # iterate over all multi-indicies from 0 to order (possible combinations)
    for k in range(order + 1):
        multi_indices_generator = IntegerVectors(k, dimension, min_part=0)

        for j_vec in multi_indices_generator:
            j_tuple = tuple(j_vec)

            # compute the term's partial derivative
            deriv_args = []
            for i in range(dimension):
                if j_tuple[i] > 0:
                    deriv_args.append(vars_list[i])
                    deriv_args.append(j_tuple[i])

            if not deriv_args:
                deriv_expr = func  # handle 0-th derivative
            else:
                deriv_expr = func.derivative(*deriv_args)

            # substitute rigorous ref_point into rigorous derivative term
            rigorous_deriv_expr = _make_expression_rigorous(deriv_expr)
            coeff_interval = rigorous_deriv_expr.subs(ref_point_dict)

            # evaluate deriv at the ref_point
            deriv_at_c_numeric = N(coeff_interval)
            deriv_at_c_real = deriv_at_c_numeric.real()

            # compute the factorial term: j! = j_1! * ...
            j_factorial = prod(factorial(j) for j in j_tuple)

            # compute the radius term: r^j = r_1^j_1 * ...
            radii_term = prod(domain_rads[i] ** j_tuple[i] for i in range(dimension))

            # compute final coefficient: (f^(j)(c) / j!) * r^j
            final_coeff = (RIF(deriv_at_c_real) / j_factorial) * radii_term

            # extract the monomial and add to polynomial
            monomial = tm_ring.monomial(*j_tuple)
            total_poly += final_coeff * monomial

    return total_poly, vars_list, y_names


def compute_univariate_remainder(func: 'sage.symbolic.expression.Expression', var: 'sage.symbolic.expression.Expression',
                                 domain: Interval, ref_point_val: float, order: int) -> Interval:
    """Computes the Lagrange remainder bound for the univariate case (old, simple approach)"""
    df_n1 = func.derivative(var, order + 1)

    if df_n1 == 0:
        return Interval(0)

    df_n1_rigorous = _make_expression_rigorous(df_n1)

    deriv_bound = Interval.bound_function(
        func=df_n1_rigorous,
        var_intervals={str(var): domain}
    )
    domain_centered = domain - Interval(ref_point_val)
    domain_term = domain_centered ** (order + 1)

    n1_factorial = factorial(order + 1)
    remainder_bound = (deriv_bound / n1_factorial) * domain_term
    return remainder_bound


# Multivariate Legrange Remainder
def compute_multivariate_remainder(
    func: 'sage.symbolic.expression.Expression', vars: Tuple,
    var_names: Tuple[str, ...], domains: List[Interval],
    ref_point_vals: Tuple, order: int, dimension: int
) -> Interval:
    """computes the Lagrange remainder bound for the multivariate case
        xin chen method"""
    k_plus_1 = order + 1
    derivative_bounds = {}
    var_intervals = {name: dom for name, dom in zip(var_names, domains)}

    multi_indices_generator = IntegerVectors(k_plus_1, dimension, min_part=0)

    for j_vec in multi_indices_generator:
        j_tuple = tuple(j_vec)

        deriv_args = []
        for i in range(dimension):
            if j_tuple[i] > 0:
                deriv_args.append(vars[i])
                deriv_args.append(j_tuple[i])

        deriv_expr = func.derivative(*deriv_args)

        if deriv_expr == 0:
            deriv_bound = Interval(0)
        else:
            # rigorous_deriv = _make_expression_rigorous(deriv_expr)
            deriv_bound = Interval.bound_function(
                func=deriv_expr,
                var_intervals=var_intervals
            )

        derivative_bounds[j_tuple] = deriv_bound

    return multivariate_lagrange_remainder(
        max_order=order,
        dimension=dimension,
        domain=domains,
        ref_point=ref_point_vals,
        derivative_bounds=derivative_bounds
    )


def multivariate_lagrange_remainder(max_order, dimension, domain: List[Interval], ref_point: Tuple, derivative_bounds: Dict[Tuple[int, ...], Interval]) -> Interval:
    """
    calculates lagrange remainder bound using equation 2.7 from xin chen's thesis

    requires calling function to provide bounds for all (k+1)-th order partial derivatives
    of the original funciton over the domain

    returns an interval object representing the bound for the  lagrange remainder r_k(x)
    """
    if max_order is None:
        raise ValueError("TaylorModel must have a 'max_order' (k) to compute the (k+1) remainder.")

    k_plus_1 = max_order + 1
    n = dimension

    # calculate the (x-c) term over the domain
    centered_domain = tuple(d - Interval(r) for d, r in zip(domain, ref_point))

    # calculate the 1/(k+1)! term using sage factorial to get exact rational number
    try:
        factorial_term = 1 / factorial(k_plus_1)
    except (OverflowError, ValueError):
        raise ValueError(f"(k+1) = {k_plus_1} is too large to compute factorial.")

    total_remainder_bound = Interval(0)

    # iterate over all multi-indicies j = (j1,...,jn) such that sum(j) = k+1
    # sage's integervectors(s, k, min_part=0) finds all vectors of length k that sum to s
    multi_indices_generator = IntegerVectors(k_plus_1, n, min_part=0)

    num_indices_found = 0
    for j_vec in multi_indices_generator:
        num_indices_found += 1

        j_tuple = tuple(j_vec)

        # get bound for hte partial derivative (d^(k+1)f / dx^j)
        if j_tuple not in derivative_bounds:
            raise KeyError(f"Missing derivative bound for multi-index {j_tuple} ")

        deriv_bound = derivative_bounds[j_tuple]

        # get bound for the monomial term Product[ (x_i - c_i)^j_i ]
        monomial_bound = prod(centered_domain[i] ** j_tuple[i] for i in range(n))

        # add bound to total sum
        total_remainder_bound += (deriv_bound * monomial_bound)

    if num_indices_found == 0 and k_plus_1 > 0:
        pass  # No terms to sum, total_remainder_bound is Interval(0)

    # multiply the final sum by the factorial term
    final_remainder = factorial_term * total_remainder_bound

    return final_remainder


def compute_taylor_model(
        func: 'sage.symbolic.expression.Expression', var_names: Tuple[str, ...], domains: List[Interval],
        order: int, ref_point: Tuple = None) -> 'TaylorModel':
    """computes a rigorous TM for a univariate or multivariate function
        - func: Sage symbolic expression to compute around
        - var_names: tuple of variable names like ('x', 'y')
        - domains: list of intervals, one for each variable
        - order: maximum order (total degree) of the polynomial
        - ref_point: expansion point(defaults to the domain midpoint)
    """
    # validate inputs and generate reference point
    dimension, ref_point_vals = validate_and_prepare_inputs(var_names, domains, order, ref_point)

    sage_poly, vars, y_names = compute_polynomial_terms(
        func, var_names, domains, ref_point_vals, order, dimension
    )
    tm_poly = Polynomial(_poly=sage_poly)

    # compute the remainder bound
    if dimension == 1:
        var = vars[0] if isinstance(vars, tuple) else vars
        remainder_bound = compute_univariate_remainder(func, var, domains[0], ref_point_vals[0], order)
    else:
        remainder_bound = compute_multivariate_remainder(func, vars, var_names, domains, ref_point_vals, order, dimension)

    # store the normalized domains/refpts NOT real world domains
    normalized_domains = [Interval(-1, 1)] * dimension
    normalized_refpts = tuple([0.0] * dimension)

    return TaylorModel(
        poly=tm_poly,
        rem=remainder_bound,
        domain=normalized_domains,
        ref_point=normalized_refpts,
        max_order=order
    )


def init_taylor_model(my_func, var_names, domains, order, ref_point: Tuple = None, expand_function: bool = False) -> TaylorModel:
    """
    taylor model initializer!
    factory function for creating new Taylor model objects depending on the 'expand_function' flag

    if expand_function = True:
    - computes a TM for an arbitrary symbolic function
    - computes a non-zero lagrange remainder bound

    if expand_function = False:
    - intialize a TM for a constant, variable or polynomial
    - initial remainder is set to [0,0] (mimic INTLAB, CORA, Flow*)
    - any terms in the function (if its a polynomial) with degree > order are truncated
        and bound is added to remainder
    """

    dimension, ref_point_vals = validate_and_prepare_inputs(var_names, domains, order, ref_point)

    # path 1: expand and compute the lagrange remainder
    if expand_function:

        return compute_taylor_model(
            func=my_func,
            var_names=var_names,
            domains=domains,
            order=order,
            ref_point=ref_point_vals
        )

    # path 2: initialize from a constant, variable or polynomial with R=0

    # create the polynomial ring with RIF points
    ring = PolynomialRing(RIF, names=var_names)
    gens = ring.gens()

    def _const_poly(val) -> 'Polynomial':
        return Polynomial(_poly=ring(RIF(val)), _ring=ring)

    # default remainder is [0, 0]-> add any truncation error
    zero_rem = Interval(0, 0)

    # case a: initialize from constant
    if isinstance(my_func, Interval):
        # my_func is already an Interval object. Unwrap it to get the RIF.
        poly = _const_poly(my_func._interval)
        tm = TaylorModel(poly=poly, rem=zero_rem, domain=list(domains), ref_point=ref_point_vals, max_order=order)
        tm.sweep()
        return tm

    # case b: init from scalar constants
    if isinstance(my_func, (int, float, str, RIF_TYPE, RealLiteral, RealNumber, Rational, Integer)):
        try:
            poly = _const_poly(my_func)
            tm = TaylorModel(poly=poly, rem=zero_rem, domain=list(domains), ref_point=ref_point_vals, max_order=order)
            tm.sweep()
            return tm
        except Exception as e:
            raise TypeError(f"Could not initialize constant TM from source '{my_func}': {e}")

    # case c: initialize from symbolic expression
    # must be poly. no normalization done here
    elif isinstance(my_func, Expression):
        try:
            vset = list(my_func.variables())
            if len(vset) == 1 and my_func == SR.var(str(vset[0])):
                vname = str(vset[0])
                if vname in list(var_names):
                    idx = list(var_names).index(vname)
                    poly = Polynomial(_poly=gens[idx], _ring=ring)
                    tm = TaylorModel(poly=poly, rem=zero_rem, domain=list(domains), ref_point=ref_point_vals, max_order=order)
                    tm.sweep()
                    return tm
        except Exception:
            pass
        
        # coerce expression into poly over RIF 
        # if it fails its not a polynomial (contains sin/cos/exp/etc)
        try:
            expr_rig = _make_expression_rigorous(my_func)
            sage_poly_untruncated = ring(expr_rig)
        except Exception as e:
            raise TypeError(
                "init_taylor_model(expand_function=False) only supports polynomial Expressions "
                f"in variables {var_names}. Expression='{my_func}' could not be coerced to a polynomial. "
                "Use expand_function=True for non-polynomials. "
                f"Coercion error: {e}"
            )
        
        kept_coeffs = {}
        trunc_rem = Interval(0.0, 0.0)
        dom_tuple = tuple(domains)

        raw_items = sage_poly_untruncated.dict().items()
        for exps, coeff in raw_items:
            # normalize exponent tuple to length=dimension
            if dimension == 1:
                if exps == ():
                    exp_tuple = (0,)
                    dict_key = 0  
                elif isinstance(exps, (int,)):
                    exp_tuple = (int(exps),)
                    dict_key = int(exps)
                else:
                    # handle if Sage gives tuples in univariate
                    exp_tuple = (int(exps[0]),)
                    dict_key = int(exps[0])
            else:
                if exps == ():
                    exp_tuple = (0,) * dimension
                    dict_key = (0,) * dimension
                else:
                    exp_tuple = tuple(int(e) for e in exps)
                    dict_key = exp_tuple

            total_deg = sum(exp_tuple)

            if total_deg <= order:
                kept_coeffs[dict_key] = coeff
            else:
                # bound the dropped monomial over provided domain
                trunc_rem += bound_monomial(coeff, exp_tuple, dom_tuple)

        # build truncated polynomial
        try:
            sage_poly_trunc = ring(kept_coeffs)
        except Exception as e:
            raise RuntimeError(f"Failed to reconstruct truncated polynomial in ring {ring}: {e}")

        poly = Polynomial(_poly=sage_poly_trunc, _ring=ring)
        rem = zero_rem + trunc_rem

        # enforce remainder contains 0
        if not rem.contains(0.0):
            # should never happen but explcit fallback
            rem = rem.union(Interval(0.0, 0.0))

        tm = TaylorModel(
            poly=poly,
            rem=rem,
            domain=list(domains),
            ref_point=ref_point_vals,
            max_order=order
        )
        tm.sweep()
        return tm

    raise TypeError(f"Unsupported `my_func` type: {type(my_func)}. Must be Interval, scalar, or polynomial Expression.")
