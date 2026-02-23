"""TMVector utilities for collections of Taylor models."""

from typing import List, Union
from copy import deepcopy
import numpy as np
from sage.all import RIF

from TERA.TMCore.TaylorModel import TaylorModel
from TERA.TMCore.Polynomial import Polynomial
from TERA.TMCore.Interval import Interval
from TERA.TMCore.TMComputer import init_taylor_model

Scalar = Union[int, float, Interval]

class TMVector:
    """Container for a vector of Taylor models."""
    # used to force numpy delegates to TMVector's operators
    __array_priority__ = 1000
    def __init__(self, tms: List[TaylorModel]):
        """Initialize a TMVector and validate compatibility."""

        if not tms:
            raise ValueError("TMVector cannot be initialized with an empty list")

        # validate compatibility of tms
        first_tm = tms[0]

        if not hasattr(first_tm, 'poly') or not hasattr(first_tm, 'remainder'):
            raise TypeError("All items in 'tms' must be TaylorModel objects")

        # check compatibility
        for i, tm in enumerate(tms):
            if tm.dimension != first_tm.dimension:
                raise ValueError(f"TaylorModel at index {i} introduces an incompatible dimension")
            if tm.max_order != first_tm.max_order:
                raise ValueError(f"TaylorModel at index {i} introduces an incompatible maximum order")

        # no errors raised? initialize vars
        self.domain = deepcopy(first_tm.domain)
        self.ref_point = deepcopy(first_tm.ref_point)
        self.max_order = first_tm.max_order
        self.dimension = first_tm.dimension
        self.tms = tms
        self.vector_dimension = len(tms)

    def __len__(self) -> int:
        return self.vector_dimension

    def __getitem__(self, key: int) -> TaylorModel:
        return self.tms[key]

    def __repr__(self) -> str:
        return f"TMVector (dim={self.vector_dimension}) [\n,\n".join(f"  {tm}" for tm in self.tms) + "\n]"

    def copy(self) -> 'TMVector':
        """Create a safe full copy of the TMVector."""
        return TMVector([tm.copy() for tm in self.tms])

    # unary operations
    def __neg__(self) -> 'TMVector':
        """Return element-wise negation."""
        new_tms = [-tm for tm in self.tms]
        return TMVector(new_tms)

    def power(self, n: int) -> 'TMVector':
        """Return element-wise powers."""
        new_tms = [tm ** n for tm in self.tms]
        return TMVector(new_tms)

    def get_remainders(self) -> List[Interval]:
        """Return remainder intervals for each element."""
        return [tm.remainder for tm in self.tms]

    def get_polynomials(self) -> 'TMVector':
        """Return a TMVector with zeroed remainders."""
        new_tms = []
        for tm in self.tms:
            poly_only_tm = tm.copy()

            poly_only_tm.remainder = Interval(0)

            new_tms.append(poly_only_tm)

        return TMVector(new_tms)

    def get_state_remainders(self) -> List[Interval]:
        """Return remainders for state variables only."""
        state_dimension = len(self.tms) - 1
        if state_dimension < 0:
            return []

        return [tm.remainder for tm in self.tms[:state_dimension]]

    def substitute(self, substitutions: dict) -> 'TMVector':
        """Apply substitutions to each Taylor model."""
        new_tms = [tm.substitute(substitutions) for tm in self.tms]
        return TMVector(new_tms)

    def bound(self) -> List[Interval]:
        """Return bounds for each Taylor model."""
        return [tm.bound() for tm in self.tms]
    
    def truncate(self, new_order: int):
        for tm in self.tms:
            tm.truncate(new_order)
        return self
    
    def evaluate(self, point: List) -> List[Interval]:
        return [tm.evaluate(point) for tm in self.tms]
        
    def compose(self, replacements: List['TaylorModel']) -> 'TMVector':
        new_tms = [tm.compose(replacements) for tm in self.tms]
        return TMVector(new_tms)
    
    def get_constant_part(self) -> List[Interval]:
        constants = [tm.get_constant_part() for tm in self.tms]
        return constants
    
    def get_jacobian(self) -> np.ndarray:
        """ extracts the linear coefficients aka the jacobian matrix A from the TM vector
        used in the qr preconditioning strategy in Precondition.py

            Florian's paper defines A as the matrix of the linear part A*x
            A_ij is the coefficient of variable x_j in polynomial p_i
        """
        dimension = len(self.tms)
        
        # initialize as an n by n matrix of 0s
        A = np.zeros((dimension, dimension), dtype=float)
        
        # iterate through state variable's polynomial (rows of A)
        for i in range(dimension):
            poly_obj = self.tms[i].poly.poly
            vars_list = poly_obj.parent().gens()
            
            for j in range(dimension):
                # get coefficient of variable x_j (this might be a polynomial in other vars)
                coeff_poly = poly_obj.derivative(vars_list[j]).constant_coefficient()
                
                # evaluate that coefficient at 0 (constant term) to get the linear Jacobian entry
                # if coeff_poly is constant: constant_coefficient() returns the value
                # if coeff_poly is '2*y': constant_coefficient() returns 0 (Correct for Jacobian)
                try:
                    if hasattr(coeff_poly, 'constant_coefficient'):
                        val_obj = coeff_poly.constant_coefficient()
                    else:
                        val_obj = coeff_poly
                    
                    # handle Interval (RIF) types by taking the center
                    if hasattr(val_obj, 'center'):
                        val = float(val_obj.center())
                    else:
                        val = float(val_obj)
                        
                except Exception:
                    # fallback: if term doesn't exist or conversion fails, derivative is 0
                    val = 0.0
                
                A[i, j] = val
                
        return A
    
    def is_empty(self) -> bool:
        """performs simple check returning True if any interval in domain or remainder
        is Nan or has lower > upper"""
        for tm in self.tms:
            for dim in tm.domain:
                if dim.is_nan:
                    return True
            
            for dim in tm.remainder:
                if dim.is_nan:
                    return True

        return False

    
    # binary operations
    def __add__(self, other: Union['TMVector', TaylorModel, List, np.ndarray, float]) -> 'TMVector':
        """element-wise addition: v1 + v2, v1 + tm, or v1 + constant_vector"""
        
        # Case 1: Vector + Vector (TMVector)
        if isinstance(other, self.__class__):
            if len(self) != len(other):
                raise ValueError("Vector dimensions must match for addition")

            new_tms = [tm1 + tm2 for tm1, tm2 in zip(self.tms, other.tms)]

        # Case 2: Vector + Single TM (Broadcast)
        elif isinstance(other, TaylorModel):
            new_tms = [tm + other for tm in self.tms]

        # Case 3: Vector + List/Array of constants (Element-wise)
        elif isinstance(other, (list, np.ndarray)):
            if len(other) != len(self):
                raise ValueError(f"Vector dimensions must match: TMVector({len(self)}) vs List({len(other)})")
            new_tms = [tm + val for tm, val in zip(self.tms, other)]

        # Case 4: Vector + Scalar Constant (Broadcast)
        elif isinstance(other, (int, float, Interval)):
             new_tms = [tm + other for tm in self.tms]

        else:
            raise NotImplementedError(f"Other of type {type(other)} is not supported for addition")

        return TMVector(new_tms)

    def __sub__(self, other: Union['TMVector', TaylorModel, List, np.ndarray, float]) -> 'TMVector':
        """element-wise subtraction: v1 - v2, v1 - tm, or v1 - constant_vector"""
        
        # Case 1: Vector - Vector (TMVector)
        if isinstance(other, self.__class__):
            if len(self) != len(other):
                raise ValueError("Vector dimensions must match for subtraction")
            new_tms = [tm1 - tm2 for tm1, tm2 in zip(self.tms, other.tms)]

        # Case 2: Vector - Single TM (Broadcast)
        elif isinstance(other, TaylorModel):
            new_tms = [tm - other for tm in self.tms]

        # Case 3: Vector - List/Array of constants (Element-wise)
        elif isinstance(other, (list, np.ndarray)):
            if len(other) != len(self):
                raise ValueError(f"Vector dimensions must match: TMVector({len(self)}) vs List({len(other)})")
            new_tms = [tm - val for tm, val in zip(self.tms, other)]

        # Case 4: Vector - Scalar Constant (Broadcast)
        elif isinstance(other, (int, float, Interval)):
             new_tms = [tm - other for tm in self.tms]

        else:
            raise NotImplementedError(f"Other of type {type(other)} is not supported for subtraction")

        return TMVector(new_tms)

    def __mul__(self, other: Union['TMVector', TaylorModel, List, np.ndarray, float]) -> 'TMVector':
        """
        element-wise multiplication (hadamard product):
        - v1 * v2 (vector * vector)
        - v1 * tm (vector * scalar TM)
        - v1 * vector_constants (element-wise scaling)
        - v1 * c (vector * scalar constant)
        """
        if isinstance(other, self.__class__):
            if len(self) != len(other):
                raise ValueError("Vector dimensions must match for element-wise multiplication")
            new_tms = [tm1 * tm2 for tm1, tm2 in zip(self.tms, other.tms)]

        # multiply all by single TM
        elif isinstance(other, TaylorModel):  # Broadcast single TM
            new_tms = [tm * other for tm in self.tms]

        # multiply by a list/array (element-wise scaling)
        elif isinstance(other, (list, np.ndarray)):
            if len(other) != len(self):
                raise ValueError(f"Scaling vector length ({len(other)}) must match TMVector dimension ({len(self)})")
            new_tms = [tm * val for tm, val in zip(self.tms, other)]

        # multiply all by single constant
        elif isinstance(other, (int, float, Interval)):
            new_tms = [tm * other for tm in self.tms]

        else:
            raise NotImplementedError(
                f"Multiplication not supported for type {type(other)}. Must be TaylorModel, TMVector, list, array, or constant."
            )

        return TMVector(new_tms)
    
    def __truediv__(self, other: Union[List, np.ndarray, float, int]) -> 'TMVector':
        """
        Element-wise division:
        - v1 / vector_constants (element-wise scaling)
        - v1 / constant (broadcast)
        """
        # Case 1: Division by List/Array (Element-wise)
        if isinstance(other, (list, np.ndarray)):
            if len(other) != len(self):
                raise ValueError(f"Scaling vector length ({len(other)}) must match TMVector dimension ({len(self)})")
            
            # Divide each TM by its corresponding scalar
            new_tms = [tm / val for tm, val in zip(self.tms, other)]

        # Case 2: Division by Scalar Constant (Broadcast)
        elif isinstance(other, (int, float, Interval)):
             new_tms = [tm / other for tm in self.tms]

        else:
            return NotImplemented

        return TMVector(new_tms)

    def __rmul__(self, other: Union[TaylorModel, Scalar]) -> 'TMVector':
        """handles scalar * vector"""
        return self.__mul__(other)

    # matrix multiplication A @ v
    def __rmatmul__(self, other):
        if not isinstance(other, (np.ndarray, list)):
            raise TypeError(f"Matrix must be a numpy array or list of lists, not {type(other)}")

        matrix = np.array(other)
        if matrix.ndim != 2:
            raise ValueError("Matrix must be 2D")

        if matrix.shape[1] != len(self):
            raise ValueError(f"Matrix column count ({matrix.shape[1]}) must match vector length ({len(self)}) for A @ v")

        new_tms = []
        for row in matrix:
            # compute dot product, initialize as 0-valued tm
            sum_tm = self.tms[0] * 0.0
            for i in range(len(self)):
                sum_tm = sum_tm + (self.tms[i] * row[i])

            new_tms.append(sum_tm)
        return TMVector(new_tms)

    def __radd__(self, other: TaylorModel) -> 'TMVector':
        """handles tm_scalar + v1"""
        if isinstance(other, TaylorModel):
            new_tms = [other + tm for tm in self.tms]            
            return TMVector(new_tms)
        return NotImplemented

    def __rsub__(self, other: TaylorModel) -> 'TMVector':
        """handles tm_scalar - v1"""
        if isinstance(other, TaylorModel):
            new_tms = [other - tm for tm in self.tms]
            return TMVector(new_tms)
        return NotImplemented
    
    @classmethod
    def from_constants(cls, values: list, prototype: 'TMVector') -> 'TMVector':
        """
        factory method to create a TMVector of constant TMs
        """
        # ensure dimensions match
        if len(values) != len(prototype.tms):
            raise ValueError(f"Dimension mismatch: values ({len(values)}) vs prototype ({len(prototype.tms)})")
            
        new_tms = []
        for val, proto_tm in zip(values, prototype.tms):
            # call the TaylorModel factory method
            const_tm = TaylorModel.from_constant(val, proto_tm)
            new_tms.append(const_tm)
            
        return cls(new_tms)

    def evaluate_symbolic(self, expr, state_vars: list, time_var_name: str = 't') -> TaylorModel:
        """
        Evaluates a symbolic expression (like a reset map or guard) 
        using Taylor Model arithmeti
        """
        import math
        
        # 1. Handle relations: Convert (h <= 0) to (h - 0)
        # Use built-in Sage method to check for relation
        if hasattr(expr, 'is_relational') and expr.is_relational():
            expr = expr.lhs() - expr.rhs()

        # 2. Build the evaluation context
        mapping = {str(v): self.tms[i] for i, v in enumerate(state_vars)}
        
        # Map the time variable (last dimension)
        ring = self.tms[0].poly.ring
        gens = ring.gens()
        domain0 = self.tms[0].domain
        
        try:
            ref_ok = True
            for i, point_i in enumerate(ref0):
                if not domain0[i].contains(point_i):
                    ref_ok = False
                    break
        except Exception:
            # if ref0 malformed, fall back to midpoint recentering
            ref_ok = False

        if not ref_ok:
            # choose a valid center inside the current domain (midpoints)
            ref0 = tuple(float(iv.midpoint()) for iv in domain0)

        time_tm = TaylorModel(
            poly=Polynomial(_poly=gens[-1], _ring=ring),
            rem=Interval(0),
            domain=domain0,
            ref_point=ref0,
            max_order=self.max_order
        )
        mapping[time_var_name] = time_tm

        # 3. Context for math functions
        context = {
            'sin': lambda x: x.sin() if hasattr(x, 'sin') else math.sin(x),
            'cos': lambda x: x.cos() if hasattr(x, 'cos') else math.cos(x),
            'exp': lambda x: x.exp() if hasattr(x, 'exp') else math.exp(x),
            'log': lambda x: x.log() if hasattr(x, 'log') else math.log(x),
        }
        context.update(mapping)

        try:
            # Convert Sage expr to string and handle power notation
            expr_str = str(expr).replace('^', '**')
            # Evaluate using TaylorModel operator overrides
            result = eval(expr_str, {"__builtins__": None}, context)
            
            if not hasattr(result, 'poly'):
                template = self.tms[0].copy()
                template.domain = domain0
                template.ref_point = ref0
                return TaylorModel.from_constant(float(result), template)
            return result
        except Exception as e:
            raise RuntimeError(f"Symbolic evaluation failed: {e}")
