"""Interval arithmetic utilities backed by Sage RIF."""

import numbers
from typing import Callable, Dict, Union

from sage.all import RIF
RIF_TYPE = type(RIF(0))


class Interval:
    """Closed interval with directed rounding operations."""
    def __init__(self, lower: Union[float, str, RIF, 'Interval'], upper: Union[float, str, None] = None):
        """Initialize an interval.

        Args:
            lower: Lower bound or another Interval.
            upper: Upper bound. If None, creates a point interval.

        Raises:
            ValueError: If bounds are invalid or non-finite.
        """

        if isinstance(lower, Interval):
            self._interval = lower._interval
        elif isinstance(lower, RIF_TYPE):
            self._interval = lower

        else:
            # Coerce inputs to RIF (strings, floats, etc)
            if upper is None:
                # Point/degenerate interval
                self._interval = RIF(lower)
            else:
                # Interval w explicit bounds
                # Relies on RIF auto-verifying lower <= upper during creation
                self._interval = RIF(lower, upper)

        if self._interval.is_NaN():
            # Check for NaN input
            raise ValueError(f"Invalid interval: {self._interval} is not finite or lower > upper.")

    @property
    def is_nan(self):
        return self._interval.is_NaN()

    @staticmethod
    def bound_function(func: Callable, var_intervals: Dict[str, 'Interval']) -> 'Interval':
        """Bound a symbolic function using interval substitution."""

        # safety checks - what format is the input function in?
        if hasattr(func, 'expression'):  # callableSYmbolicExpression -> func.function(x)
            expression_to_bound = func.expression()
            symbolic_vars = func.variables()

        elif hasattr(func, 'variables'):  # plain expression
            expression_to_bound = func
            symbolic_vars = func.variables()

        else:  # unknown/unsupported
            raise TypeError(f"Unsupported format of 'func' {func}. bound_function() expects a Sage symbolic expression or callable expression.")

        subs_dict = {}
        for svar in symbolic_vars:
            var_name = str(svar)
            if var_name in var_intervals:
                # unwrap Interval object & replace with underlying Sage RIF (to avoid unerlying issues)
                subs_dict[svar] = var_intervals[var_name]._interval
            else:
                raise ValueError(f"An interval for variable '{var_name}' was not provided.")

        result = expression_to_bound.subs(subs_dict)

        return Interval(result)

    # Properties and Helpers
    @property
    def lower(self):
        # returns lower bound as an MPFR number
        return self._interval.lower()

    @property
    def upper(self):
        # returns upper bound as an MPFR number
        return self._interval.upper()

    def width(self):
        return self.upper - self.lower

    def midpoint(self):
        return self._interval.center()

    def radius(self):
        return self.width() / 2.0

    @staticmethod
    def _coerce(other) -> 'Interval':
        if isinstance(other, Interval):
            return other
        return Interval(other)

    # Core arithemtic using Sage's optimized formulas and directed rounding
    def __add__(self, other: Union['Interval', float, int]) -> 'Interval':
        other = self._coerce(other)
        return Interval(self._interval + other._interval)

    def __radd__(self, other: Union['Interval', float, int]) -> 'Interval':
        return self.__add__(other)

    def __sub__(self, other: Union['Interval', float, int]) -> 'Interval':
        other = self._coerce(other)
        return Interval(self._interval - other._interval)

    def __rsub__(self, other: Union['Interval', float, int]) -> 'Interval':
        other = self._coerce(other)
        # NOTE: RIF handles R-L rigorously, which is equivalent to -(L-R)
        return Interval(other._interval - self._interval)

    def __mul__(self, other: Union['Interval', float, int]) -> 'Interval':
        other = self._coerce(other)
        return Interval(self._interval * other._interval)

    def __rmul__(self, other: Union['Interval', float, int]) -> 'Interval':
        return self.__mul__(other)

    def __truediv__(self, other: Union['Interval', float, int]) -> 'Interval':
        other = self._coerce(other)

        # NOTE: raises an error for division by an interval containing zero
        if other._interval.contains_zero():
            raise ZeroDivisionError(f"Interval {other} contains zero, division undefined")

        return Interval(self._interval / other._interval)

    def __rtruediv__(self, other: Union['Interval', float, int]) -> 'Interval':
        other = self._coerce(other)
        # NOTE: RIF handles R / L rigorously
        if self._interval.contains_zero():
            raise ZeroDivisionError(f"Interval {self} contains zero, division undefined")
        return Interval(other._interval / self._interval)

    # Intrinsic/Elementary functions
    def __pow__(self, n: int) -> 'Interval':
        # NOTE: sage rigorously handles all cases (even/add, negative & crossing 0) - don't need xin chen approach
        if not isinstance(n, numbers.Integral):
            raise TypeError("Exponent must be an integer for rigorous interval power.")
        return Interval(self._interval ** n)
    
    def __neg__(self) -> 'Interval':
        return Interval(-self._interval)

    def abs(self) -> 'Interval':
        return Interval(self._interval.abs())

    def __abs__(self) -> 'Interval':
        return self.abs()

    def sqrt(self) -> 'Interval':
        # NOTE: sage rigorously defines sqrt() and error cases
        return Interval(self._interval.sqrt())

    def exp(self) -> 'Interval':
        return Interval(self._interval.exp())

    def log(self) -> 'Interval':
        # NOTE: sage .log() is rigorously defined + handles non-positive case internally
        if self.lower <= 0:
            raise ValueError("Log undefined for nonpositive interval")
        return Interval(self._interval.log())

    def sin(self) -> 'Interval':
        # NOTE: sage's .sin() is a wrapper around the MPFI library's rigorous sine.
        # handles critical points (k*pi/2) correctly and computes the range
        return Interval(self._interval.sin())

    def cos(self) -> 'Interval':
        return Interval(self._interval.cos())

    def tan(self) -> 'Interval':
        # NOTE: sage tan() raises an error if the interval contains a singularity (e.g., pi/2)
        return Interval(self._interval.tan())

    # Set operations & Comparison
    def intersection(self, other: 'Interval') -> 'Interval':
        """ returns intersection of current interval with other object.
        if empty -> raises an exception
        """
        other = self._coerce(other)
        result = self._interval.intersection(other._interval)

        if result.is_NaN():
            raise ValueError(f"The intervals {self} and {other} do not intersect.")

        return Interval(result)

    def hull(self, other: 'Interval') -> 'Interval':
        """returns the interval hull (union) of self and other"""
        other = self._coerce(other)
        return Interval(self._interval.union(other._interval))

    def contains(self, x: Union[float, int]) -> bool:
        return x in self._interval

    def encloses(self, other: Union['Interval', float, int]) -> bool:
        """checks if this interval completely encloses the 'other' interval"""
        other = self._coerce(other)
        return self.lower <= other.lower and self.upper >= other.upper

    def __lt__(self, other):
        other = self._coerce(other)
        return self._interval < other._interval

    def __gt__(self, other):
        other = self._coerce(other)
        return self._interval > other._interval

    def __eq__(self, other):
        """check for exact equality of bounds"""
        if not isinstance(other, (Interval, float, int, str)):
            return NotImplemented
        other = self._coerce(other)
        return self._interval == other._interval

    # Utils
    def __repr__(self):
        return f"Interval([{self.lower}, {self.upper}])"
