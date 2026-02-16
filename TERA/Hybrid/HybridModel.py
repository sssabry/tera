"""Hybrid automaton data structures."""
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from TERA.TMCore.TMVector import TMVector

@dataclass
class Condition:
    """Represent guard or invariant constraints g(x) <= 0."""
    # List of symbolic expressions (e.g., [x - 5, y + x]) representing g_i(x) <= 0
    constraints: List[Any] 

    def evaluate(tmVector):
        """takes an interval or TM representing g(x)
        used to detect if a flowpipe intersects a guard g(x) <= 0"""

@dataclass
class ResetMap:
    """Represent a discrete reset map x' = r(x)."""
    # e.g., {'v': -0.8 * v, 'x': x}
    mapping: Dict[str, Any]

class Mode:
    """Hybrid mode with dynamics, invariants, and transitions."""
    def __init__(self, name: str, ode_exprs: List[Any], invariant: Condition):
        self.name = name
        self.ode_exprs = ode_exprs  # dx/dt = f(x) for this mode
        self.invariant = invariant
        self.transitions: List[Transition] = []

@dataclass
class Transition:
    """Discrete transition between hybrid modes."""
    source: Mode
    target: Mode
    guard: Condition
    reset: ResetMap
    label: str = ""

class HybridAutomaton:
    """Container for modes, transitions, and initial sets."""
    def __init__(self, modes: List[Mode], state_vars: List[Any], time_var: str):
        # store modes in a dict for easy lookup by name
        self.modes = {m.name: m for m in modes}
        self.state_vars = state_vars
        self.time_var = time_var
        self.initial_sets: List[Tuple[str, TMVector]] = []

    def add_initial_state(self, mode_name: str, tmv: TMVector):
        """Register an initial set for the reachability queue."""
        if mode_name not in self.modes:
            raise ValueError(f"Mode {mode_name} not found in automaton.")
        self.initial_sets.append((mode_name, tmv))

    def get_mode(self, name: str) -> Mode:
        """Return a mode by name."""
        return self.modes[name]
