"""Task configuration data model."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from TERA.TMCore.Interval import Interval

@dataclass
class TaskConfig:
    """Define inputs and expectations for a reachability task."""
    name: str
    system_type: str  # 'continuous', 'hybrid', 'stochastic'
    vars: List[Any]

    f_expr: Optional[List[Any]] = None
    initial_set: List[Interval] = field(default_factory=list)
    initial_mode: Optional[str] = None

    unsafe_sets: Optional[List[Dict]] = None
    urgent_jumps_mode: bool = True

    time_horizon: float = 1.0
    order: int = 4
    step_size: float = 0.05
    time_var: str = "t"

    remainder_estimation: Optional[List[Interval]] = None
    engine_params: Dict[str, Any] = field(default_factory=dict)

    expected_final_bounds: Optional[List[Interval]] = None
    expected_final_width: float = None

    def __post_init__(self):
        # Ensure engine_params has defaults if not provided
        if "fixed_step_mode" not in self.engine_params:
            self.engine_params["fixed_step_mode"] = True
        if "precondition_setup" not in self.engine_params:
            self.engine_params["precondition_setup"] = "ID"
