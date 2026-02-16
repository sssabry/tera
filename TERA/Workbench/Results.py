"""Result container utilities for reachability runs."""

import csv
import time

from TERA.TMFlow import Precondition

class ReachResult:
    """Store outputs from a reachability run."""
    def __init__(self, flowpipe, system_type, state_vars, config, status="SUCCESS", runtime=0.0, safety_status="SAFE"):
        self.flowpipe = flowpipe        # list of reachable segments
        self.system_type = system_type  # 'continuous', 'hybrid', or 'stochastic'
        self.state_vars = state_vars    # symbolic variables
        self.config = config            # dict of order, step_size, etc.
        self.status = status            # computation status
        self.runtime = runtime          # time taken in seconds
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.validation_data = {}         # monte carlo traces or external bounds
        self.safety_status = safety_status

    def get_final_bounds(self):
        """Return bounds for the final flowpipe segment, if available."""
        if not self.flowpipe:
            return None
        last_seg = self.flowpipe[-1]

        if 'tmv' not in last_seg:
            return None
        
        tmv_seg = last_seg['tmv']
        t_abs = last_seg.get('time_interval_abs', None)
        if t_abs is None:
            return tmv_seg.bound()
        
        h = float(t_abs.upper) - float(t_abs.lower)
        tmv_end = Precondition.evaluate_at_t_end(tmv_seg, h, self.config.get("time_var", "t"))
        return tmv_end.bound()
    
    def add_validation_traces(self, traces):
        """Attach validation traces to the result."""
        self.validation_data['traces'] = traces
    
    def export_to_csv(self, file_path):
        """
        Export flowpipe bounds to a CSV file.
        Columns: step, t_start, t_end, optional mode/is_valid/stochastic_radius, then per-variable lower/upper bounds.
        Returns the number of rows written (excluding header).
        """
        flowpipe = self.flowpipe
        state_vars = self.state_vars or []
        var_names = [str(v) for v in state_vars]
        var_count = len(var_names)

        has_mode = any('mode' in seg for seg in flowpipe)
        has_valid = any('is_valid' in seg for seg in flowpipe)
        has_stoch = any('stochastic_radius' in seg for seg in flowpipe)

        header = ["step", "t_start", "t_end"]
        if has_mode:
            header.append("mode")
        if has_valid:
            header.append("is_valid")
        if has_stoch:
            header.append("stochastic_radius")
        for name in var_names:
            header.append(f"{name}_lower")
            header.append(f"{name}_upper")

        rows_written = 0
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for idx, segment in enumerate(flowpipe):
                tmv = segment.get("tmv")
                if tmv is None:
                    continue

                bounds = tmv.bound()
                if var_count and len(bounds) >= var_count:
                    bounds = bounds[:var_count]

                t_abs = segment.get("time_interval_abs")
                if t_abs is not None:
                    t_start = float(t_abs.lower)
                    t_end = float(t_abs.upper)
                else:
                    t_start = ""
                    t_end = ""

                row = [idx, t_start, t_end]
                if has_mode:
                    row.append(segment.get("mode", ""))
                if has_valid:
                    row.append(segment.get("is_valid", True))
                if has_stoch:
                    row.append(segment.get("stochastic_radius", ""))

                for iv in bounds:
                    row.append(float(iv.lower))
                    row.append(float(iv.upper))

                missing = var_count - len(bounds)
                if missing > 0:
                    row.extend(["", ""] * missing)

                writer.writerow(row)
                rows_written += 1

        return rows_written
