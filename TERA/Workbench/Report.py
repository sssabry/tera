"""Console reporting and validation helpers."""

from typing import Dict
import math

import numpy as np

from TERA.Workbench.Results import ReachResult

class Report:
    """
    Emit standardized console output and validation summaries.
    """

    @staticmethod
    def print_header(result: ReachResult):
        """Print a standardized run header."""
        cfg = result.config
        ep = cfg.get('engine_params', {})
        
        print("\n" + "="*70)
        print(f"--- Starting {result.system_type.upper()} Test: {cfg.get('name')} ---")
        print(f"Mode: {ep.get('setting', 'single_step')}")
        print(f"System: {cfg.get('f_expr')}")
        print(f"Fixed Step Mode: {ep.get('fixed_step_mode')}, Preconditioning: {ep.get('precondition_setup')}")
        print(f"Initial Set: {cfg.get('initial_set')}")
        print(f"Time: [0, {cfg.get('time_horizon')}], Step: {cfg.get('step_size')}, Order: {cfg.get('order')}")

        # system-specific context
        if result.system_type == 'hybrid':
            print(f"Hybrid Config: Max Jumps={ep.get('max_jumps')}, Methods: Intersection={ep.get('intersection_method')}, Aggregation={ep.get('aggregation_method')}")
        elif result.system_type == 'stochastic':
            print(f"Stochastic Config: Delta={ep.get('delta','0.001 (Default)')}")

        print("-" * 70)

        if result.flowpipe:
            print(f"STATUS: {result.status}. Total computation time: {result.runtime:.4f} seconds")
            print(f"Computed {len(result.flowpipe)} segments.")
            
        print("-" * 70)

    @staticmethod
    def check_safety(result: ReachResult, unsafe_sets: Dict):
        """
        Check whether the flowpipe intersects any unsafe set.

        Args:
            result: ReachResult to analyze.
            unsafe_sets: List of unsafe set specs keyed by dimension index.

        Returns:
            "SAFE", "BREACHED", or "N/A".
        """
        if not unsafe_sets or not result.flowpipe:
            return "N/A"
        
        print("\n--- SAFETY VERIFICATION ---")
        
        for seg_idx, segment in enumerate(result.flowpipe):
            # get the interval bounds for the current reachable segment
            t_int = segment['time_interval_abs']
            t_lower = float(t_int.lower)
            t_upper = float(t_int.upper)
            computed_bounds = segment['tmv'].bound() 
            
            for unsafe_item in unsafe_sets:
                # skip if unsafe set hasn't "activated" yet
                t0 = float(unsafe_item.get("start_time", 0.0))
                if t_upper < t0:
                    continue
                    
                is_breached = True
                dims_to_check = unsafe_item["dims"]
                
                for dim_idx, unsafe_interval in dims_to_check.items():
                    comp = computed_bounds[dim_idx]
                    
                    # standard intersection check
                    if comp.lower > unsafe_interval.upper or comp.upper < unsafe_interval.lower:
                        is_breached = False
                        break
                
                if is_breached:
                    print(f"  [!] BREACH DETECTED on t ∈ [{t_lower:.4f}, {t_upper:.4f}]")
                    print(f"  Reason: Unsafe set active from t={t0:.4f} (violation may occur within this segment)")
                    return "BREACHED"

        print("  RESULT: No unsafe sets intersected. System is SAFE.")
        return "SAFE"

    @staticmethod
    def validate(result: ReachResult, tolerance=1e-4):
        """
        Validate final bounds against expected benchmarks.

        Args:
            result: ReachResult to validate.
            tolerance: Absolute tolerance used for precision checks.

        Returns:
            True if validation passes, otherwise False.
        """
        if not result.flowpipe:
            print("Validation: No flowpipe available.")
            return False

        computed_bounds = result.get_final_bounds()
        expected_bounds = result.config.get('expected_final_bounds')
        expected_width = result.config.get('expected_final_width')
        
        print("\n--- FINAL STATE VALIDATION ---")
        all_pass = True
        
        for i, var in enumerate(result.state_vars):
            comp = computed_bounds[i]
            width = float(comp.upper) - float(comp.lower)
            print(f"{str(var):<10} | [{float(comp.lower):.6f}, {float(comp.upper):.6f}] (Width: {width:.6f})")

        if expected_bounds:
            print("\nComparing against expected benchmarks...")
            for i, (comp, exp) in enumerate(zip(computed_bounds, expected_bounds)):
                if i < len(result.state_vars):
                    var_name = str(result.state_vars[i])
                else:
                    var_name = f"Var {i}"

                l_diff = abs(float(comp.lower) - float(exp.lower))
                u_diff = abs(float(comp.upper) - float(exp.upper))
                
                is_precise = (l_diff < tolerance) and (u_diff < tolerance)
                is_enclosed = comp.encloses(exp) if hasattr(comp, 'encloses') else False

                if is_precise:
                    print(f"\n  PASS (Precise): {var_name}")
                    print(f"    Computed: {comp}")
                    print(f"    Expected: {exp}")
                    print(f"    Diff: L={l_diff:.2e}, U={u_diff:.2e}")
                elif is_enclosed:
                    print(f"\n  PASS (Loose): {var_name} (Encloses expected)")
                    print(f"    Computed: {comp}")
                    print(f"    Expected: {exp}")
                    print(f"    Diff: L={l_diff:.2e}, U={u_diff:.2e}")
                else:
                    print(f"\n  FAIL: {var_name} (Does not enclose or match within tolerance)")
                    print(f"    Computed: {comp}")
                    print(f"    Expected: {exp}")
                    print(f"    Diff: L={l_diff:.2e}, U={u_diff:.2e}")
                    all_pass = False


        if expected_width is not None:
            width_pass = Report.validate_precision_width(computed_bounds, expected_width, tolerance)
            all_pass = all_pass and width_pass

        status = "PASSED" if all_pass else "FAILED"
        print(f"\nSUMMARY: Validation {status}")
        print("="*70 + "\n")
        return all_pass
    
    @staticmethod
    def validate_precision_width(computed_bounds, target_width: float, tolerance=1e-3):
        """
        Validate enclosure width against a target maximum width.

        Args:
            computed_bounds: List of interval bounds.
            target_width: Target maximum width.
            tolerance: Allowed slack.

        Returns:
            True if the width is within tolerance, otherwise False.
        """        
        # find max width across dimensions
        max_comp_width = max([float(b.upper - b.lower) for b in computed_bounds])
        
        print(f"  Target Max Width:   {target_width:.6f}")
        print(f"  Computed Max Width: {max_comp_width:.6f}")
        
        if max_comp_width <= target_width + tolerance:
            print(f"  [WIDTH]  PASS: Enclosure precision matches or exceeds benchmark.")
            return True
        else:
            diff = max_comp_width - target_width
            print(f"  [WIDTH]  FAIL: Enclosure is {diff:.4f} wider than benchmark.")
            return False
        
    @staticmethod
    def print_final_set(result: ReachResult):
        """Print final reachable-set bounds and widths for any successful run."""
        if not result.flowpipe:
            print("\n[Final Set] Empty flowpipe.\n")
            return

        computed_bounds = result.get_final_bounds()

        print("\nFinal Reachable Set Bounds (t_end)")
        print(f"{'Dim':<10} | {'Interval':<30} | {'Width':<12}")
        print("-" * 60)

        for i, var in enumerate(result.state_vars):
            comp = computed_bounds[i]
            low = float(comp.lower)
            high = float(comp.upper)
            width = high - low
            interval_str = f"[{low:10.5f}, {high:10.5f}]"
            print(f"{str(var):<10} | {interval_str:<30} | {width:<12.6f}")
    
    @staticmethod
    def print_stochastic_stats(result: ReachResult):
        """Print stochastic statistics and MC comparison if available."""
        if result.system_type != 'stochastic' or 'traces' not in result.validation_data:
            return

        flowpipe = result.flowpipe
        if not flowpipe:
            print("\n[Stochastic Stats] Empty flowpipe.\n")
            return

        payload = result.validation_data['traces']

        # Backwards compatible unpacking:
        # Old: (t_grid, Xstoch)
        # New: (t_grid, Xstoch, Xdet, X0)
        if not isinstance(payload, (tuple, list)) or len(payload) < 2:
            print("\n[Stochastic Stats] Invalid trace payload format.\n")
            return

        t_grid = payload[0]
        Xstoch = payload[1]
        Xdet = payload[2] if len(payload) >= 3 else None

        # # Final bounds
        # print("\nFinal Reachable Set Bounds (t_end)")
        # last_seg = flowpipe[-1]
        # final_tmv = last_seg['tmv']
        # r_final = float(last_seg.get('stochastic_radius', 0.0))

        # det_bounds = final_tmv.bound()
        # diag_p_inv = result.validation_data.get('diag_P_inv_upper')

        # # NOTE: This is a box overapproximation of ball dilation by radius r_final.
        # print(f"{'Dim':<10} | {'Deterministic Interval':<30} | {'Stochastic box overapprox (1-δ)':<30}")
        # print("-" * 90)

        # for i, var in enumerate(result.state_vars):
        #     det_low, det_high = float(det_bounds[i].lower), float(det_bounds[i].upper)
        #     if diag_p_inv is not None and i < len(diag_p_inv):
        #         delta_i = r_final * math.sqrt(diag_p_inv[i])
        #         delta_i = math.nextafter(delta_i, math.inf)
        #     else:
        #         delta_i = r_final
        #     stoch_low, stoch_high = det_low - delta_i, det_high + delta_i

        #     det_str = f"[{det_low:10.5f}, {det_high:10.5f}]"
        #     stoch_str = f"[{stoch_low:10.5f}, {stoch_high:10.5f}]"
        #     print(f"{str(var):<10} | {det_str:<30} | {stoch_str:<30}")

        # Validation statistics: compare empirical deviation vs AMGF radius:
        max_empirical_dev = 0.0
        max_theoretical_bound = 0.0
        violation_count = 0
        used_pairing = (Xdet is not None)
        L_mat = result.validation_data.get("L_matrix")

        for segment in flowpipe:
            t_end = float(segment['time_interval_abs'].upper)
            r_bound = float(segment.get('stochastic_radius', 0.0))
            max_theoretical_bound = max(max_theoretical_bound, r_bound)

            # Map segment end time to nearest MC grid index
            idx = int(np.argmin(np.abs(t_grid - t_end)))

            step_stoch = Xstoch[:, idx, :]

            if used_pairing:
                step_det = Xdet[:, idx, :]
                errs = step_stoch - step_det
                if L_mat is not None:
                    # weighted norm: ||L e||_2 (theorem-faithful when weighted metric used)
                    errs_hat = errs @ np.asarray(L_mat, dtype=float).T
                    distances = np.linalg.norm(errs_hat, axis=1)
                else:
                    distances = np.linalg.norm(errs, axis=1)
            else:
                # Fallback nominal "center slice" at local time h (NOT theorem-faithful)
                tmv = segment['tmv']
                h = t_end - float(segment['time_interval_abs'].lower)

                dom_len = len(tmv.tms[0].domain) - 1  # spatial vars count
                eval_args = tuple([0.0] * dom_len + [h])

                center = np.array([float(tm.poly.evaluate(eval_args)) for tm in tmv.tms], dtype=float)
                distances = np.linalg.norm(step_stoch - center, axis=1)

            max_dev_at_step = float(np.max(distances))
            max_empirical_dev = max(max_empirical_dev, max_dev_at_step)

            tol = 1e-12 * (1.0 + abs(r_bound))
            if max_dev_at_step > r_bound + tol:
                violation_count += 1

        print("\nValidation Statistics")
        if used_pairing:
            if L_mat is not None:
                print("Metric: max_k ||L (X_t^k - x_t^k)|| (paired deterministic trajectories, weighted)")
            else:
                print("Metric: max_k ||X_t^k - x_t^k|| (paired deterministic trajectories)")
        else:
            print("Metric: distance to nominal TM center slice (fallback; not theorem-faithful)")

        if used_pairing:
            status = "PASS" if max_empirical_dev <= max_theoretical_bound else "WARN (Local Violation)"
        else:
            status = "INFO (Not theorem-faithful)"

        print(f"  Status: {status}")
        print(f"  Max Theoretical Bound (r): {max_theoretical_bound:.6f}")
        print(f"  Max Empirical Deviation:   {max_empirical_dev:.6f}")
        print(f"  Safety Margin:             {max_theoretical_bound - max_empirical_dev:.6f}")

        if violation_count == 0:
            print("  Trajectory Violations:     0 steps had outliers (100% enclosed at sampled times)")
        else:
            print(f"  Trajectory Violations:     {violation_count} steps had outliers (sampled times)")

        print("=" * 70 + "\n")

    @staticmethod
    def print_hybrid_summary(result: ReachResult):
        """Print a brief hybrid mode path summary."""
        if result.system_type != 'hybrid':
            return

        unique_modes = sorted(list(set(seg['mode'] for seg in result.flowpipe)))
        total_time = float(result.flowpipe[-1]['time_interval_abs'].upper)
        
        print(f"Hybrid Path Summary:")
        print(f"  Total Segments: {len(result.flowpipe)}")
        print(f"  Modes Traversed: {', '.join(unique_modes)}")
        print(f"  Final Horizon: {total_time:.4f}s")
        print("-" * 70)
