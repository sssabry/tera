"""Visualization utilities for reachability results."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import itertools
from TERA.Workbench.Results import ReachResult
from TERA.Stochastic.Plotter import StochasticPlotter

class Visualizer:
    """
    Render reachable sets for continuous, hybrid, and stochastic runs.
    """

    @staticmethod
    def plot(result: ReachResult, dims=None, mode="phase", **kwargs):
        """
        Plot a reachability result.
        
        Args:
            result: ReachResult object containing flowpipe and metadata
            dims: tuple of state variable indices to plot
            mode: "phase" for state-space portrait, "time" for temporal evolution
            **kwargs: plotting overrides (e.g., title, color_map, samples)
        """
        if not result.flowpipe:
            print("Visualizer: No reachable data found to plot.")
            return None

        if dims is None:
            dims = result.config.get('engine_params', {}).get('plot_dims')
            
            if dims is None:
                dims = (0, 1) if len(result.state_vars) > 1 else (0, 0)
        
        x_idx, y_idx = dims
        x_name = str(result.state_vars[x_idx])
        y_name = str(result.state_vars[y_idx])

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 7)))
        
        if mode == "phase":
            Visualizer._render_phase_portrait(ax, result, dims, **kwargs)
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
        elif mode == "time":
            target_idx = kwargs.pop("y_idx", x_idx)
            time_y_name = str(result.state_vars[target_idx])
            Visualizer._render_time_evolution(ax, result, y_idx=target_idx, **kwargs)
            ax.set_xlabel(kwargs.get("time_label", "Time (s)"))
            ax.set_ylabel(time_y_name)

        title_str = kwargs.get("title")
        if not title_str:
            name = result.config.get('name', result.system_type.upper())
            if mode == "phase":
                title_str = f"{name} Analysis: {y_name} vs {x_name}"
            else:
                time_y_name = kwargs.get("time_y_name", time_y_name if 'time_y_name' in locals() else y_name)
                title_str = f"{name} Analysis: {time_y_name} over time"

        ax.set_title(title_str)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if kwargs.get("save_path"):
            plt.savefig(kwargs.get("save_path"))
            print(f"Plot saved to {kwargs.get('save_path')}")
            
        return fig

    @staticmethod
    def _render_phase_portrait(ax, result, dims, **kwargs):
        """dispatches to specific geometry engines based on system type"""
        if result.system_type == 'hybrid':
            Visualizer._draw_hybrid_geometry(ax, result, dims, **kwargs)
        elif result.system_type == 'stochastic':
            Visualizer._draw_stochastic_geometry(ax, result, dims, **kwargs)
        else:
            # check if it's a 'left_right' or standard continuous run
            setting = result.config.get('engine_params', {}).get('setting', 'single_step')
            if setting == 'left_right':
                Visualizer._draw_nonlinear_geometry(ax, result, dims, **kwargs)
            else:
                Visualizer._draw_linear_geometry(ax, result, dims, **kwargs)

    @staticmethod
    def _draw_linear_geometry(ax, result, dims, **kwargs):
        if result is None or getattr(result, "flowpipe", None) is None:
            if kwargs.get("warn", False):
                print("Visualizer: No result/flowpipe to plot.")
            return

        x_idx, y_idx = dims
        edgecolor = kwargs.get("edgecolor", "magenta")
        linewidth = kwargs.get("linewidth", 0.8)
        alpha = kwargs.get("alpha", 0.8)
        facecolor = kwargs.get("facecolor", "none")
        pad_frac = kwargs.get("pad_frac", 0.10)
        warn = kwargs.get("warn", False)
        assume_square_A = kwargs.get("assume_square_A", True)
        projection_mode = kwargs.get("projection_mode", "aabb")  # "aabb" (sound projection enclosure) or "slice" (legacy)
        use_tmv_bounds = kwargs.get("use_tmv_bounds", True)  # prefer flowpipe enclosure over per-step local box

        drawn = 0
        skipped = 0
        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")

        corners_2d = [
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
        ]

        for seg_idx, segment in enumerate(result.flowpipe):
            if use_tmv_bounds:
                tmv = segment.get("tmv", None)
                if tmv is not None:
                    try:
                        bbox = tmv.bound()
                        if x_idx < 0 or y_idx < 0 or x_idx >= len(bbox) or y_idx >= len(bbox):
                            raise IndexError(f"dims {dims} out of bounds for tmv.bound() len={len(bbox)}")

                        bx = bbox[x_idx]
                        by = bbox[y_idx]

                        x0 = float(bx.lower)
                        x1 = float(bx.upper)
                        y0 = float(by.lower)
                        y1 = float(by.upper)

                        rect = patches.Rectangle(
                            (x0, y0),
                            x1 - x0,
                            y1 - y0,
                            linewidth=linewidth,
                            edgecolor=edgecolor,
                            facecolor=facecolor,
                            alpha=alpha,
                        )
                        ax.add_patch(rect)
                        min_x = min(min_x, x0)
                        max_x = max(max_x, x1)
                        min_y = min(min_y, y0)
                        max_y = max(max_y, y1)
                        drawn += 1
                        continue
                    except Exception as exc:
                        if warn:
                            print(f"Visualizer: seg {seg_idx} tmv.bound enclosure failed; falling back to affine: {exc}")

            A = segment.get("A_l", None)
            c = segment.get("c_l", segment.get("center", segment.get("global_center", None)))
            if A is None or c is None:
                skipped += 1
                continue

            try:
                A = np.asarray(A, dtype=float)
                c = np.asarray(c, dtype=float).reshape(-1)
            except Exception:
                if warn:
                    print(f"Visualizer: skip seg {seg_idx} (cannot coerce A/c to float arrays).")
                skipped += 1
                continue

            if A.ndim != 2 or c.ndim != 1:
                if warn:
                    print(f"Visualizer: skip seg {seg_idx} (bad shapes) A.ndim={A.ndim}, c.ndim={c.ndim}")
                skipped += 1
                continue

            global_dim = c.shape[0]
            if x_idx < 0 or y_idx < 0 or x_idx >= global_dim or y_idx >= global_dim:
                if warn:
                    print(f"Visualizer: skip seg {seg_idx} (dims out of bounds) dims={dims}, len(c)={global_dim}")
                skipped += 1
                continue

            if A.shape[0] != global_dim:
                if warn:
                    print(f"Visualizer: skip seg {seg_idx} (A shape mismatch) A={A.shape}, len(c)={global_dim}")
                skipped += 1
                continue
            local_dim = A.shape[1]
            if assume_square_A and (local_dim != global_dim):
                if warn:
                    print(f"Visualizer: skip seg {seg_idx} (A not square) A={A.shape}, len(c)={global_dim}")
                skipped += 1
                continue

            if projection_mode == "slice":
                corners_global = []
                try:
                    for ux, uy in corners_2d:
                        xi = np.zeros(local_dim, dtype=float)
                        if x_idx < local_dim:
                            xi[x_idx] = ux
                        if y_idx < local_dim:
                            xi[y_idx] = uy
                        xg = A @ xi + c
                        corners_global.append((float(xg[x_idx]), float(xg[y_idx])))
                except Exception as exc:
                    if warn:
                        print(f"Visualizer: skip seg {seg_idx} (transform failed): {exc}")
                    skipped += 1
                    continue

                xs = [p[0] for p in corners_global]
                ys = [p[1] for p in corners_global]
                min_x = min(min_x, min(xs))
                max_x = max(max_x, max(xs))
                min_y = min(min_y, min(ys))
                max_y = max(max_y, max(ys))

                poly = patches.Polygon(
                    corners_global,
                    closed=True,
                    linewidth=linewidth,
                    edgecolor=edgecolor,
                    facecolor=facecolor,
                    alpha=alpha,
                )
                ax.add_patch(poly)
                drawn += 1
                continue

            try:
                rowx = A[x_idx, :]
                rowy = A[y_idx, :]

                local_half_widths = segment.get("local_half_widths", segment.get("r_l", None))
                if local_half_widths is None:
                    r = np.ones(local_dim, dtype=float)
                else:
                    r = np.asarray(local_half_widths, dtype=float).reshape(-1)
                    if r.shape[0] != local_dim:
                        if warn:
                            print(
                                f"Visualizer: seg {seg_idx} local_half_widths length mismatch "
                                f"(got {r.shape[0]}, expected {local_dim}); falling back to ones."
                            )
                        r = np.ones(local_dim, dtype=float)

                cx = float(c[x_idx])
                cy = float(c[y_idx])
                rx = float(np.sum(np.abs(rowx) * r))
                ry = float(np.sum(np.abs(rowy) * r))
            except Exception as exc:
                if warn:
                    print(f"Visualizer: skip seg {seg_idx} (projection enclosure failed): {exc}")
                skipped += 1
                continue

            rect = patches.Rectangle(
                (cx - rx, cy - ry),
                2.0 * rx,
                2.0 * ry,
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor=facecolor,
                alpha=alpha,
            )
            ax.add_patch(rect)
            min_x = min(min_x, cx - rx)
            max_x = max(max_x, cx + rx)
            min_y = min(min_y, cy - ry)
            max_y = max(max_y, cy + ry)
            drawn += 1

        if min_x == float("inf") or min_y == float("inf"):
            if warn:
                print("Visualizer: No drawable linear segments (all skipped).")
            return

        dx = max_x - min_x
        dy = max_y - min_y
        pad_x = (dx * pad_frac) if dx > 0 else (1.0 * pad_frac)
        pad_y = (dy * pad_frac) if dy > 0 else (1.0 * pad_frac)

        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(min_y - pad_y, max_y + pad_y)
        ax.grid(True)

    @staticmethod
    def _segment_aabb_xy(segment, x_idx: int, y_idx: int, *, warn: bool = False):
        """
        Compute a sound 2D axis-aligned bounding box (AABB) enclosure of the segment's
        projection onto (x_idx, y_idx).

        Preference order:
          1) Use tmv.bound() if available (encloses the flowpipe over the whole segment).
          2) Fall back to affine enclosure from (c_l, A_l) over local box [-r,r].
        """
        tmv = segment.get("tmv", None)
        if tmv is not None:
            bbox = tmv.bound()
            if x_idx < 0 or y_idx < 0 or x_idx >= len(bbox) or y_idx >= len(bbox):
                raise IndexError(f"dims {(x_idx, y_idx)} out of bounds for tmv.bound() len={len(bbox)}")
            bx = bbox[x_idx]
            by = bbox[y_idx]
            return float(bx.lower), float(bx.upper), float(by.lower), float(by.upper)

        A = segment.get("A_l", None)
        c = segment.get("c_l", segment.get("center", segment.get("global_center", None)))
        if A is None or c is None:
            return None

        A = np.asarray(A, dtype=float)
        c = np.asarray(c, dtype=float).reshape(-1)

        global_dim = c.shape[0]
        if A.ndim != 2 or A.shape[0] != global_dim:
            if warn:
                print(f"Visualizer: cannot compute affine AABB (bad A/c shapes) A={getattr(A,'shape',None)} len(c)={global_dim}")
            return None
        if x_idx < 0 or y_idx < 0 or x_idx >= global_dim or y_idx >= global_dim:
            return None

        local_dim = A.shape[1]
        local_half_widths = segment.get("local_half_widths", segment.get("r_l", None))
        if local_half_widths is None:
            r = np.ones(local_dim, dtype=float)
        else:
            r = np.asarray(local_half_widths, dtype=float).reshape(-1)
            if r.shape[0] != local_dim:
                if warn:
                    print(
                        f"Visualizer: local_half_widths length mismatch (got {r.shape[0]}, expected {local_dim}); using ones."
                    )
                r = np.ones(local_dim, dtype=float)

        cx = float(c[x_idx])
        cy = float(c[y_idx])
        rx = float(np.sum(np.abs(A[x_idx, :]) * r))
        ry = float(np.sum(np.abs(A[y_idx, :]) * r))
        return cx - rx, cx + rx, cy - ry, cy + ry

    @staticmethod
    def _segment_aabb_y(segment, y_idx: int, *, warn: bool = False):
        """1D AABB enclosure for a single coordinate y_idx (prefers tmv.bound())."""
        tmv = segment.get("tmv", None)
        if tmv is not None:
            bbox = tmv.bound()
            if y_idx < 0 or y_idx >= len(bbox):
                raise IndexError(f"y_idx {y_idx} out of bounds for tmv.bound() len={len(bbox)}")
            by = bbox[y_idx]
            return float(by.lower), float(by.upper)

        A = segment.get("A_l", None)
        c = segment.get("c_l", segment.get("center", segment.get("global_center", None)))
        if A is None or c is None:
            return None

        A = np.asarray(A, dtype=float)
        c = np.asarray(c, dtype=float).reshape(-1)
        global_dim = c.shape[0]
        if A.ndim != 2 or A.shape[0] != global_dim or y_idx < 0 or y_idx >= global_dim:
            return None

        local_dim = A.shape[1]
        local_half_widths = segment.get("local_half_widths", segment.get("r_l", None))
        if local_half_widths is None:
            r = np.ones(local_dim, dtype=float)
        else:
            r = np.asarray(local_half_widths, dtype=float).reshape(-1)
            if r.shape[0] != local_dim:
                if warn:
                    print(
                        f"Visualizer: local_half_widths length mismatch (got {r.shape[0]}, expected {local_dim}); using ones."
                    )
                r = np.ones(local_dim, dtype=float)

        cy = float(c[y_idx])
        ry = float(np.sum(np.abs(A[y_idx, :]) * r))
        return cy - ry, cy + ry

    @staticmethod
    def _draw_nonlinear_geometry(ax, result, dims, **kwargs):
        samples = kwargs.get("samples", 5)
        x_idx, y_idx = dims
        
        x_name = str(result.state_vars[x_idx]) if hasattr(result, 'state_vars') else f'x{x_idx}'
        y_name = str(result.state_vars[y_idx]) if hasattr(result, 'state_vars') else f'y{y_idx}'
        
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)

        colors = itertools.cycle(['magenta', 'purple', 'blue'])
        color = next(colors)

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        for segment in result.flowpipe:
            tmv = segment.get('tmv')
            if not tmv:
                continue
                
            full_domain = tmv.tms[0].domain
            
            dim = len(full_domain) - 1
            
            t_domain = full_domain[-1]
            t_start, t_end = float(t_domain.lower), float(t_domain.upper)
            
            t_steps = np.linspace(t_start, t_end, samples)
            
            corners_norm = list(itertools.product([-1, 1], repeat=dim))
            
            for t_val in t_steps:
                poly_points_x = []
                poly_points_y = []
                
                for corner in corners_norm:
                    eval_point = list(corner) + [t_val]
                    
                    val_x_res = tmv.tms[x_idx].poly.evaluate(tuple(eval_point))
                    val_y_res = tmv.tms[y_idx].poly.evaluate(tuple(eval_point))
                    
                    if hasattr(val_x_res, 'midpoint'):
                        val_x = float(val_x_res.midpoint())
                    else:
                        val_x = float(val_x_res)

                    if hasattr(val_y_res, 'midpoint'):
                        val_y = float(val_y_res.midpoint())
                    else:
                        val_y = float(val_y_res)
                    
                    poly_points_x.append(val_x)
                    poly_points_y.append(val_y)
                    
                    min_x = min(min_x, val_x)
                    max_x = max(max_x, val_x)
                    min_y = min(min_y, val_y)
                    max_y = max(max_y, val_y)

                if len(poly_points_x) >= 3:
                    # Ordering for 2D Quad: (-1,-1), (-1,1), (1,1), (1,-1)
                    px = [poly_points_x[0], poly_points_x[1], poly_points_x[3], poly_points_x[2]]
                    py = [poly_points_y[0], poly_points_y[1], poly_points_y[3], poly_points_y[2]]
                    
                    poly_xy = list(zip(px, py))
                    poly = patches.Polygon(
                        poly_xy, 
                        closed=True,
                        linewidth=0.5, 
                        edgecolor=color, 
                        facecolor='none', 
                        alpha=0.5
                    )
                    ax.add_patch(poly)
        
        if min_x == float('inf'):
            min_x, max_x = 0, 1
        if min_y == float('inf'):
            min_y, max_y = 0, 1

        pad_x = (max_x - min_x) * 0.1 if max_x != min_x else 0.1
        pad_y = (max_y - min_y) * 0.1 if max_y != min_y else 0.1
        
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(min_y - pad_y, max_y + pad_y)
        
        ax.set_title("Nonlinear Flowpipe (Left-Right Composition)")
        ax.grid(True)

    @staticmethod
    def _draw_hybrid_geometry(ax, result, dims=(0, 1), **kwargs):
        x_idx, y_idx = dims
        filled = kwargs.get("filled", True)
        use_tmv_bounds = kwargs.get("use_tmv_bounds", True)
        legend_fontsize = kwargs.get("legend_fontsize", 12)
        
        mode_colors = kwargs.get("mode_colors")
        if mode_colors is None:
            unique_modes = sorted(list(set(seg['mode'] for seg in result.flowpipe)))
            cmap = plt.get_cmap('tab10')
            mode_colors = {mode_id: cmap(i % 10) for i, mode_id in enumerate(unique_modes)}

        corners_unit = [np.array([-1, -1]), np.array([1, -1]), np.array([1, 1]), np.array([-1, 1])]

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        for segment in result.flowpipe:
            if not segment.get('is_valid', True): continue

            mode_name = segment['mode']
            tmv = segment['tmv']
            A = segment.get('A_l')
            c = segment.get('center')
            color = mode_colors.get(mode_name, 'magenta')
            facecolor = color if filled else 'none'

            if use_tmv_bounds and tmv is not None:
                try:
                    bbox = tmv.bound()
                    bx, by = bbox[x_idx], bbox[y_idx]
                    rect = patches.Rectangle(
                        (float(bx.lower), float(by.lower)),
                        float(bx.upper - bx.lower),
                        float(by.upper - by.lower),
                        linewidth=0.6,
                        edgecolor=color,
                        facecolor=facecolor,
                        alpha=0.3,
                    )
                    ax.add_patch(rect)
                    min_x, max_x = min(min_x, float(bx.lower)), max(max_x, float(bx.upper))
                    min_y, max_y = min(min_y, float(by.lower)), max(max_y, float(by.upper))
                    continue
                except Exception:
                    pass

            if A is not None and c is not None:
                corners_global = []
                for pt in corners_unit:
                    vec_local = np.zeros(len(c))
                    vec_local[x_idx], vec_local[y_idx] = pt[0], pt[1]
                    vec_global = (A @ vec_local) + c
                    corners_global.append([float(vec_global[x_idx]), float(vec_global[y_idx])])

                poly = patches.Polygon(corners_global, closed=True, linewidth=0.6, 
                                       edgecolor=color, facecolor=facecolor, alpha=0.5)
                ax.add_patch(poly)
                
                xs, ys = zip(*corners_global)
                min_x, max_x = min(min_x, *xs), max(max_x, *xs)
                min_y, max_y = min(min_y, *ys), max(max_y, *ys)
            else:
                bbox = tmv.bound()
                bx, by = bbox[x_idx], bbox[y_idx]
                rect = patches.Rectangle((bx.lower, by.lower), bx.upper-bx.lower, by.upper-by.lower,
                                         linewidth=0.6, edgecolor=color, facecolor=facecolor, alpha=0.3)
                ax.add_patch(rect)
                min_x, max_x = min(min_x, bx.lower), max(max_x, bx.upper)
                min_y, max_y = min(min_y, by.lower), max(max_y, by.upper)

        if min_x != float('inf'):
            ax.set_xlim(min_x - abs(min_x)*0.1, max_x + abs(max_x)*0.1)
            ax.set_ylim(min_y - abs(min_y)*0.1, max_y + abs(max_y)*0.1)

        handles = [patches.Patch(color=c, label=f"Mode: {m}") for m, c in mode_colors.items()]
        ax.legend(handles=handles, loc='upper right', fontsize=legend_fontsize)

    @staticmethod
    def _draw_stochastic_geometry(ax, result, dims, **kwargs):
        """render deterministic core + stochastic radius expansion"""
        px, py = dims
        traces = result.validation_data.get('traces')
        legend_fontsize = kwargs.get("legend_fontsize", 12)

        plotter = StochasticPlotter(result.flowpipe, result.state_vars)

        L_mat = kwargs.get('L_matrix')
        if L_mat is None:
            L_mat = result.validation_data.get('L_matrix')
        ep = result.config.get('engine_params', {})
        P_mat = result.validation_data.get('P_matrix', ep.get('P_matrix'))
        diag_p_inv = result.validation_data.get('diag_P_inv_upper')

        title = kwargs.get('title', f"{result.config.get('name', 'Stochastic')} Reachable Set")
        plotter.plot_2d_projection(
            px, py,
            traces=traces,
            L_matrix=L_mat,
            P_matrix=P_mat,
            diag_P_inv_upper=diag_p_inv,
            title=title,
            ax=ax,
            legend_fontsize=legend_fontsize,
        )
        plotter.plot_deviation_history(px, py, traces=traces, legend_fontsize=legend_fontsize)

    @staticmethod
    def _render_time_evolution(ax, result, y_idx, **kwargs):
        if result.system_type == 'hybrid':
            Visualizer._render_hybrid_time_evolution(ax, result, y_idx=y_idx, **kwargs)
        elif result.system_type == 'continuous':
            Visualizer._render_continuous_time_evolution(ax, result, y_idx=y_idx, **kwargs)
        else:
            raise ValueError("Time evolution plots are only available for continuous and hybrid systems.")
    @staticmethod
    def _render_hybrid_time_evolution(ax, result, y_idx=0, **kwargs):
        filled = kwargs.get("filled", True)
        mode_colors = kwargs.get("mode_colors")
        legend_fontsize = kwargs.get("legend_fontsize", 12)
        
        if mode_colors is None:
            unique_modes = sorted(list(set(seg.get('mode', 'default') for seg in result.flowpipe)))
            cmap = plt.get_cmap('tab10')
            mode_colors = {mode_id: cmap(i % 10) for i, mode_id in enumerate(unique_modes)}

        max_t, min_y, max_y = 0, float('inf'), float('-inf')
        seen_modes = set()

        for segment in result.flowpipe:
            mode_name = segment.get('mode', 'default')
            t_abs = segment['time_interval_abs']
            t_start = float(t_abs.lower)
            t_end = float(t_abs.upper)
            bbox = segment['tmv'].bound()
            y_b = bbox[y_idx]
            
            color = mode_colors.get(mode_name, 'magenta')
            rect = patches.Rectangle(
                (t_start, float(y_b.lower)), 
                t_end - t_start, 
                float(y_b.upper - y_b.lower),
                linewidth=0.5, edgecolor=color, facecolor=color if filled else 'none', alpha=0.5,
                label=f"Mode: {mode_name}" if mode_name not in seen_modes else None
            )
            ax.add_patch(rect)
            seen_modes.add(mode_name)
            max_t = max(max_t, float(t_abs.upper))
            min_y, max_y = min(min_y, float(y_b.lower)), max(max_y, float(y_b.upper))

        ax.set_xlim(0, max_t * 1.1)
        ax.set_ylim(min_y - abs(min_y)*0.1, max_y + abs(max_y)*0.1)
        ax.legend(loc='upper right', fontsize=legend_fontsize)

    
    @staticmethod
    def _render_continuous_time_evolution(ax, result, y_idx=0, **kwargs):

        color = kwargs.get("color", "royalblue")
        alpha = kwargs.get("alpha", 0.5)
        filled = kwargs.get("filled", True)
        warn = kwargs.get("warn", False)
        
        min_y, max_y = float('inf'), float('-inf')
        total_time = 0

        for seg_idx, segment in enumerate(result.flowpipe):
            t_abs = segment.get('time_interval_abs')
            if t_abs is None:
                continue
            t_start = float(t_abs.lower)
            t_end = float(t_abs.upper)
            duration = t_end - t_start
            if duration <= 0:
                continue

            try:
                y_aabb = Visualizer._segment_aabb_y(segment, y_idx, warn=warn)
            except Exception as exc:
                if warn:
                    print(f"Visualizer: skip time seg {seg_idx} (y AABB failed): {exc}")
                continue
            if y_aabb is None:
                continue
            y_min, y_max = y_aabb

            rect = patches.Rectangle(
                (t_start, y_min), 
                duration, 
                y_max - y_min,
                linewidth=0.5, 
                edgecolor=color, 
                facecolor=color if filled else 'none', 
                alpha=alpha
            )
            ax.add_patch(rect)

            total_time = max(total_time, t_end)
            min_y, max_y = min(min_y, y_min), max(max_y, y_max)

        ax.set_xlim(0, total_time * 1.05)
        ax.set_ylim(min_y - abs(min_y)*0.1, max_y + abs(max_y)*0.1)
        plt.show()
