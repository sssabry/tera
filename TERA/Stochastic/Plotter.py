"""Plotting utilities for stochastic reachability results."""
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import affinity
from matplotlib.patches import Polygon as MplPolygon
from sage.all import RIF, matrix, sqrt


class StochasticPlotter:
    """Render stochastic reachability projections."""
    def __init__(self, flowpipe, var_names):
        """Initialize a plotter for a flowpipe."""
        self.flowpipe = flowpipe
        self.var_names = [str(v) for v in var_names]
        
    def _tm_to_box_polygon(self, tmv, x_idx, y_idx):
        """Return an AABB polygon for a TMV projection."""
        b = tmv.bound()
        ix = b[x_idx]
        iy = b[y_idx]

        x_lo, x_hi = float(ix.lower), float(ix.upper)
        y_lo, y_hi = float(iy.lower), float(iy.upper)

        # rectangle corners (closed ring)
        return Polygon([(x_lo, y_lo), (x_hi, y_lo), (x_hi, y_hi), (x_lo, y_hi)])
    
    def _apply_weighted_buffer(self, poly_det, r_stoch, L_matrix=None):
        """Apply a weighted buffer in transformed space."""
        if L_matrix is None:
            # Standard Euclidean case: buffer is a circle
            return poly_det.buffer(r_stoch, quad_segs=64)
        
        # 1. Extract 2D transformation for the plotted indices
        # Shapely affine: [a, b, d, e, xoff, yoff] for matrix [[a, b], [d, e]]
        a, b = L_matrix[0, 0], L_matrix[0, 1]
        d, e = L_matrix[1, 0], L_matrix[1, 1]
        
        # 2. Transform Polygon to P-space (where the bound is a circle)
        poly_P = affinity.affine_transform(poly_det, [a, b, d, e, 0, 0])
        
        # 3. Buffer with circle of radius r_stoch in P-space
        poly_P_buffered = poly_P.buffer(r_stoch, quad_segs=64)
        
        # 4. Transform back to original state space: x = L_inv * x_hat
        L_inv = np.linalg.inv(L_matrix)
        ai, bi = L_inv[0, 0], L_inv[0, 1]
        di, ei = L_inv[1, 0], L_inv[1, 1]

        return affinity.affine_transform(poly_P_buffered, [ai, bi, di, ei, 0, 0])

    def _rif_cholesky_lower(self, P):
        """Rigorous Cholesky factorization in RIF: P = L * L^T."""
        n = P.nrows()
        L = matrix(RIF, n, n, [RIF(0)] * (n * n))
        for i in range(n):
            for j in range(i + 1):
                s = RIF(0)
                for k in range(j):
                    s += L[i, k] * L[j, k]
                if i == j:
                    val = P[i, i] - s
                    if val.lower() <= 0:
                        raise ValueError(f"P not provably SPD at pivot {i}: {val}")
                    L[i, j] = sqrt(val)
                else:
                    if L[j, j].lower() <= 0:
                        raise ValueError(f"Cholesky failed: nonpositive diagonal at {j}")
                    L[i, j] = (P[i, j] - s) / L[j, j]
        return L

    def _rif_lower_solve(self, L, B):
        """Solve L X = B for X using forward substitution (RIF)."""
        n = L.nrows()
        m = B.ncols()
        X = matrix(RIF, n, m, [RIF(0)] * (n * m))
        for i in range(n):
            for j in range(m):
                s = RIF(0)
                for k in range(i):
                    s += L[i, k] * X[k, j]
                if L[i, i].lower() == 0:
                    raise ValueError("Lower solve failed: zero diagonal.")
                X[i, j] = (B[i, j] - s) / L[i, i]
        return X

    def _rif_upper_solve(self, U, B):
        """Solve U X = B for X using backward substitution (RIF)."""
        n = U.nrows()
        m = B.ncols()
        X = matrix(RIF, n, m, [RIF(0)] * (n * m))
        for i in reversed(range(n)):
            for j in range(m):
                s = RIF(0)
                for k in range(i + 1, n):
                    s += U[i, k] * X[k, j]
                if U[i, i].lower() == 0:
                    raise ValueError("Upper solve failed: zero diagonal.")
                X[i, j] = (B[i, j] - s) / U[i, i]
        return X

    def _projected_metric(self, P_rif, idx_pair):
        """Compute 2D projected metric via Schur complement in RIF."""
        n = P_rif.nrows()
        i0, i1 = idx_pair
        I = [i0, i1]
        J = [k for k in range(n) if k not in I]

        P_II = matrix(RIF, 2, 2, [P_rif[i, j] for i in I for j in I])
        if not J:
            return P_II

        m = len(J)
        P_IJ = matrix(RIF, 2, m, [P_rif[i, j] for i in I for j in J])
        P_JI = matrix(RIF, m, 2, [P_rif[i, j] for i in J for j in I])
        P_JJ = matrix(RIF, m, m, [P_rif[i, j] for i in J for j in J])

        L = self._rif_cholesky_lower(P_JJ)
        Y = self._rif_lower_solve(L, P_JI)
        X = self._rif_upper_solve(L.transpose(), Y)
        return P_II - (P_IJ * X)

    def plot_2d_projection(self, x_idx, y_idx, traces=None, L_matrix=None, P_matrix=None,
                           diag_P_inv_upper=None, show_axis_bound_overlay=False,
                           title="Stochastic Reachability", ax=None,
                           mc_downsample=5, mc_alpha=0.35, mc_lw=0.6, show_deterministic_center=True, center_lw=1.5,
                           envelope_alpha=0.4, envelope_label="Probabilistic Set", core_label=None,
                           legend_fontsize=12,):
        """Plot a 2D stochastic projection with optional traces."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        green_shapes = []
        core_shapes = []
        axis_shapes = []

        # Determine weighted metric for plotting
        L_plot = L_matrix
        if P_matrix is not None:
            P_rif = matrix(RIF, [[RIF(v) for v in row] for row in P_matrix])
            if P_rif.nrows() >= 2:
                P_proj = self._projected_metric(P_rif, (x_idx, y_idx))
                P_mid = np.array(
                    [[0.5 * (float(P_proj[i, j].lower()) + float(P_proj[i, j].upper())) for j in range(2)] for i in range(2)],
                    dtype=float,
                )
                try:
                    L_plot = np.linalg.cholesky(P_mid)
                except np.linalg.LinAlgError:
                    L_plot = L_matrix
        
        print(f"[Plotter] Generating {'Weighted' if L_plot is not None else 'Euclidean'} Minkowski Envelopes...")
        for segment in self.flowpipe:
            tmv = segment['tmv']
            r_stoch = segment['stochastic_radius']
            
            # deterministic overapprox
            poly_core = self._tm_to_box_polygon(tmv, x_idx, y_idx)
            if not poly_core.is_valid:
                poly_core = poly_core.buffer(0).convex_hull
            core_shapes.append(poly_core)

            
            # minkowski inflation by r(t) in either Euclidean or weighted metric
            poly_stoch = self._apply_weighted_buffer(poly_core, r_stoch, L_plot)
            if not poly_stoch.is_valid:
                poly_stoch = poly_stoch.buffer(0).convex_hull
            green_shapes.append(poly_stoch)

            # optional axis-aligned bound overlay (diagnostic)
            if show_axis_bound_overlay:
                b = tmv.bound()
                x_lo, x_hi = float(b[x_idx].lower), float(b[x_idx].upper)
                y_lo, y_hi = float(b[y_idx].lower), float(b[y_idx].upper)
                if diag_P_inv_upper is not None and x_idx < len(diag_P_inv_upper) and y_idx < len(diag_P_inv_upper):
                    dx = r_stoch * np.sqrt(diag_P_inv_upper[x_idx])
                    dy = r_stoch * np.sqrt(diag_P_inv_upper[y_idx])
                else:
                    dx = r_stoch
                    dy = r_stoch
                poly_axis = Polygon([(x_lo - dx, y_lo - dy), (x_hi + dx, y_lo - dy),
                                     (x_hi + dx, y_hi + dy), (x_lo - dx, y_hi + dy)])
                axis_shapes.append(poly_axis)
            
        # merge all steps into one continuous tube (unary_union)
        full_green_envelope = unary_union(green_shapes)
        full_core = unary_union(core_shapes) if core_label is not None else None
        full_axis = unary_union(axis_shapes) if axis_shapes else None
        
        # plot Green Envelope
        self._plot_shapely(ax, full_green_envelope, color='#77dd77', alpha=envelope_alpha, label=envelope_label)

        # optional determinstic core plot:
        if core_label is not None and full_core is not None:
            self._plot_shapely(ax, full_core, color="#4C72B0", alpha=0.25, label=core_label)
        if full_axis is not None:
            self._plot_shapely(ax, full_axis, color="#B0B0B0", alpha=0.18, label="Axis Bound (diag)")

        # plot MC traces
        Xdet = None
        if traces is not None:
            if not isinstance(traces, (tuple, list)) or len(traces) < 2:
                raise ValueError("plot_2d_projection: traces must be (t_grid, Xstoch[, Xdet]).")

            t_grid = np.asarray(traces[0], dtype=float)
            Xstoch = np.asarray(traces[1], dtype=float)
            if Xstoch.ndim != 3:
                raise ValueError(f"plot_2d_projection: Xstoch must be (N,T,D), got shape {Xstoch.shape}.")

            if len(traces) >= 3 and traces[2] is not None:
                Xdet = np.asarray(traces[2], dtype=float)
                if Xdet.ndim == 2:
                    # promote to (1,T,D) for uniform plotting
                    Xdet = Xdet[None, :, :]

            # downsample time for speed
            step = int(mc_downsample) if mc_downsample and mc_downsample > 1 else 1

            N = Xstoch.shape[0]
            # rainbow colormap (jet/rainbow)
            cmap = plt.get_cmap('jet')
            colors = [cmap(i) for i in np.linspace(0, 1, max(N, 1))]
            for k in range(N):
                ax.plot(
                    Xstoch[k, ::step, x_idx],
                    Xstoch[k, ::step, y_idx],
                    color=colors[k],
                    linewidth=mc_lw,
                    alpha=mc_alpha,
                )
                
            # add dummy line for legend
            ax.plot([], [], color=colors[0], linewidth=1.5, label='Monte Carlo Traces')

        ax.set_xlabel(self.var_names[x_idx])
        ax.set_ylabel(self.var_names[y_idx])
        ax.set_title(title)
        ax.legend(fontsize=legend_fontsize)
        ax.grid(True, alpha=0.3)
        #plt.show()

    def _plot_shapely(self, ax, geom, color, alpha, label):
        if geom.geom_type == 'Polygon':
            mpl_poly = MplPolygon(np.array(geom.exterior.coords), facecolor=color, alpha=alpha, label=label)
            ax.add_patch(mpl_poly)
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                mpl_poly = MplPolygon(np.array(poly.exterior.coords), facecolor=color, alpha=alpha, label=label)
                ax.add_patch(mpl_poly)
                label = None

    def plot_deviation_history(self, x_idx, y_idx, traces, title="Deviation Analysis", legend_fontsize=12):
        """
        reproduces fig 3b (stochastic deviation ||X_t - x_t|| over time vs bound)
        """
        # 1. extract theoretical bound (step-wise)
        t_bounds = []
        r_bounds = []

        for segment in self.flowpipe:
            t_end = float(segment['time_interval_abs'].upper)
            r = float(segment.get('stochastic_radius', 0.0))
            t_bounds.append(t_end)
            r_bounds.append(r)
            
        # 2. compute empirical deviation
        payload = traces
        if not isinstance(payload, (tuple, list)) or len(payload) < 2:
            raise ValueError("plot_deviation_history: invalid traces payload format.")

        t_grid = np.asarray(payload[0], dtype=float)
        Xstoch = np.asarray(payload[1], dtype=float)
        Xdet = np.asarray(payload[2], dtype=float) if len(payload) >= 3 and payload[2] is not None else None

        #max_k ||X_t^k - x_t^k||
        if Xdet is not None:
            # deviation in full state space (matches theorem)
            dists = np.linalg.norm(Xstoch - Xdet, axis=2)  # (N, T)
            max_devs = np.max(dists, axis=0)               # (T,)
            empirical_label = "Empirical Max Deviation: max_k ||X_t^k - x_t^k||"
        else:
            # fallback: deviation in chosen 2D plane to a nominal center
            centers = []
            for segment in self.flowpipe:
                t_end = float(segment['time_interval_abs'].upper)
                tmv = segment['tmv']
                h = t_end - float(segment['time_interval_abs'].lower)
                dom_len = len(tmv.tms[0].domain) - 1
                args = [0.0]*dom_len + [h]
                cx = float(tmv.tms[x_idx].poly.evaluate(tuple(args)))
                cy = float(tmv.tms[y_idx].poly.evaluate(tuple(args)))
                centers.append((t_end, cx, cy))

            center_arr = np.array(centers, dtype=float)  # [t, x, y]
            ref_x = np.interp(t_grid, center_arr[:, 0], center_arr[:, 1])
            ref_y = np.interp(t_grid, center_arr[:, 0], center_arr[:, 2])

            # deviation only in the (x_idx,y_idx) plane
            max_devs = []
            for i in range(len(t_grid)):
                dx = Xstoch[:, i, x_idx] - ref_x[i]
                dy = Xstoch[:, i, y_idx] - ref_y[i]
                max_devs.append(np.max(np.sqrt(dx*dx + dy*dy)))
            max_devs = np.asarray(max_devs, dtype=float)
            empirical_label = "Empirical Deviation (fallback): to nominal TM center slice"

        # plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # step plot for theoretical bound r(t)
        ax.step(t_bounds, r_bounds, where='post', linewidth=2, label='Theoretical Bound (AMGF)')

        # empirical max deviation curve
        ax.plot(t_grid, max_devs, 'r--', linewidth=1.5, label=empirical_label)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Deviation')
        ax.set_title(title)
        ax.legend(fontsize=legend_fontsize)
        ax.grid(True)
        #plt.show()
