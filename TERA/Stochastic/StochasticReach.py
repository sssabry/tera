"""Stochastic reachability engine built on AMGF bounds."""

import numpy as np
import warnings
import math
from scipy.optimize import minimize_scalar
from sage.all import matrix, SR, fast_callable, RIF, exp, log, sqrt

from TERA.TMFlow.TMReach import TMReach

class StochasticTMReach(TMReach):
    """Extend TMReach with stochastic AMGF bounds."""
    def __init__(self, delta: float, g_exprs: list, P_matrix=None, amgf_eps: float = None, *args, **kwargs):
        """Initialize stochastic reachability parameters."""
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.amgf_eps = amgf_eps

        # 1. compile jacobian of deterministic dynamics f(x)
        t_sym = SR(self.time_var)

        # 2. compile diffusion matrix g(x)
        # TODO: require nested list n x m
        if not isinstance(g_exprs[0], list):
            # legacy: treat 1D list as diagonal if size matches state dim.
            if len(g_exprs) == self.state_dim:
                warnings.warn(
                    "g_exprs provided as 1D list; interpreting as diagonal (legacy behavior).",
                    RuntimeWarning
                )
                g_exprs = np.diag(g_exprs).tolist()
        
        self.g_sym = matrix(SR, g_exprs)
        vars_in_g = set()
        for x in self.g_sym.list():
            vars_in_g.update(x.variables())
        
        self.g_is_constant = len(vars_in_g) == 0
        
        if self.g_is_constant:
            # constant? pre-compute G bounds once & evaluate at 0
            zero_int = [RIF(0)] * (len(self.state_vars) + 1)
            self.g_interval_fn = self._compile_interval_fn(self.g_sym, list(self.state_vars) + [t_sym])
            self.G_const_int = self.g_interval_fn(zero_int)
        else:
            self.g_interval_fn = self._compile_interval_fn(self.g_sym, list(self.state_vars) + [t_sym])

        # 2a. handle weighted norm matrix P if provided
        self.use_weighted = False
        self.L_sage = None
        self.L_inv_sage = None
        self.L_np = None  # plotting only (non-rigorous)
        if P_matrix is not None:
            # build symmetric P over RIF
            P_rif = matrix(RIF, [[RIF(v) for v in row] for row in P_matrix])

            if P_rif.nrows() != self.state_dim or P_rif.ncols() != self.state_dim:
                raise ValueError(f"P_matrix must be {self.state_dim}x{self.state_dim}")
            if P_rif != P_rif.transpose():
                raise ValueError("P_matrix must be symmetric.")

            # rigorous cholesky: P = L * L^T
            L = self._rif_cholesky_lower(P_rif)
            Linv = self._rif_lower_tri_inverse(L)

            self.use_weighted = True
            self.L_sage = L
            self.L_inv_sage = Linv

            # for plotting only:
            n = L.nrows()
            self.L_np = np.array([[float(L[i, j].center()) for j in range(n)] for i in range(n)], dtype=float)

        # now that L is known, cache sigma_sq for constant G if possible
        if self.g_is_constant:
            if self.use_weighted:
                self.sigma_sq_const = self._sigma_sq_bound(self.L_sage * self.G_const_int)
            else:
                self.sigma_sq_const = self._sigma_sq_bound(self.G_const_int)

        # 3. optimize AMGF constants (theorem 1)
        # use provided eps or optimize
        if self.amgf_eps is not None:
            e = float(self.amgf_eps)
            if not (0.0 < e < 1.0):
                raise ValueError(f"amgf_eps must be in (0,1), got {e}")
            self.eps1 = (2.0 * np.log(1.0 + 2.0 / e)) / ((1.0 - e) ** 2)
            self.eps2 = 2.0 / ((1.0 - e) ** 2)
        else:
            self.eps1, self.eps2 = self._optimize_epsilon(self.state_dim, delta)

        # convert constants to RIF for rigorous arithmetic
        self.eps1_rif = RIF(self.eps1)
        self.eps2_rif = RIF(self.eps2)
        self.log_term_rif = log(RIF(1.0) / RIF(self.delta))
        
        # 4. initialize integration state
        self.psi_t = RIF(0.0) # integral of c_t (contraction)
        self.Psi_t = RIF(0.0) # accumulated energy
        self.stochastic_records = []

    def _rif_cholesky_lower(self, P):
        """
        Rigorous Cholesky factorization in RIF: P = L * L^T
        P: Sage matrix over RIF (assumed symmetric, SPD)
        Returns L: lower-triangular Sage matrix over RIF.
        """
        n = P.nrows()
        L = matrix(RIF, n, n, [RIF(0)] * (n * n))

        for i in range(n):
            for j in range(i + 1):
                s = RIF(0)
                for k in range(j):
                    s += L[i, k] * L[j, k]

                if i == j:
                    val = P[i, i] - s
                    # require strictly positive lower bound
                    if val.lower() <= 0:
                        raise ValueError(f"P_matrix not provably SPD at pivot {i}: {val}")
                    L[i, j] = sqrt(val)
                else:
                    if L[j, j].lower() <= 0:
                        raise ValueError(f"Cholesky failed: nonpositive diagonal at {j}")
                    L[i, j] = (P[i, j] - s) / L[j, j]
        return L

    def _rif_lower_tri_inverse(self, L):
        """
        Rigorous inverse of lower-triangular matrix L in RIF.
        Computes Linv such that Linv encloses the exact L^{-1}.
        """
        n = L.nrows()
        Linv = matrix(RIF, n, n, [RIF(0)] * (n * n))

        for i in range(n):
            if L[i, i].lower() <= 0:
                raise ValueError(f"Cannot invert: diagonal not provably nonzero at {i}: {L[i, i]}")
            Linv[i, i] = RIF(1) / L[i, i]

            for j in range(i):  # below diagonal entries in inverse
                s = RIF(0)
                for k in range(j, i):
                    s += L[i, k] * Linv[k, j]
                Linv[i, j] = -s / L[i, i]

        return Linv
    def _compile_interval_fn(self, sym_matrix, vars):
        """helper to compile a symbolic matrix into a fast function
        that takes interval inputs and returns a Sage matrix over RIF"""
        flat_exprs = sym_matrix.list()
        
        # fast_callable with domain=RIF
        compiled = [fast_callable(ex, vars=vars, domain=RIF) for ex in flat_exprs]
        rows, cols = sym_matrix.nrows(), sym_matrix.ncols()
        
        def evaluator(interval_inputs):
            # interval_inputs: list of RIF objects [x1_int, x2_int, ..., t_int]
            vals = [f(*interval_inputs) for f in compiled]
            # construct a Sage matrix over RIF
            return matrix(RIF, rows, cols, vals)
        
        return evaluator


    def _optimize_epsilon(self, n: int, delta: float):
        """
        the stochastic bound relies on two constant epsilon values eps_1, eps_2
        from the range (0,1). they need to be chosen to minimize the bound for a specific
        dimension n and probability delta

        the TAC paper proposes Lemma 5.1 on how to minimize the scaling factor:
        (equation 21 from TAC paper)
        - eps_1 = (2 log(1+2/e))/(1-e)^2
        - eps_2 = 2/(1-e)^2
        """

        # cost function we want to minimize
        def cost_function(eps):
            # if out of range: cost = infinity
            if eps <= 0 or eps >= 1: return np.inf

            eps_1 = (2 * np.log(1+2/eps)) / ((1-eps)**2)
            eps_2 = 2 / ((1-eps)**2)

            # inner term of equationo 20
            return eps_1 * n + eps_2 * np.log(1/delta)
        
        # use scipy bounded scalar minimization
        res = minimize_scalar(cost_function, bounds=(1e-6, 1-1e-6), method='bounded')
        optimal_eps = res.x
        
        # calculate final constants
        opt_eps_1 = (2 * np.log(1 + 2/optimal_eps)) / ((1 - optimal_eps)**2)
        opt_eps_2 = 2 / ((1 - optimal_eps)**2)
        
        return opt_eps_1, opt_eps_2
    
    def _compute_inf_norm_bound(self, interval_matrix):
        """optimized calcualtion of upper bound for infinity norm M_2^2
        M = A * A.T, returns bound on max eigenvalue (rigorous via RIF)"""
        # 1. compute M = A * A^T
        M_int = interval_matrix * interval_matrix.transpose()

        # 2. compute row sums of upper bounds
        rows = M_int.nrows()
        cols = M_int.ncols()
        max_row_sum = RIF(0.0)
        for i in range(rows):
            row_sum = RIF(0.0)
            for j in range(cols):
                row_sum += RIF(M_int[i, j].abs().upper())
            if row_sum.upper() > max_row_sum.upper():
                max_row_sum = row_sum

        # 3. infinity norm (max row sum) bounds the spectral radius for symmetric matrices
        return max_row_sum
    
    def _compute_contraction_bound(self, jacobian_matrix):
        """
        optimized Gershgorin circle calculation for c_t (rigorous via RIF)
        """
        # J_sym = 0.5 * (J + J.T)
        J_curr = jacobian_matrix
        J_transpose = J_curr.transpose()
        sym_part = (J_curr + J_transpose) * RIF(0.5)
        
        rows = sym_part.nrows()

        # tight 2x2 symmetric eigenvalue bound via closed-form lambda_max
        if rows == 2:
            a = sym_part[0, 0]
            b = sym_part[0, 1]
            d = sym_part[1, 1]
            tr = a + d
            disc = (a - d) ** 2 + RIF(4.0) * (b ** 2)
            lam_max = (tr + sqrt(disc)) * RIF(0.5)
            return RIF(lam_max.upper())

        # gershgorin radii (rigorous upper bounds)
        max_radius = RIF(0.0)
        for i in range(rows):
            center = RIF(sym_part[i, i].upper())
            radius = RIF(0.0)
            for j in range(rows):
                if i == j:
                    continue
                radius += RIF(sym_part[i, j].abs().upper())
            g_bound = center + radius
            if g_bound.upper() > max_radius.upper():
                max_radius = g_bound

        return max_radius

    def _sigma_sq_bound(self, G_curr):
        """Rigorous upper bound on ||G||_2^2 using RIF arithmetic."""
        rows = G_curr.nrows()
        cols = G_curr.ncols()

        # single wiener (m=1): exact ||g||_2^2 = sum_i g_i^2
        if cols == 1:
            s = RIF(0.0)
            for i in range(rows):
                s += G_curr[i, 0] ** 2
            return RIF(s.upper())

        # fallback: conservative infinity-norm bound on GG^T
        return self._compute_inf_norm_bound(G_curr)
    
    def _post_step_hook(self, step_info: dict, h: float):
        """
        intercepts the deterministic reachability engine's step to calculate the AMGF params

        computes parameters in the transformed (preconditioned) coordinate system 
        so that the standard AMGF bound (not the weighted one)
        """
        # 1. retrieve jacobian & convert into sage matrix
        J_list = step_info['jacobian']
        rows = len(J_list)
        cols = len(J_list[0])
        rif_vals = [J_list[r][c]._interval for r in range(rows) for c in range(cols)]
        J_int = matrix(RIF, rows, cols, rif_vals)

        # 2. retrieve/compute Diffusion G
        if self.g_is_constant:
            G_int = self.G_const_int
        else:
            # not constant? must re-evaluate
            tmv = step_info['tmv']
            rif_inputs = [b._interval for b in tmv.bound()]
            
            # handle time variable for G evaluation
            t_interval = step_info['time_interval_abs']
            rif_inputs.append(t_interval._interval)
            
            G_int = self.g_interval_fn(rif_inputs)

        # 3. apply weighted metric if enabled
        if self.use_weighted:
            # transform into P-space: J_hat = L J L^{-1}, G_hat = L G
            J_curr = self.L_sage * J_int * self.L_inv_sage
            G_curr = self.L_sage * G_int
        else:
            J_curr = J_int
            G_curr = G_int

        # 4. compute c_t 
        c_t = self._compute_contraction_bound(J_curr)

        # 5. compute sigma_sq 
        if self.g_is_constant:
            # constant? fully cached
            sigma_sq_t = self.sigma_sq_const
        else:
            sigma_sq_t = self._sigma_sq_bound(G_curr)
        
        # 6. integrate AMGF energy (Psi): psi_t = int(c_t)
        h_rif = RIF(h)
        psi_prev = self.psi_t
        self.psi_t += c_t * h_rif

        # pre calculate exp term for this step
        # K = sigma_sq_t * exp(-2 * psi_prev)
        K = sigma_sq_t * exp(-RIF(2.0) * psi_prev)
        
        # analytical integral of integrand exp(-2 * c_t * tau) from 0 to h
        # if c_t near zeroL avoid division by zero using limit which is just h
        # if 0 in c_t, using integral_factor = h is a sound upper bound.
        if hasattr(c_t, "contains_zero"):
            zero_in = c_t.contains_zero()
        else:
            try:
                zero_in = c_t.contains(0)
            except Exception:
                zero_in = (c_t.lower() <= 0 <= c_t.upper())

        if zero_in:
            integral_factor = h_rif
        else:
            # exact integral: (1 - exp(-2 * c_t * h)) / (2 * c_t)
            integral_factor = (RIF(1.0) - exp(-RIF(2.0) * c_t * h_rif)) / (RIF(2.0) * c_t)

        dPsi = K * integral_factor
        self.Psi_t += dPsi
        
        # 7. compute final radius
        # equation (25): r = sqrt( Psi * exp(2*psi) * (terms) )
        # paper defines Psi_t = int(sigma^2 * exp(-2psi)), 
        # so bound is sqrt( Psi_t * exp(2*psi_t) * scaling )
        scaling = (self.eps1_rif * RIF(self.state_dim)) + (self.eps2_rif * self.log_term_rif)
        
        # exp(2*psi_t) term brings the energy back to current time scale
        r_squared = (self.Psi_t * exp(RIF(2.0) * self.psi_t)) * scaling

        # clamp tiny negative enclosure to 0 before sqrt
        lower = r_squared.lower()
        upper = r_squared.upper()
        if upper < 0:
            upper = 0.0
        if lower < 0:
            lower = 0.0
        r_squared = RIF(lower, upper)
        r_delta_t = sqrt(r_squared)
        
        # 8. save to step_info so visualization can see it
        r_upper = float(r_delta_t.upper())
        r_upper = math.nextafter(r_upper, math.inf)
        step_info['stochastic_radius'] = r_upper
        step_info['stochastic_radius_rif'] = r_delta_t
        self.stochastic_records.append({
            't': step_info['time_interval_abs'].upper,
            'r': r_delta_t,
            'c_t': c_t,
            'sigma_sq': sigma_sq_t
        })
