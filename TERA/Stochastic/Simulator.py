"""Monte Carlo simulation utilities for stochastic validation."""
import numpy as np


class MonteCarloValidator:
    """Monte Carlo validator using Euler-Maruyama integration."""
    def __init__(self, f_exprs, g_exprs, vars, time_var='t'):
        """Initialize a validator for an SDE model."""
        self.dt_sim = 0.001
        self.vars = vars
        self.state_dim = len(vars)
        self.var_names = [str(v) for v in vars]

        self.context = {
            'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'sqrt': np.sqrt,
            'log': np.log, 'pi': np.pi
        }
        
        # 1. compile drift f(x)
        self.f_compiled = self._compile_vectorized(f_exprs, vars, time_var)
        
        # 2. compile diffusion G(X) & standardize to matrix form
        if isinstance(g_exprs[0], list):
            g_mat_list = g_exprs
        else:
            # diagonal or flat list
            if len(g_exprs) == len(vars):
                g_mat_list = np.diag(g_exprs).tolist()
            else:
                # assuming flattened square
                dim = int(np.sqrt(len(g_exprs)))
                g_mat_list = np.array(g_exprs).reshape((dim, dim)).tolist()
                
        self.g_shape = (len(g_mat_list), len(g_mat_list[0]))
        self.noise_dim = self.g_shape[1]
        
        # compile each cell of G
        self.g_compiled = []
        for row in g_mat_list:
            compiled_row = self._compile_vectorized(row, vars, time_var)
            self.g_compiled.append(compiled_row)
    
    def _compile_vectorized(self, expr_list, vars, time_var):
        """Compile expressions into a vectorized function."""
        args = ", ".join(self.var_names) + f", {time_var}"
        
        expr_strs = [str(ex).replace('^', '**') for ex in expr_list]
        body = f"[{', '.join(expr_strs)}]"
        
        lambda_src = f"lambda {args}: {body}"
        
        try:
            # Create the scalar lambda
            scalar_func = eval(lambda_src, self.context)
            
            # Create a wrapper that unpacks the columns of the state matrix X
            def vector_func(X, t):
                # X is (N, D). Transpose to (D, N) so we can unpack rows into variables
                # Inputs: x0_vec, x1_vec, ... t
                inputs = list(X.T) + [t]
                
                # Result is list of arrays [res_0_vec, res_1_vec ...]
                result_list = scalar_func(*inputs)
                
                # Case where result is scalar (constant expression)
                N = X.shape[0]
                broadcasted_result = []
                for res in result_list:
                    if np.isscalar(res) or res.shape == ():
                        broadcasted_result.append(np.full(N, res))
                    else:
                        broadcasted_result.append(res)
                        
                return np.column_stack(broadcasted_result)
                
            return vector_func
        except Exception as e:
            raise RuntimeError(f"Failed to compile vectorized Monte Carlo function: {e}")
    
    def simulate_traces(self, X0_intervals, t_span, num_traces=100, dt=None,
                         return_deterministic=False, return_x0=False, seed=None):
        """
        generates N sample trajectories using Euler-Maruyama: X_{k+1} = X_k + f(X_k,t) dt + G(X_k,t) dW_k
        - X0_intervals: list of Intervals defining the initial box
        - if return_determinstic=True it also returns the deterministic trajectories
        integrated from the same sampled initial conditions using: x_{k+1} = x_k + f(x_k,t) dt
        - 
        """
        if seed is not None:
            np.random.seed(seed)
        dt_sim = float(dt) if dt is not None else float(self.dt_sim)
        t0, tf = float(t_span[0]), float(t_span[1])

        n_steps = int(np.ceil((tf - t0) / dt_sim)) + 1
        t_grid = t0 + dt_sim * np.arange(n_steps)
        t_grid[-1] = tf
        
        traces = np.zeros((num_traces, n_steps, self.state_dim), dtype=float)
        
        # sample the initial conditions uniformly from X_0 box
        lows = np.array([float(x.lower) if hasattr(x, 'lower') else float(x[0]) for x in X0_intervals], dtype=float)
        highs = np.array([float(x.upper) if hasattr(x, 'upper') else float(x[1]) for x in X0_intervals], dtype=float)

        X0 = np.random.uniform(lows, highs, size=(num_traces, self.state_dim))
        X_curr = X0.copy()
        traces[:, 0, :] = X_curr

        if return_deterministic:
            det_traces = np.zeros_like(traces)
            x_det_curr = X0.copy()
            det_traces[:, 0, :] = x_det_curr
        else:
            det_traces = None
            x_det_curr = None
        
        sqrt_dt = np.sqrt(dt_sim)

        print(f"[Monte Carlo] Simulating {num_traces} traces ({t0}->{tf}) with dt={dt_sim}...")
        
        for i in range(1, n_steps):
            t = t_grid[i-1]

            # generate noise: (N, noise_dim)
            dW = np.random.normal(0.0, 1.0, (num_traces, self.noise_dim))
            
            # evaluate drift: f(X, t) -> (N, D)
            drift = self.f_compiled(X_curr, t)
            
            # evaluate diffusion: G(X, t) -> (N, D, Noise_Dim)
            G_rows = [row_func(X_curr, t) for row_func in self.g_compiled]
            
            # stack into tensor (N, D, Noise_Dim)
            G_val = np.stack(G_rows, axis=1)

            # compute diffusion term: (G @ dW) w/ batch matrix-vector multiply
            # g is (N, D, K), dW is (N, K) -> Out is (N, D)
            diffusion = np.einsum('ijk,ik->ij', G_val, dW)
            
            # euler maruyama update
            X_curr = X_curr + (drift * dt_sim) + (diffusion * sqrt_dt)
            
            traces[:, i, :] = X_curr

            # deterministic paired euler (same dt, no noise)
            if return_deterministic:
                drift_det = self.f_compiled(x_det_curr, t)
                x_det_curr = x_det_curr + drift_det * dt_sim
                det_traces[:, i, :] = x_det_curr
            
        # build return tuple
        out = [t_grid, traces]
        if return_deterministic:
            out.append(det_traces)
        if return_x0:
            out.append(X0)
        return tuple(out)
