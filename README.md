![Continuous Single Step Jet Engine](Docs/Saved%20Plots/TERA_logo.png)
# TERA

**T**aylor Model **E**nabled **R**eachability **A**nalysis Framework.

**TERA** is a Python-native, open-source **Taylor Model (TM)** reachability framework for computing **rigorous flowpipe enclosures** of nonlinear dynamical systems under uncertainty. It unifies **continuous**, **hybrid**, and **stochastic** analysis within a single symbolic–numeric workflow to support rapid prototyping without sacrificing enclosure guarantees.

---

## Highlights

- **Validated TM flowpipe computation for nonlinear ODEs**
  - Local **single-step** TM integration (validated ODE solving in the sense of Berz/Makino-style TM integrators).
  ![Continuous Single Step Jet Engine](Docs/Saved%20Plots/Continuous_Lorenz_xy.png)
  - **Left–right (compositional) propagation** to mitigate wrapping over long horizons (“shrink-wrapping” preconditioning, formalized in later preconditioning work).
  ![Continuous Left-Right Quadratic Model](Docs/Saved%20Plots/Continuous_LeftRight_QuadraticModel_y1y2.png)
- **Hybrid reachability**
  - Discrete transition processing with **guard intersection** and **reset mapping**, consistent with TM-based hybrid reachability semantics. 
  ![Hybrid Bouncing Ball](Docs/Saved%20Plots/Hybrid_BouncingBall_xt.png)
- **Stochastic reachability**
  - **$\delta$-probabilistic reachable set ($\delta$-PRS)** enclosures: sets that contain trajectories with probability at least **1 − $\delta$**.
  - Computes $\delta$-PRS by combining deterministic TM flowpipes with probabilistic deviation bounds (AMGF-style separation strategy). 
  ![Stochastic Spring Model](Docs/Saved%20Plots/Stochastic_SpringModel_x1x2.png)
- **Visualization utilities**
  - Plotting/projection tools for deterministic flowpipes and $\delta$-PRS enclosures, plus comparisons against Monte Carlo samples. 
- **Examples and notebooks** for continuous, hybrid, and stochastic workflows and benchmarks.

---

## Project Layout

- `TERA/TMCore` — core algebra: intervals, polynomials, Taylor models, TM vectors.
- `TERA/TMFlow` — continuous reachability: flowpipe construction, remainder handling, preconditioning (single-step + left–right).
- `TERA/Hybrid` — hybrid automata: modes, invariants, guard intersection, resets, worklist orchestration.
- `TERA/Stochastic` — stochastic reachability and $\delta$-PRS workflows; sampling utilities for comparison.
- `TERA/Workbench` — visualization and result helpers.
- `Examples` — Jupyter notebooks for end-to-end usage.

---

## Installation

See `INSTALL.txt` for up-to-date installation instructions. SageMath is required for core functionality, and installation is currently documented as an editable install inside a SageMath environment.
Note: this project is not currently pip-installable; please use the SageMath environment workflow described in `INSTALL.txt`.

## Examples/Benchmarks
Run one of the example notebooks in `Examples/`:

- Continuous_*.ipynb

- Hybrid_*.ipynb

- Stochastic_*.ipynb

## Configuration Notes

For full configuration details, see:
`Docs/Configuration_Ref.md` and `Docs/Workbench_API.md`.

## Status

Research prototype. APIs and configuration knobs may evolve as the engine matures. Current focus is consolidating hybrid + stochastic components into a unified SHS formulation. 
CI tests are planned for a future release (not yet enabled).

Notebook size: example notebooks in `Examples/` include saved outputs. If repo size matters, clear notebook outputs before pushing.

## License
This project is licensed under the GNU General Public License v2.0 (GPL-2.0). TERA is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License, version 2 (GPL-2.0). A copy of the license is provided in `LICENSE.GPL2`.

