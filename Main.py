# -*- coding: utf-8 -*-
"""
==============================================================
1D Buckley-Leverett Equation Solver
==============================================================

This script solves the 1D Buckley-Leverett equation using:

    • WENO reconstruction
    • Roe or Lax-Friedrich numerical flux
    • SSP-RK3 time integration

The framework is designed to experiment with:
    • high-order shock-capturing methods
    • nonlinear hyperbolic conservation laws
    • WENO reconstructions
    • conservative finite-volume schemes
    • comparison with analytical solutions

==============================================================
Execution Flow of the Solver
==============================================================

Main.py
│
├── 1. Define numerical setup
│       • order of method
│       • CFL number
│       • final time
│       • grid resolution
│
├── 2. Generate computational grid
│
├── 3. Define initial condition
│
├── 4. Call:
│
│       time_integrator(...)
│
▼
TimeIntegrator.py
│
├── Compute WENO coefficients
│
├── Compute smoothness indicators
│
├── Advance solution using SSP-RK3
│
├── Call:
│
│       compute_rhs(...)
│
▼
RHS.py
│
├── Apply ghost-cell extension
│
├── Perform WENO reconstruction
│
├── Compute interface fluxes
│
└── Return semi-discrete residual

▼
Fluxes.py
│
├── roe_flux(...)
│
└── lax_friedrich_flux(...)

▼
Main.py
│
├── Compute analytical solution
│
└── Plot numerical vs exact solution

==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Time integration
# ==========================================================

from TimeIntegrator import time_integrator

# ==========================================================
# Exact solution
# ==========================================================

from ExactSolution import exact_solution

# ==========================================================
# Numerical setup
# ==========================================================

domain_length = 1.0

num_cells = 528

final_time = 0.5

CFL = 0.1

# ==========================================================
# WENO order
# ==========================================================

weno_order = 3

# ==========================================================
# Grid
# ==========================================================

dx = domain_length / num_cells

x = np.arange(0.0, domain_length + dx, dx)

# ==========================================================
# Initial condition
# ==========================================================

# Constant initial condition
u0 = np.zeros_like(x)

# ==========================================================
# Solve Buckley-Leverett equation
# ==========================================================

u = time_integrator(
    x,
    u0,
    dx,
    weno_order,
    CFL,
    final_time
)

# ==========================================================
# Compute exact solution
# ==========================================================

x_exact, u_exact = exact_solution(
    final_time,
    4096
)

# ==========================================================
# Plot solutions
# ==========================================================

plt.figure(figsize=(8, 5))

# Numerical solution
plt.plot(
    x,
    u,
    linewidth=2,
    label='WENO Solution'
)

# Exact solution
plt.plot(
    x_exact,
    u_exact,
    '--',
    linewidth=2,
    label='Exact Solution'
)

# ==========================================================
# Plot formatting
# ==========================================================

plt.xlabel('x')
plt.ylabel('u')

plt.title(
    f'Buckley-Leverett Equation | '
    f'WENO Order = {weno_order}'
)

plt.grid(True)

plt.legend()

plt.show()

# ==========================================================
# Error calculation
# ==========================================================

# Interpolate exact solution onto numerical grid
u_exact_interp = np.interp(
    x,
    x_exact,
    u_exact
)

# Infinity norm error
error = np.linalg.norm(
    np.abs(u_exact_interp - u),
    np.inf
)

print('Infinity norm error = ', error)