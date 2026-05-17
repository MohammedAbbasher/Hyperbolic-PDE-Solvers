# -*- coding: utf-8 -*-
"""
Created on Thu May 14 09:16:49 2026

@author: mohammed.abbasher
"""

"""
==============================================================
1D Burgers Equation Solver
==============================================================

This script solves the 1D inviscid Burgers equation

    u_t + ((u^2)/2)_x = 0

using finite-difference / finite-volume methods together with
optional nonlinear shock filters from:

    Engquist et al.
    "Nonlinear Filters for Efficient Shock Computation"

The framework is designed to experiment with:
    • shock-capturing methods
    • nonlinear post-processing filters
    • conservative and non-conservative schemes
    • oscillation suppression near discontinuities
    • comparison with analytical solutions

==============================================================
Execution Flow of the Solver
==============================================================

Main.py
│
├── 1. Select simulation options
│       • test case
│       • boundary condition
│       • RHS discretization
│       • numerical flux
│       • nonlinear filter
│
├── 2. Generate grid and initial condition
│
├── 3. Call:
│
│       time_integrator(...)
│
│
▼
TimeIntegrator.py
│
├── Compute time step from CFL condition
│
├── Call selected RHS discretization:
│
│       rhs_function(...)
│
│
▼
RHS.py
│
├── Apply ghost-cell extension:
│
│       boundary_condition(...)
│
├── Depending on chosen discretization:
│
│   • compute_rhs_flux_form(...)
│   │
│   │    └── Uses selected numerical flux:
│   │
│   │           lax_wendroff_flux(...)
│   │           lax_friedrich_flux(...)
│   │           roe_flux(...)
│   │
│   └── compute_rhs_lw(...)
│
│        Classical finite-difference
│        Lax-Wendroff formulation
│
│
▼
TimeIntegrator.py
│
├── Advance solution in time
│
├── Optional nonlinear filtering:
│
│       engquist_filter_21(...)
│       engquist_filter_22(...)
│       engquist_filter_24(...)
│
└── Repeat until final_time reached
│
▼
Main.py
│
├── Optional exact solution evaluation:
│
│       exact_shock_solution(...)
│
└── Plot final solution

==============================================================
Available boundary conditions
==============================================================

Periodic
    Wrap-around boundary condition.

Neumann
    Zero-gradient / transmissive boundary condition.

Dirichlet
    Fixed boundary value.

==============================================================
Available test cases
==============================================================

Smooth
    Smooth sinusoidal wave.

Shock
    Discontinuous step function.

==============================================================
Available analytical solutions
==============================================================

exact_shock_solution
--------------------

Exact entropy solution of the inviscid Burgers equation
for Riemann-type shock problems.

Used to compare:
    • shock position
    • numerical diffusion
    • oscillation behavior
    • filter performance

==============================================================
Available RHS discretizations
==============================================================

compute_rhs_flux_form
    Conservative finite-volume formulation.

    Requires a numerical flux function.

    Recommended for:
        • standard conservative simulations
        • monotone shock-capturing methods
        • Roe / Lax-Friedrich experiments


compute_rhs_lw
    Classical finite-difference Lax-Wendroff scheme.

    Naturally produces dispersive oscillations near shocks,
    making it ideal for nonlinear filter experiments.

    Does NOT use numerical fluxes directly.

==============================================================
Available numerical fluxes
==============================================================

lax_wendroff_flux
    Second-order accurate.

    Produces oscillations near discontinuities.

    Recommended for observing the effect of nonlinear filters.


lax_friedrich_flux
    Very robust and monotone.

    More diffusive / smeared shocks.


roe_flux
    Sharper shock capturing.

    Less diffusive than Lax-Friedrich.

==============================================================
Available nonlinear filters
==============================================================

None
    No filtering.


engquist_filter_21
    Algorithm 2.1

    Removes simple isolated extrema.


engquist_filter_22
    Algorithm 2.2

    Improved version of Algorithm 2.1.

    Handles neighboring extrema better.


engquist_filter_24
    Algorithm 2.4

    Most advanced filter.

    Distinguishes admissible and non-admissible extrema.

    Produces sharper shocks with less unnecessary smoothing.

==============================================================
Typical configurations
==============================================================

1) Shock + nonlinear filter experiment
--------------------------------------

rhs_function    = compute_rhs_lw

selected_filter = engquist_filter_24

Purpose:
    Observe oscillation removal near shocks.


2) Conservative finite-volume simulation
----------------------------------------

rhs_function   = compute_rhs_flux_form

numerical_flux = roe_flux

Purpose:
    Standard conservative shock capturing.


3) Diffusive monotone simulation
--------------------------------

rhs_function   = compute_rhs_flux_form

numerical_flux = lax_friedrich_flux

Purpose:
    Very stable but smeared shocks.

==============================================================
Important notes
==============================================================

To clearly observe the effect of nonlinear filters,
it is recommended to use:

    rhs_function = compute_rhs_lw

because the classical Lax-Wendroff scheme naturally generates
dispersive oscillations near discontinuities.


Algorithms 2.1 and 2.2:
    • can remove many oscillations
    • may smear smooth extrema
    • may not fully suppress strong Gibbs oscillations


Algorithm 2.4:
    • preserves admissible extrema better
    • produces sharper shocks
    • reduces unnecessary smoothing


The finite-difference Lax-Wendroff scheme used for filter
experiments is intentionally non-monotone, following the
philosophy of Engquist et al., where nonlinear filters are
used to recover non-oscillatory behavior while preserving
high-order accuracy away from shocks.

==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from TimeIntegrator import time_integrator 

# ==========================================================
# Spatial discretization
# ==========================================================
from RHS import (
    compute_rhs_flux_form,
    compute_rhs_lw
)

# ==========================================================
# Numerical fluxes
# ==========================================================

from Fluxes import (
    lax_wendroff_flux,
    lax_friedrich_flux,
    roe_flux
)

# ==========================================================
# Nonlinear filters
# ==========================================================

from Filters import (
    engquist_filter_21,
    engquist_filter_22,
    engquist_filter_24
)



# ==========================================================
# Numerical setup
# ==========================================================

xmin = -1.0
xmax =  1.0

num_grid_points = 180

final_time = 0.75

CFL = 0.8


# ==========================================================
# Boundary condition selection
# ==========================================================

# Options:
# "Periodic"
# "Neumann"
# "Dirichlet"

boundary_condition = "Neumann"


# ==========================================================
# Test case selection
# ==========================================================

# Options:
# "Smooth"
# "Shock"

test_case = "Shock"

# ==========================================================
# Spatial discretization selection
# ==========================================================

# Options:
#
# compute_rhs_flux_form
#     Conservative finite-volume formulation.
#     Requires a numerical flux.
#
# compute_rhs_lw
#     Classical finite-difference Lax-Wendroff scheme.
#     Used for nonlinear filter experiments.
#     Does NOT use numerical fluxes.



# Recommended:
#   Smooth -> compute_rhs_flux_form
#   Shock  -> compute_rhs_lw

rhs_function = compute_rhs_flux_form

# ==========================================================
# Numerical flux selection
# ==========================================================

# Only used with:
#
#     compute_rhs_flux_form
#
# Ignored when using:
#
#     compute_rhs_lw

numerical_flux = lax_wendroff_flux


# ==========================================================
# Filter selection
# ==========================================================

# Options:
# None
# engquist_filter_21
# engquist_filter_22
# engquist_filter_24

selected_filter = None


# ==========================================================
# Grid
# ==========================================================

x = np.linspace(xmin, xmax, num_grid_points + 1)

dx = (xmax - xmin) / num_grid_points


# ==========================================================
# Initial conditions
# ==========================================================

if test_case == "Smooth":

    u0 = 0.25 + 0.5*np.sin(np.pi*x)

elif test_case == "Shock":
    uL = 1.0
    uR = 0.0

    u0 = np.where(x < 0.0, uL, uR)

else:

    raise ValueError("Unknown test case")


# ==========================================================
# Solve Burgers equation numerically
# ==========================================================

u = time_integrator(
    x,
    u0,
    dx,
    CFL,
    final_time,
    rhs_function,
    numerical_flux,
    selected_filter,
    boundary_condition
)

# ==========================================================
# Exact solution for shock test
# ==========================================================

if test_case == "Shock":

    from ExactSolutions import exact_shock_solution

    u_exact = exact_shock_solution(
        x,
        final_time,
        uL,
        uR
    )

# ==========================================================
# Plot solution
# ==========================================================

plt.figure(figsize=(8,5))

# Numerical solution
plt.plot(
    x,
    u,
    'r-',
    linewidth=2,
    label='Numerical'
)

# Exact solution only for shock case
if test_case == "Shock":

    plt.plot(
        x,
        u_exact,
        'k--',
        linewidth=2,
        label='Exact'
    )

plt.xlabel('x')
plt.ylabel('u')

plt.title(
    f"{test_case} test | "
    f"BC = {boundary_condition}"
)

plt.grid(True)

plt.legend()

plt.show()