# -*- coding: utf-8 -*-
"""
==============================================================
Spatial discretizations for the 1D Burgers equation
==============================================================

Conservation law:

    u_t + (u^2 / 2)_x = 0

Available discretizations
-------------------------

compute_rhs_flux_form
    Conservative finite-volume formulation.

compute_rhs_lw
    Classical finite-difference Lax-Wendroff scheme.

==============================================================
"""

import numpy as np

from Fluxes import (
    lax_friedrich_flux,
    lax_wendroff_flux,
    roe_flux
)

from BoundaryCondition import boundary_condition


# ==========================================================
# Conservative flux-form RHS
# ==========================================================

def compute_rhs_flux_form(
        x,
        u,
        dx,
        dt,
        max_wave_speed,
        numerical_flux=lax_wendroff_flux,
        BC="Neumann"
):
    """
    Compute the conservative finite-volume RHS.

    Parameters
    ----------
    x : ndarray
        Spatial grid.

    u : ndarray
        Solution vector.

    dx : float
        Grid spacing.

    dt : float
        Time step.

    max_wave_speed : float
        Maximum characteristic speed.

    numerical_flux : function
        Numerical flux function.

    problem_type : str
        "shock" or "smooth"

    BC : str
        Boundary condition type.

    Returns
    -------
    du : ndarray
        Spatial residual.
    """

    N = len(x)

    # ======================================================
    # Boundary conditions
    # ======================================================

    x_boundar, u_boundary = boundary_condition(
        x,
        u,
        dx,
        1,
        BC,
        0.0,
        BC,
        0.0
    )

    

    # ======================================================
    # Numerical fluxes
    # ======================================================

    flux_right = numerical_flux(

        u_boundary[1:N+1],
        u_boundary[2:N+2],
        dt / dx,
        max_wave_speed

    )

    flux_left = numerical_flux(

        u_boundary[0:N],
        u_boundary[1:N+1],
        dt / dx,
        max_wave_speed

    )

    # ======================================================
    # Conservative residual
    # ======================================================

    du = -(flux_right - flux_left) / dx

    return du


# ==========================================================
# Classical finite-difference Lax-Wendroff RHS
# ==========================================================

def compute_rhs_lw(
        x,
        u,
        dx,
        dt,
        max_wave_speed,
        BC="Neumann"
):
    """
    Classical finite-difference Lax-Wendroff RHS.

    This form naturally generates oscillations near shocks,
    making it useful for nonlinear filter experiments.

    Parameters
    ----------
    x : ndarray
        Spatial grid.

    u : ndarray
        Solution vector.

    dx : float
        Grid spacing.

    dt : float
        Time step.

    max_wave_speed : float
        Maximum characteristic speed.

    problem_type : str
        "shock" or "smooth"

    BC : str
        Boundary condition type.

    Returns
    -------
    du : ndarray
        Spatial residual.
    """

    N = len(u)

    lam = dt / dx

    # ======================================================
    # Boundary conditions
    # ======================================================


    x_boundary, u_boundary = boundary_condition(
        x,
        u,
        dx,
        1,
        BC,
        0.0,
        BC,
        0.0

    )

   

    # Flux
    f = 0.5 * u_boundary**2

    du = np.zeros_like(u)

    # ======================================================
    # Lax-Wendroff residual
    # ======================================================

    for j in range(N):

        jp = j + 2
        jc = j + 1
        jm = j

        # Roe averages
        ap = 0.5 * (u_boundary[jc] + u_boundary[jp])
        am = 0.5 * (u_boundary[jm] + u_boundary[jc])

        u_new = (

            u_boundary[jc]

            - 0.5 * lam * (f[jp] - f[jm])

            + 0.5 * lam**2 * (

                ap * (f[jp] - f[jc])

                - am * (f[jc] - f[jm])

            )
        )

        # Convert update into RHS form
        du[j] = (u_new - u[j]) / dt

    return du