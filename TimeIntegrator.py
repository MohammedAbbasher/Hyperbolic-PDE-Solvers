# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:51:29 2026

@author: mohammed.abbasher
"""

# -*- coding: utf-8 -*-
"""
==============================================================
Time integration for the 1D Burgers equation
==============================================================

This module advances the semi-discrete system

    du/dt = RHS(u)

using explicit forward Euler time integration.

Supported features
------------------
• Conservative finite-volume methods
• Classical finite-difference Lax-Wendroff
• Optional nonlinear shock filters
• CFL-controlled timestep adaptation

==============================================================
"""

import numpy as np

from RHS import (
    compute_rhs_flux_form,
    compute_rhs_lw
)


def time_integrator(
        x,
        u0,
        dx,
        CFL,
        final_time,
        rhs_function,
        numerical_flux,
        selected_filter,
        boundary_condition
):
    """
    Advance the 1D Burgers equation in time.

    Parameters
    ----------
    x : ndarray
        Spatial grid.

    u : ndarray
        Initial solution.

    h : float
        Grid spacing.

    cfl : float
        CFL number.

    final_time : float
        Final simulation time.

    rhs_function : function
        Spatial discretization function.

    numerical_flux : function or None
        Numerical flux for flux-form methods.

     selected_filter : function or None
        Nonlinear post-processing filter.

    Returns
    -------
    u : ndarray
        Solution at final time.
    """

    # ======================================================
    # Initialization
    # ======================================================

    time = 0.0
    time_step = 0

    # ======================================================
    # Time integration loop
    # ======================================================
    u = u0.copy()

    while time < final_time:

        # Maximum wave speed
        max_wave_speed = np.max(np.abs(u))

        # Avoid division by zero
        if max_wave_speed < 1e-14:
            break

        # CFL timestep
        dt = CFL * dx / max_wave_speed

        # Prevent overshooting
        if time + dt > final_time:

            dt = final_time - time

        # ==================================================
        # Flux-form methods
        # ==================================================

        if rhs_function == compute_rhs_flux_form:

            du = rhs_function(

                x,
                u,
                dx,
                dt,
                max_wave_speed,
                numerical_flux

            )

        # ==================================================
        # Classical FD-LW method
        # ==================================================

        else:

            du = rhs_function(

                x,
                u,
                dx,
                dt,
                max_wave_speed

            )

        # --------------------------------------------------
        # Forward Euler update
        # --------------------------------------------------

        u += dt * du

        # ==================================================
        # Optional nonlinear filter
        # ==================================================

        if  selected_filter is not None:

            u =  selected_filter(u)

        # Advance time
        time += dt
        time_step += 1

    return u