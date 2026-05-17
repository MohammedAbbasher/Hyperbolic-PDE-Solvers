# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:05:46 2026

@author: mohammed.abbasher
"""

# -*- coding: utf-8 -*-
"""
==============================================================
Numerical flux functions for the 1D Burgers equation
==============================================================

Conservation law:

    u_t + (u^2 / 2)_x = 0

In the conservative finite-volume formulation,
the spatial derivative is approximated by:

           F(i+1/2) - F(i-1/2)
    u_t + ------------------- = 0
                    dx

where:

    F(i+1/2)

is a numerical approximation of the physical flux
at the interface between two neighboring cells.

The numerical flux determines:
    • stability
    • numerical dissipation
    • shock resolution
    • oscillatory behavior

==============================================================
Available numerical fluxes
==============================================================

lax_friedrich_flux
------------------

Replaces the interface flux using a strongly
dissipative average:

    • very robust
    • monotone
    • stable near discontinuities
    • smears shocks due to high diffusion


lax_wendroff_flux
-----------------

Uses a second-order Taylor expansion in time
to approximate the interface flux.

Characteristics:
    • higher-order accuracy
    • lower numerical diffusion
    • sharper smooth solutions
    • generates dispersive oscillations near shocks

Useful for nonlinear filter experiments because
the oscillations become clearly visible.


roe_flux
---------

Uses a local linearized Riemann solver based on
the Roe average wave speed.

Characteristics:
    • better shock resolution
    • less diffusive than Lax-Friedrich
    • captures wave propagation direction
    • may require entropy fixes near sonic points

==============================================================
"""

import numpy as np


# ==========================================================
# Lax-Friedrich flux
# ==========================================================

def lax_friedrich_flux(u, v, lam, max_wave_speed):

    """
    Lax-Friedrich numerical flux.
    """

    flux_u = 0.5 * u**2
    flux_v = 0.5 * v**2

    numflux = (
        0.5 * (flux_u + flux_v)
        - 0.5 * max_wave_speed * (v - u)
    )

    return numflux


# ==========================================================
# Lax-Wendroff flux
# ==========================================================

def lax_wendroff_flux(u, v, lam, max_wave_speed):

    """
    Lax-Wendroff numerical flux.
    """

    flux_u = 0.5 * u**2
    flux_v = 0.5 * v**2

    # Characteristic speed approximation
    alpha = lam * (u + v) / 2.0

    numflux = (
        0.5 * (flux_u + flux_v)
        - 0.5 * alpha * (flux_v - flux_u)
    )

    return numflux


# ==========================================================
# Roe flux
# ==========================================================

def roe_flux(u, v, lam, max_wave_speed):

    """
    Roe approximate Riemann solver.

    No entropy fix applied.
    """

    flux_u = 0.5 * u**2
    flux_v = 0.5 * v**2

    # Roe speed
    alpha = 0.5 * (u + v)

    # Upwind flux selection
    numflux = np.where(alpha >= 0.0, flux_u, flux_v)

    return numflux