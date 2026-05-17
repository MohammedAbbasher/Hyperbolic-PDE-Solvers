# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:18:50 2026

@author: mohammed.abbasher
"""
"""
The exact solution automatically uses:
    • left state  uL
    • right state uR

defined directly in the initial condition.

Shock speed is computed from the
Rankine-Hugoniot condition:

                f(uL) - f(uR)
    s = ------------------------------
              uL - uR

For Burgers equation:

            f(u) = u^2 / 2

giving:

            s = (uL + uR)/2
"""
import numpy as np


def exact_shock_solution(x, t, uL, uR):
    """
    Exact entropy solution of the inviscid
    Burgers Riemann problem.

    Parameters
    ----------
    x : ndarray
        Spatial grid.

    t : float
        Time.

    uL : float
        Left state.

    uR : float
        Right state.

    Returns
    -------
    u_exact : ndarray
        Exact solution.
    """

    # Shock speed
    s = 0.5 * (uL + uR)

    shock_position = s * t

    u_exact = np.where(
        x < shock_position,
        uL,
        uR
    )

    return u_exact