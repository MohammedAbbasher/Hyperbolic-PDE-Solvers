# -*- coding: utf-8 -*-
"""
==============================================================
Ghost-cell boundary-condition extension
for Buckley-Leverett equation
==============================================================
"""

import numpy as np

from numba import njit


@njit(cache=True, fastmath=True)
def boundary_condition(
        x,
        u,
        dx,
        ng,
        left_BC="Dirichlet",
        uL=1.0,
        right_BC="Dirichlet",
        uR=0.0
):
    """
    Extend solution using ghost cells.

    Parameters
    ----------
    x : ndarray
        Physical grid.

    u : ndarray
        Solution vector.

    dx : float
        Grid spacing.

    ng : int
        Number of ghost cells.

    left_BC : str
        Left boundary condition.

        Options:
        - "Dirichlet"
        - "Neumann"
        - "Periodic"

    right_BC : str
        Right boundary condition.

        Options:
        - "Dirichlet"
        - "Neumann"
        - "Periodic"

    uL : float
        Left Dirichlet value.

    uR : float
        Right Dirichlet value.

    Returns
    -------
    x_ext : ndarray
        Extended grid.

    u_ext : ndarray
        Extended solution.
    """

    # ======================================================
    # Basic quantities
    # ======================================================

    N = len(u)

    xl = np.min(x)
    xr = np.max(x)

    # ======================================================
    # Allocate extended arrays
    # ======================================================

    x_ext = np.zeros(N + 2 * ng)

    u_ext = np.zeros(N + 2 * ng)

    q = np.arange(1, ng + 1)

    # ======================================================
    # Interior values
    # ======================================================

    x_ext[ng:ng + N] = x[:N]

    u_ext[ng:ng + N] = u[:N]

    # ======================================================
    # Extend coordinates
    # ======================================================

    x_ext[ng - q] = xl - q * dx

    x_ext[N + ng - 1 + q] = xr + q * dx

    # ======================================================
    # PERIODIC boundary conditions
    # ======================================================

    if (left_BC == "Periodic") or (right_BC == "Periodic"):

        u_ext[ng - q] = u[N - q - 1]

        u_ext[N + ng - 1 + q] = u[q]

        return x_ext, u_ext

    # ======================================================
    # LEFT boundary condition
    # ======================================================

    if left_BC == "Dirichlet":

        u_ext[ng - q] = uL

    elif left_BC == "Neumann":

        u_ext[ng - q] = u[q]

    else:

        raise ValueError("Unknown LEFT boundary condition")

    # ======================================================
    # RIGHT boundary condition
    # ======================================================

    if right_BC == "Dirichlet":

        u_ext[N + ng - 1 + q] = uR

    elif right_BC == "Neumann":

        u_ext[N + ng - 1 + q] = u[N - q - 1]

    else:

        raise ValueError("Unknown RIGHT boundary condition")

    return x_ext, u_ext