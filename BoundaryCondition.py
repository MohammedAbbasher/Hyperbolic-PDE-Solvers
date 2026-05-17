# -*- coding: utf-8 -*-
"""
==============================================================
Ghost-cell boundary-condition extension
==============================================================
"""

import numpy as np


def boundary_condition(
        x,
        u,
        dx,
        ng,
        left_BC="Neumann",
        uL=0.0,
        right_BC="Neumann",
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

    right_BC : str
        Right boundary condition.

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

    N = len(u)

    # ======================================================
    # Allocate extended arrays
    # ======================================================

    x_ext = np.zeros(N + 2*ng)
    u_ext = np.zeros(N + 2*ng)

    # ======================================================
    # Interior values
    # ======================================================

    x_ext[ng:ng+N] = x
    u_ext[ng:ng+N] = u

    # ======================================================
    # Extend coordinates
    # ======================================================

    x_ext[:ng] = x[0] - dx*np.arange(ng, 0, -1)
    x_ext[-ng:] = x[-1] + dx*np.arange(1, ng + 1)

    # ======================================================
    # LEFT boundary condition
    # ======================================================

    if left_BC == "Periodic":

        u_ext[:ng] = u[-ng:]

    elif left_BC == "Neumann":

        u_ext[:ng] = u[0]

    elif left_BC == "Dirichlet":

       u_ext[:ng] = 2*uL - u[:ng][::-1]

    else:

        raise ValueError("Unknown LEFT boundary condition")

    # ======================================================
    # RIGHT boundary condition
    # ======================================================

    if right_BC == "Periodic":

        u_ext[-ng:] = u[:ng]

    elif right_BC == "Neumann":

        u_ext[-ng:] = u[-1]

    elif right_BC == "Dirichlet":

       u_ext[N + ng:N + 2*ng] = 2*uR - u[-ng:][::-1]

    else:

        raise ValueError("Unknown RIGHT boundary condition")

    return x_ext, u_ext