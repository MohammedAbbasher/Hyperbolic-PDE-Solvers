# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:10:20 2026

@author: mohammed.abbasher
"""

import numpy as np

from BoundaryCondition import boundary_condition
from WENO import WENO
from Fluxes import(
	BLRoe,
	BLLF
)

from numba import njit
@njit(cache=True, fastmath=True)
def compute_rhs(x, u, dx, dt, weno_order, Crec, dw, beta, max_wave_speed):
    """
    Purpose:
    Evaluate the RHS of Burgers equations using a WENO reconstruction
    """

    N = len(x)

    du = np.zeros(N)

    # Extend data and assign boundary conditions
    # Constant boundary conditions
    xe, ue = boundary_condition(
    x,
    u,
    dx,
    weno_order,
    left_BC="Dirichlet",
    uL=1.0,
    right_BC="Dirichlet",
    uR=0.0
    )

    # Define cell left and right interface values
    ul = np.zeros(N + 2)
    ur = np.zeros(N + 2)

    for i in range(0, N + 2):

        stencil_x = xe[i:i + 2 * (weno_order - 1) + 1]
        stencil_u = ue[i:i + 2 * (weno_order - 1) + 1]

        ul[i], ur[i] = WENO(
            stencil_x,
            stencil_u,
            weno_order,
            Crec,
            dw,
            beta
        )

    # Compute residual
    du = -(
        BLRoe(ur[1:N + 1], ul[2:N + 2], 0, max_wave_speed)
        - BLRoe(ur[0:N], ul[1:N + 1], 0, max_wave_speed)
    ) / dx

    return du
