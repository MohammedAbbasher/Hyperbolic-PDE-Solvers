# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:03:03 2026

@author: mohammed.abbasher
"""

import numpy as np

from Utilities import (
    compute_reconstruction_weights,
    compute_linear_weights,
    compute_smoothness_matrix
)
from RHS import compute_rhs

def time_integrator(x, u, dx, weno_order, CFL, FinalTime):
    """
    Purpose:
    Integrate 1D Buckley-Leverett equation until FinalTime
    using a WENO scheme and 3rd order SSP-RK method
    """

    time = 0.0
    time_step = 0

    # Initialize reconstruction weights
    Crec = np.zeros((weno_order + 1, weno_order))

    for r in range(-1, weno_order):
        Crec[r + 1, :] = compute_reconstruction_weights(weno_order, r)
        
      
    # Initialize linear weights
    dw =  compute_linear_weights(weno_order, 0)

    # Compute smoothness indicator matrices
    beta = np.zeros((weno_order, weno_order, weno_order))

    for r in range(0, weno_order):
        xl = -0.5 + np.arange(-r, weno_order - r + 1, 1)
        beta[:, :, r] = compute_smoothness_matrix(xl, weno_order)

    # Integrate scheme
    while time < FinalTime:

        # Decide on timestep
        # y = 2*(2*u*(1-u))
        # w = (u**2) + 2*((1-u)**2)**2
        # max_wave_speed = max(abs(y/w))

        #The derivative was calculated manually and the value was entered directly
        max_wave_speed = 2

        dt = CFL * dx / max_wave_speed

        if (time + dt) > FinalTime:
            dt = FinalTime - time

        # Update solution

        rhsu = compute_rhs(x, u, dx, dt, weno_order, Crec, dw, beta, max_wave_speed)
        u1 = u + dt * rhsu

        rhsu = compute_rhs(x, u1, dx, dt, weno_order, Crec, dw, beta, max_wave_speed)
        u2 = (3 * u + u1 + dt * rhsu) / 4

        rhsu = compute_rhs(x, u2, dx, dt, weno_order, Crec, dw, beta, max_wave_speed)
        u = (u + 2 * u2 + 2 * dt * rhsu) / 3

        time = time + dt
        time_step = time_step + 1

    return u
