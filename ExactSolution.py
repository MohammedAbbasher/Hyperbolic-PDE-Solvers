# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:48:36 2026

@author: mohammed.abbasher
"""

import numpy as np

from numba import njit

@njit(cache=True, fastmath=True)
def exact_solution(FinalTime, n=4096):
    """
    Purpose:
    Compute the analytical solution of the
    1D Buckley-Leverett problem.

    Parameters
    ----------
    FinalTime : float
        Physical time

    n : int
        Number of points

    Returns
    -------
    y : numpy array
        Spatial coordinates

    U : numpy array
        Analytical solution
    """

    # Domain spacing
    dx = 1.0 / n

    # Allocate arrays
    y = np.zeros(n)
    U = np.zeros(n)

    T = FinalTime

    # Build analytical solution
    for i in range(n):

        y[i] = i * dx

        # Avoid division by zero
        if i == 0:
            U[i] = 1.0
            continue

        z = y[i] / T

        if z < 0.5 * (1 + np.sqrt(2)):

            term = (
                (
                    -2 * z
                    + np.sqrt(4 * z + 1)
                    - 1
                ) / z
            ) + 1

            U[i] = 0.5 * (np.sqrt(term) + 1)

        else:

            U[i] = 0.0

    return y, U