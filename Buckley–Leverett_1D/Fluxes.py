# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:00:33 2026

@author: mohammed.abbasher
"""

from numba import njit


@njit(cache=True, fastmath=True)
def BLLF(u, v, lambd, max_wave_speed):
    """
    Purpose:
    Evaluate the Lax-Friedrich numerical flux
    for Buckley-Leverett equation
    """

    M = 1.0

    fu = (u**2) / ((u**2) + M * (1.0 - u)**2)

    fv = (v**2) / ((v**2) + M * (1.0 - v)**2)

    numflux = (fu + fv) / 2.0 - max_wave_speed / 2.0 * (v - u)

    return numflux


@njit(cache=True, fastmath=True)
def BLRoe(u, v, lambd, max_wave_speed):
    """
    Purpose:
    Evaluate Roe numerical flux
    for Buckley-Leverett equation.

    No sonic fix.
    """

    fu = (u**2) / ((u**2) + (1.0 - u)**2)

    fv = (v**2) / ((v**2) + (1.0 - v)**2)

    alpha = u + v

    numflux = (alpha >= 0.0) * fu + (alpha < 0.0) * fv

    return numflux
