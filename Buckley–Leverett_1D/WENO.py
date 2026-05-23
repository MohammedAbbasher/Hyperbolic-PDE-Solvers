# -*- coding: utf-8 -*-
"""
==============================================================
WENO Reconstruction
==============================================================

Optimized WENO reconstruction routine for the
Buckley-Leverett equation solver.

Main optimizations:
    • Numba JIT compilation
    • Reduced temporary allocations
    • Reduced repeated slicing
    • Explicit loops instead of nested np.dot
    • Better memory locality

==============================================================
"""

import numpy as np

from numba import njit


@njit(cache=True, fastmath=True)
def WENO(xloc, uloc, weno_order, Crec, dw, beta):

    """
    Purpose:
    Compute left/right interface values using
    classical WENO reconstruction.
    """

    # ==========================================================
    # WENO parameters
    # ==========================================================

    p = 1

    vareps = 1e-6

    # ==========================================================
    # Special case: first-order
    # ==========================================================

    if weno_order == 1:

        um = uloc[0]

        up = uloc[0]

        return um, up

    # ==========================================================
    # Allocate arrays
    # ==========================================================

    alpham = np.zeros(weno_order)

    alphap = np.zeros(weno_order)

    upl = np.zeros(weno_order)

    uml = np.zeros(weno_order)

    betar = np.zeros(weno_order)

    # ==========================================================
    # Loop over candidate stencils
    # ==========================================================

    for r in range(weno_order):

        start = weno_order - r - 1

        # ======================================================
        # Extract local stencil manually
        # ======================================================

        umh = np.empty(weno_order)

        for j in range(weno_order):

            umh[j] = uloc[start + j]

        # ======================================================
        # Reconstructed interface values
        # ======================================================

        sum_up = 0.0

        sum_um = 0.0

        for j in range(weno_order):

            sum_up += Crec[r + 1, j] * umh[j]

            sum_um += Crec[r, j] * umh[j]

        upl[r] = sum_up

        uml[r] = sum_um

        # ======================================================
        # Smoothness indicator
        # ======================================================

        beta_sum = 0.0

        for i in range(weno_order):

            for j in range(weno_order):

                beta_sum += (
                    umh[i]
                    * beta[i, j, r]
                    * umh[j]
                )

        betar[r] = beta_sum

    # ==========================================================
    # Nonlinear WENO weights
    # ==========================================================

    for r in range(weno_order):

        denom = (vareps + betar[r]) ** (2 * p)

        alphap[r] = dw[r] / denom

        alpham[r] = dw[weno_order - r - 1] / denom

    # ==========================================================
    # Compute weighted interface values
    # ==========================================================

    sum_alpham = 0.0

    sum_alphap = 0.0

    sum_um = 0.0

    sum_up = 0.0

    for r in range(weno_order):

        sum_alpham += alpham[r]

        sum_alphap += alphap[r]

        sum_um += alpham[r] * uml[r]

        sum_up += alphap[r] * upl[r]

    um = sum_um / sum_alpham

    up = sum_up / sum_alphap

    return um, up
