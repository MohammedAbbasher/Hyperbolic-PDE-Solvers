# -*- coding: utf-8 -*-
"""
Created on Fri May 15 09:53:09 2026

@author: mohammed.abbasher
"""

# -*- coding: utf-8 -*-
"""
==============================================================
Nonlinear shock filters
==============================================================

This file contains nonlinear post-processing filters inspired by:

    Engquist et al.
    "Nonlinear Filters for Efficient Shock Computation"

Available filters
-----------------

engquist_filter_21
    Algorithm 2.1

engquist_filter_22
    Algorithm 2.2

engquist_filter_24
    Algorithm 2.4

==============================================================
"""

import numpy as np


# ==========================================================
# Algorithm 2.1
# ==========================================================

def engquist_filter_21(u):
    """
    Engquist Algorithm 2.1

    Removes isolated extrema using conservative redistribution.
    """

    uf = u.copy()

    N = len(u)

    for j in range(1, N - 1):

        forward_jump  = u[j + 1] - u[j]
        backward_jump = u[j] - u[j - 1]

        # Extremum detection
        if forward_jump * backward_jump >= 0:
            continue

        # Determine larger jump
        if abs(forward_jump) > abs(backward_jump):

            delta_plus  = abs(forward_jump)
            delta_minus = abs(backward_jump)

            correction_index = j + 1

        else:

            delta_plus  = abs(backward_jump)
            delta_minus = abs(forward_jump)

            correction_index = j - 1

        # TVD correction
        delta = min(delta_plus / 2.0, delta_minus)

        # Correction sign
        s = np.sign(forward_jump)

        # Conservative redistribution
        uf[j] += s * delta

        uf[correction_index] -= s * delta

    return uf


# ==========================================================
# Algorithm 2.2
# ==========================================================

def engquist_filter_22(u):
    """
    Engquist Algorithm 2.2

    Improved version of Algorithm 2.1.

    Attempts to better handle neighboring extrema.
    """

    uf = u.copy()

    N = len(u)

    for j in range(2, N - 1):

        forward_jump  = u[j + 1] - u[j]
        backward_jump = u[j] - u[j - 1]

        # Extremum detection
        extremum = (forward_jump * backward_jump < 0)

        if not extremum:
            continue

        # Secondary extremum detection
        left_forward  = u[j] - u[j - 1]
        left_backward = u[j - 1] - u[j - 2]

        neighboring_extremum = (
            left_forward * left_backward < 0
        )

        # ==============================================
        # Standard Algorithm 2.1 correction
        # ==============================================

        if not neighboring_extremum:

            if abs(forward_jump) > abs(backward_jump):

                delta_plus  = abs(forward_jump)
                delta_minus = abs(backward_jump)

                correction_index = j + 1

            else:

                delta_plus  = abs(backward_jump)
                delta_minus = abs(forward_jump)

                correction_index = j - 1

            delta = min(delta_plus / 2.0, delta_minus)

            s = np.sign(forward_jump)

            uf[j] += s * delta

            uf[correction_index] -= s * delta

        # ==============================================
        # Neighboring extrema correction
        # ==============================================

        else:

            delta_plus  = abs(forward_jump)
            delta_minus = abs(backward_jump)

            delta_left = abs(u[j - 1] - u[j - 2])

            delta1 = min(delta_plus, delta_minus / 2.0)

            delta2 = min(delta1, delta_left)

            s = np.sign(forward_jump)

            uf[j - 1] -= s * delta2

            uf[j] += s * delta2

    return uf


# ==========================================================
# Algorithm 2.4
# ==========================================================

def engquist_filter_24(u):
    """
    Engquist Algorithm 2.4

    Most advanced admissibility-based filter.

    Distinguishes admissible and non-admissible extrema.
    """

    uf = u.copy()

    N = len(u)

    for j in range(2, N - 1):

        # First differences
        forward_jump  = u[j + 1] - u[j]
        backward_jump = u[j] - u[j - 1]

        # Extremum detection
        if forward_jump * backward_jump >= 0:
            continue

        # ==============================================
        # Local maximum
        # ==============================================

        if forward_jump < 0 and backward_jump > 0:

            extremum_strength = (
                u[j] - max(u[j - 1], u[j + 1])
            )

        # ==============================================
        # Local minimum
        # ==============================================

        else:

            extremum_strength = (
                min(u[j - 1], u[j + 1]) - u[j]
            )

        # Admissible extremum
        if extremum_strength <= 0:
            continue

        # ==============================================
        # Secondary admissibility condition
        # ==============================================

        left_extremum = (
            (u[j] - u[j - 1]) *
            (u[j - 1] - u[j - 2]) < 0
        )

        if left_extremum:

            left_strength = abs(
                u[j - 1] - u[j - 2]
            )

            if extremum_strength > left_strength:
                continue

        # ==============================================
        # Choose correction direction
        # ==============================================

        if abs(forward_jump) > abs(backward_jump):

            delta_plus  = abs(forward_jump)
            delta_minus = abs(backward_jump)

            correction_index = j + 1

        else:

            delta_plus  = abs(backward_jump)
            delta_minus = abs(forward_jump)

            correction_index = j - 1

        # TVD restriction
        delta = min(delta_plus / 2.0, delta_minus)

        # Additional admissibility restriction
        delta = min(delta, extremum_strength)

        if delta <= 0:
            continue

        # Correction sign
        s = np.sign(forward_jump)

        # Conservative redistribution
        uf[j] += s * delta

        uf[correction_index] -= s * delta

    return uf
