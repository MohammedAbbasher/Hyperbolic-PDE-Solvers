# -*- coding: utf-8 -*-
"""
==============================================================
WENO utility functions
==============================================================

This file contains auxiliary mathematical routines used by
the WENO reconstruction framework.

Included routines:
    - harmonic_number
    - compute_reconstruction_weights
    - compute_linear_weights
    - compute_lagrange_weights
    - legendre_gauss_quadrature
    - compute_smoothness_entry
    - compute_smoothness_matrix
==============================================================
"""

import numpy as np
from math import factorial


from functools import lru_cache
from numpy.polynomial.legendre import leggauss

# ============================================================
# Harmonic numbers
# ============================================================

MAX_ORDER = 10

FACTORIAL_TABLE = np.array(
    [factorial(i) for i in range(MAX_ORDER + 1)],
    dtype=np.float64
)

def harmonic_number(n):
    """
    Compute harmonic number H_n
    """

    if n <= 0:
        return 0.0

    return np.sum(1.0 / np.arange(1, n + 1))


# ============================================================
# Reconstruction weights
# ============================================================

def compute_reconstruction_weights(order, shift):
    """
    Compute reconstruction weights.

    v_{j+1/2} = sum(c_ir * v_{i-r+j})

    Parameters
    ----------
    order : int
        Reconstruction order.

    shift : int
        Stencil shift (-1 <= shift <= order-1).

    Returns
    -------
    weights : ndarray
        Reconstruction weights.
    """

    weights = np.zeros(order)

    shift_index = shift + 1

    # ========================================================
    # Compute factorial-based coefficients
    # ========================================================

    stencil_indices = np.arange(1, order + 1)

    factorial_weights = (
        (-1.0) ** (stencil_indices + order)
        * FACTORIAL_TABLE[stencil_indices]
        * FACTORIAL_TABLE[order - stencil_indices]
    )

    reference_weight = (
        (-1.0) ** (shift_index + order)
        * factorial(shift_index)
        * factorial(order - shift_index)
    )

    # ========================================================
    # Compute reconstruction weights
    # ========================================================

    for i in range(order):

        q_values = np.arange(i + 1, order + 1)

        valid_mask = q_values != shift_index

        valid_indices = q_values[valid_mask]

        weights[i] = np.sum(
            reference_weight
            / factorial_weights[valid_indices - 1]
            / (shift_index - valid_indices)
        )

        # Harmonic contribution
        if shift_index in q_values:

            weights[i] -= (
                harmonic_number(order - shift - 1)
                - harmonic_number(shift_index)
            )

    return weights


# ============================================================
# Linear weights
# ============================================================

def compute_linear_weights(order, stencil_shift):
    """
    Compute optimal linear weights for WENO.

    Parameters
    ----------
    order : int
        WENO order.

    stencil_shift : int
        Shift for upwind stencil.

    Returns
    -------
    linear_weights : ndarray
        Linear WENO weights.
    """

    coefficient_matrix = np.zeros((order, order))

    rhs_vector = np.zeros(order)

    # ========================================================
    # Build coefficient matrix
    # ========================================================

    for i in range(1, order + 1):

        reconstruction_column = compute_reconstruction_weights(
            order,
            i - 1 + stencil_shift
        )

        coefficient_matrix[
            0:(order + 1 - i),
            i - 1
        ] = reconstruction_column[i - 1:order]

    # ========================================================
    # Build RHS vector
    # ========================================================

    rhs_weights = compute_reconstruction_weights(
        2 * order - 1,
        order - 1 + stencil_shift
    )

    rhs_vector = rhs_weights[order - 1:(2 * order - 1)]

    linear_weights = np.linalg.solve(
        coefficient_matrix,
        rhs_vector
    )

    return linear_weights


# ============================================================
# Lagrange polynomial weights
# ============================================================

def compute_lagrange_weights(grid_points):
    """
    Compute Taylor expansion weights for Lagrange polynomials.

    Based on Fornberg's method:
    SIAM Review (1998), 685-691
    """

    number_of_points = len(grid_points)

    lagrange_coefficients = np.zeros(
        (number_of_points, number_of_points)
    )

    lagrange_coefficients[0, 0] = 1.0

    scaling_factor = 1.0

    current_x = grid_points[0]

    for i in range(1, number_of_points):

        max_order = min(i + 1, number_of_points - 1) + 1

        product_term = 1.0

        previous_x = current_x

        current_x = grid_points[i]

        for j in range(i):

            distance = grid_points[i] - grid_points[j]

            product_term *= distance

            # =================================================
            # Update newest row
            # =================================================

            if j == i - 1:

                derivative_indices = np.arange(
                    max_order - 1,
                    0,
                    -1
                )

                lagrange_coefficients[i, derivative_indices] = (
                    scaling_factor
                    * (
                        derivative_indices
                        * lagrange_coefficients[
                            i - 1,
                            derivative_indices - 1
                        ]
                        - previous_x
                        * lagrange_coefficients[
                            i - 1,
                            derivative_indices
                        ]
                    )
                    / product_term
                )

                lagrange_coefficients[i, 0] = (
                    -scaling_factor
                    * previous_x
                    * lagrange_coefficients[i - 1, 0]
                    / product_term
                )

            # =================================================
            # Update previous rows
            # =================================================

            derivative_indices = np.arange(
                max_order - 1,
                0,
                -1
            )

            lagrange_coefficients[j, derivative_indices] = (
                current_x
                * lagrange_coefficients[j, derivative_indices]
                - derivative_indices
                * lagrange_coefficients[
                    j,
                    derivative_indices - 1
                ]
            ) / distance

            lagrange_coefficients[j, 0] = (
                current_x
                * lagrange_coefficients[j, 0]
                / distance
            )

        scaling_factor = product_term

    return lagrange_coefficients


@lru_cache(maxsize=None)
def compute_legendre_gauss_quadrature(order):

    x, w = leggauss(order + 1)

    return x, w


# ============================================================
# Smoothness indicator entry
# ============================================================

def compute_smoothness_entry(
        derivative_matrix,
        weno_order,
        derivative_order
):
    """
    Compute smoothness indicator matrix entry.
    """

    quadrature_points, quadrature_weights = (
        compute_legendre_gauss_quadrature(weno_order)
    )

    scaled_points = quadrature_points / 2.0

    smoothness_entry = 0.0

    for i in range(0, weno_order + 1):

        basis_vector = np.zeros(
            weno_order - derivative_order + 1
        )

        for k in range(
                0,
                weno_order - derivative_order + 1
        ):

            if k == 0:

                denominator = 1.0

            else:

                denominator = factorial(k)

            basis_vector[k] = (
                scaled_points[i] ** k
            ) / denominator

        smoothness_entry += (
            basis_vector.T
            @ derivative_matrix
            @ basis_vector
        ) * quadrature_weights[i] / 2.0

    return smoothness_entry


# ============================================================
# Smoothness indicator matrix
# ============================================================

def compute_smoothness_matrix(stencil_points, order):
    """
    Compute smoothness indicator matrix for WENO.
    """

    lagrange_coefficients = (
        compute_lagrange_weights(stencil_points)
    )

    smoothness_matrix = np.zeros((order, order))

    for derivative_order in range(2, order + 1):

        derivative_weights = np.zeros(
            (
                order,
                order - derivative_order + 1
            )
        )

        # =====================================================
        # Compute derivative weights
        # =====================================================

        for k in range(
                0,
                order - derivative_order + 1
        ):

            for q in range(0, order):

                derivative_weights[q, k] = np.sum(
                    lagrange_coefficients[
                        (q + 1):order + 1,
                        k + derivative_order
                    ]
                )

        # =====================================================
        # Build local smoothness matrix
        # =====================================================

        local_smoothness_matrix = np.zeros(
            (order, order)
        )

        for p in range(0, order):

            for q in range(0, order):

                derivative_matrix = np.outer(
                    derivative_weights[q, :],
                    derivative_weights[p, :]
                )

                local_smoothness_matrix[p, q] = (
                    compute_smoothness_entry(
                        derivative_matrix,
                        order,
                        derivative_order
                    )
                )

        smoothness_matrix += local_smoothness_matrix

    return smoothness_matrix