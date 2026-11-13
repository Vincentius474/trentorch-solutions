import numpy as np


def gaussian_eliminate(A, b):
    """
    A: a square (n, n) NumPy array that does not require row swaps
    b: a length-n NumPy array

    Returns:
        (U, c): U is A row-reduced to upper-triangular form, c is b
        transformed by the same row operations, such that Ux = c has
        the same solution as Ax = b.
    """
    # TODO: Implement elimination on copies of A and b in lockstep,
    # from Theory.
    # Work on copies to avoid modifying the original inputs
    U = A.astype(float).copy()
    c = b.astype(float).copy()
    n = len(c)

    # Forward elimination
    for i in range(n):
        # Check for zero pivot
        if np.isclose(U[i, i], 0.0):
            raise np.linalg.LinAlgError("Zero pivot encountered. Row swapping required.")

        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            # Apply row operation to A
            U[j, :] -= factor * U[i, :]
            # Apply the same row operation to b
            c[j] -= factor * c[i]

    return U, c


def back_substitute(U, c):
    """
    U: an upper-triangular (n, n) NumPy array
    c: a length-n NumPy array

    Returns:
        x, the solution to Ux = c, solved from the last row up.
    """
    # TODO: Implement back substitution from Theory's formula.
    n = len(c)
    # Initialize solution vector
    x = np.zeros(n, dtype=float)

    # Solve from bottom row to top row
    for i in range(n - 1, -1, -1):
        if np.isclose(U[i, i], 0.0):
            raise np.linalg.LinAlgError("Zero diagonal element. No unique solution.")
        # Subtract contributions from already solved variables
        summation = np.dot(U[i, i + 1:], x[i + 1:])
        # Solve for the current variable
        x[i] = (c[i] - summation) / U[i, i]

    return x
