from __future__ import annotations

from fractions import Fraction


def row_reduce_fraction(
    M: list[list[int | Fraction]],
    n_rows: int,
    n_cols: int,
) -> tuple[list[list[Fraction]], list[int], int]:
    """Reduced row echelon form over the rationals.

    Returns (rref_matrix, pivot_columns, rank).
    """
    Mf = [[Fraction(M[i][j]) for j in range(n_cols)] for i in range(n_rows)]
    pivot_cols: list[int] = []
    rp = 0

    for col in range(n_cols):
        # Find pivot
        piv = None
        for r in range(rp, n_rows):
            if Mf[r][col] != 0:
                piv = r
                break
        if piv is None:
            continue

        # Swap pivot row into position
        Mf[rp], Mf[piv] = Mf[piv], Mf[rp]
        pivot_cols.append(col)

        # Scale pivot row
        scale = Mf[rp][col]
        for c in range(n_cols):
            Mf[rp][c] /= scale

        # Eliminate column in all other rows
        for r in range(n_rows):
            if r != rp and Mf[r][col] != 0:
                factor = Mf[r][col]
                for c in range(n_cols):
                    Mf[r][c] -= factor * Mf[rp][c]

        rp += 1

    return Mf, pivot_cols, len(pivot_cols)


def exact_rank(
    M: list[list[int | Fraction]],
    n_rows: int,
    n_cols: int,
) -> int:
    """Exact rank of an integer/rational matrix via Gaussian elimination."""
    _, _, rank = row_reduce_fraction(M, n_rows, n_cols)
    return rank
