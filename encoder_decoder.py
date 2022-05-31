import numpy as np
from scipy.interpolate import lagrange

from utils import time_section

@time_section
def poly_encode(A, B, n_workers, r, s, t, p, m, n):
    """Master node encodes A into A_i, B_i for each worker i, 
       as in the polynomial coding paper."""
    A_subrows, A_subcols = int(np.rint(s/p)), int(np.rint(r/m))
    B_subrows, B_subcols = int(np.rint(s/p)), int(np.rint(t/n))

    # each worker stores
    #   \tilde{A}_i = \sum_{j=0}^{p-1} \sum_{k=0}^{m-1} A_{j, k}x_i^{j+kp}
    #   \tilde{B}_i = \sum_{j=0}^{p-1} \sum_{k=0}^{m-1} B_{j, k}x_i^{p-1-j+kpm}

    # TODO: lots of improvements can be made here; for instance:
    #   1. binary exponentiation on x_i rather than recomputing powers over and over

    all_A_i = np.zeros(shape=(n_workers, A_subrows, A_subcols), dtype='int')
    all_B_i = np.zeros(shape=(n_workers, B_subrows, B_subcols), dtype='int')

    for i in range(n_workers):
        for j in range(p):
            # start and end rows of A_{j, k} and B_{j, k}
            A_rowstart, A_rowend = j*A_subrows, (j+1)*A_subrows
            B_rowstart, B_rowend = j*B_subrows, (j+1)*B_subrows

            for k in range(m):
                A_colstart, A_colend = k*A_subrows, (k+1)*A_subcols
                # block A_{j, k} 
                all_A_i[i] += A[A_rowstart : A_rowend,
                                A_colstart : A_colend] * np.power(i, j + k*p)
            for k in range(n):
                B_colstart, B_colend = k*B_subrows, (k+1)*B_subcols
                all_B_i[i] += B[B_rowstart : B_rowend,
                                B_colstart : B_colend] * np.power(i, p-1-j + k*p*m)

    return all_A_i, all_B_i


@time_section
def poly_decode(needed_C_i, needed_x_i, A_subcols, B_subcols, r, t, p, m, n):
    n_needed = len(needed_C_i)
    A_subcols, B_subcols = int(np.rint(r/m)), int(np.rint(t/n))

    C = np.empty(shape=(r, t), dtype='int')

    # C_{k, k1} is the coefficient of the (p-1 + k*p + k1*p*m)th degree term
    # do r/m * t/n interpolations (these are the dimensions of the C_i)
    for a in range(A_subcols):
        for b in range(B_subcols):
            curr_vals = [needed_C_i[i][a][b] for i in range(n_needed)]
            # TODO: fast implementation of polynomial interpolation with
            #   x values of needed_x_i and y values of curr_values
            # Gathen Gerhard describes an O(k log^2 k loglog k) algorithm
            curr_polynomial = np.rint(lagrange(needed_x_i, curr_vals).c)
            for k in range(m):
                for k1 in range(n):
                    C[k*A_subcols+a][k1*B_subcols+b] \
                        = curr_polynomial[n_needed - 1 - (p-1 + k*p + k1*p*m)]
    
    return C
