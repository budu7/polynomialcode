import argparse
import numpy as np
from mpi4py import MPI
from scipy.interpolate import lagrange

MAX = 10

def main(args):
    """Main logic."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    status = MPI.Status()

    if rank == 0:
        A = np.random.randint(MAX, size=(args.s, args.r))
        B = np.random.randint(MAX, size=(args.s, args.t))
        print(f"Matrix A: \n{A}")
        print(f"Matrix B: \n{B}")
        print(f"Desired result C: \n{A.T @ B}")
    else:
        A = np.empty(shape=(args.s, args.r), dtype='int')
        B = np.empty(shape=(args.s, args.t), dtype='int')
    
    comm.Bcast([A, MPI.INT], root=0)
    comm.Bcast([B, MPI.INT], root=0)

    # TODO: A_subrows and B_subrows are redundant but this is more readable
    A_subrows, A_subcols = int(np.rint(args.s/args.p)), int(np.rint(args.r/args.m))
    B_subrows, B_subcols = int(np.rint(args.s/args.p)), int(np.rint(args.t/args.n))

    if rank != 0:
        A_i = np.zeros(shape=(A_subrows, A_subcols), dtype='int')
        B_i = np.zeros(shape=(A_subrows, B_subcols), dtype='int')

        # each worker stores
        #   \tilde{A}_i = \sum_{j=0}^{p-1} \sum_{k=0}^{m-1} A_{j, k}x_i^{j+kp}
        #   \tilde{B}_i = \sum_{j=0}^{p-1} \sum_{k=0}^{m-1} B_{j, k}x_i^{p-1-j+kpm}

        # TODO: lots of improvements can be made here; for instance:
        #   1. taking j*subrows:(j+1)*subrows into the outer loop to avoid recomputation
        #   2. binary exponentiation on x_i rather than recomputing powers over and over
        for j in range(args.p):
            # start and end rows of A_{j, k} and B_{j, k}
            for k in range(args.m):
                # block A_{j, k}
                A_i += A[j*A_subrows:(j+1)*A_subrows, 
                         k*A_subcols:(k+1)*A_subcols] * np.power(rank, j + k*args.p)
            for k in range(args.n):
                # block B_{j, k}
                B_i += B[j*B_subrows:(j+1)*B_subrows,
                         k*B_subcols:(k+1)*B_subcols] * np.power(rank, args.p-1-j+k*args.p*args.m)

        # non-blocking send; returns a Request
        C_i = A_i.T @ B_i
        req = comm.Isend(C_i, dest=0)
        req.Wait()
        print(f"Worker {rank} has completed, returning \n{C_i}.")
    else:
        # master waits for pmn + p - 1 workers to return results
        n_needed = args.p*args.m*args.n + args.p - 1

        n_completed = 0
        needed_C_i = [np.empty(shape=(A_subcols, B_subcols), dtype='int') for i in range(n_needed)]
        needed_x_i = np.empty(n_needed, dtype='int')
        while n_completed < n_needed:
            # TODO: this is blocking; figure out a non-blocking implementation with Irecv
            comm.Recv(needed_C_i[n_completed], source=MPI.ANY_SOURCE, status=status)
            needed_x_i[n_completed] = status.Get_source()
            n_completed += 1

        print(f"Master has received all necessary matrices \n{needed_C_i}.")
        print(f"which were received from the workers with the following ranks, respectively \n{needed_x_i}")

        # TODO: write the polynomial interpolation

        C = np.empty(shape=(args.r, args.t), dtype='int')

        # C_{k,k1} is the coefficient of the (p-1+k*p+k1*p*m)-th degree term
        # do r/m * t/n interpolations (these are dimensions of the C_i)
        for a in range(A_subcols):
            for b in range(B_subcols):
                curr_vals = [needed_C_i[i][a][b] for i in range(n_needed)]
                # TODO: fast implementation of polynomial interpolation with x values of needed_x_i and y values of curr_vals
                # Gathen Gerhard describes an O(klog^2kloglogk) algorithm
                curr_polynomial = np.rint(lagrange(needed_x_i, curr_vals).c)
                for k in range(args.m):
                    for k1 in range(args.n):
                        C[k*A_subcols+a][k1*B_subcols+b] = curr_polynomial[n_needed-1-(args.p-1+k*args.p+k1*args.p*args.m)]
        print(f"Decoded C:\n{C}")


def get_args():
    """Edit code here to add more command-line arguments with argparse."""
    parser = argparse.ArgumentParser(description="Straggler mitigation.")
    #-----JUST EDIT HERE-----#
    parser.add_argument("r",
                        help="Number of rows for A.T",
                        type=int)
    parser.add_argument("s",
                        help="Number of cols for A.T and number of rows for B",
                        type=int)
    parser.add_argument("t",
                        help="Number of cols for B",
                        type=int)
    parser.add_argument("p",
                        help="Number of rows of submatrices of A and B",
                        type=int)
    parser.add_argument("m",
                        help="Number of columns of submatrices of A",
                        type=int)
    parser.add_argument("n",
                        help="Number of columns of submatrices of B",
                        type=int)
    parser.add_argument("--max",
                        help="Maximal value of element of A, B",
                        type=int)
    #-----STOP HERE----------#
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
