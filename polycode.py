import argparse
import time
import numpy as np
from mpi4py import MPI
from scipy.interpolate import lagrange

import encoder_decoder as ed

# ---------GLOBALS----------
MAX        = 10
SLEEP_TIME = 1  # in seconds
# --------------------------

def main(args):
    """Main logic."""
    # -----CHANGE GLOBALS HERE-----
    if args.max is not None:
        MAX = args.max
    # -----------------------------

    comm = MPI.COMM_WORLD
    n_workers = comm.Get_size()
    rank = comm.Get_rank()
    status = MPI.Status()
    stragglers = None if args.stragglers == None else \
                 set(np.random.default_rng().choice(np.arange(start=1, stop=n_workers),
                                                    size=args.stragglers,
                                                    replace=False))

    all_A_i, all_B_i = None, None

    if rank == 0:
        A = np.random.randint(MAX, size=(args.s, args.r))
        B = np.random.randint(MAX, size=(args.s, args.t))

        print(f"Matrix A: \n{A}")
        print(f"Matrix B: \n{B}")
        print(f"Desired result C: \n{A.T @ B}")

        all_A_i, all_B_i = ed.poly_encode(A, B, 
                                          n_workers,
                                          args.r, args.s, args.t, args.p, args.m, args.n)

    # TODO: A_subrows and B_subrows are redundant but this is more readable
    A_subrows, A_subcols = int(np.rint(args.s/args.p)), int(np.rint(args.r/args.m))
    B_subrows, B_subcols = int(np.rint(args.s/args.p)), int(np.rint(args.t/args.n))

    A_i = np.empty(shape=(A_subrows, A_subcols), dtype='int')
    B_i = np.empty(shape=(B_subrows, B_subcols), dtype='int')

    comm.Scatter(all_A_i, A_i, root=0)
    comm.Scatter(all_B_i, B_i, root=0)

    if rank != 0:
        C_i = A_i.T @ B_i
        # straggler functionality
        if stragglers is not None and rank in stragglers:
            time.sleep(SLEEP_TIME)
        # non-blocking send; returns a Request
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

        C = ed.poly_decode(needed_C_i,
                           needed_x_i,
                           A_subcols,
                           B_subcols,
                           args.r, args.t, args.p, args.m, args.n)
        
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
    parser.add_argument("--stragglers",
                        help="Randomly add k artificial stragglers.",
                        type=int)
    #-----STOP HERE----------#
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
