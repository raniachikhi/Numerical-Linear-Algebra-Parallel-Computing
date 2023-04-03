import numpy as np
from mpi4py import MPI

# Set N to 840
N = 840

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Divide the range of i among processes
start = rank * (N // size)
stop = (rank + 1) * (N // size)
if rank == size - 1:
    stop = N

# Compute the sum for this process's range of i
s = 0.0
for i in range(start, stop):
    s += 1.0 / (1.0 + ((i - 0.5) / N)**2)

# Reduce the sums across all processes
total_sum = comm.reduce(s, op=MPI.SUM, root=0)

# Print the result for process 0
if rank == 0:
    pi = 4.0 * total_sum / N
    print("Approximation of pi: {:.10f}".format(pi))
