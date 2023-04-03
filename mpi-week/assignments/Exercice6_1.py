import numpy as np
from mpi4py import MPI

# Set N to 840
N = 840

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Each process calculates its own slice of the sum
slice_size = N // size
start = rank * slice_size
stop = (rank + 1) * slice_size
if rank == size - 1:
    # Last process also includes the remainder
    stop = N

# Initialize the sum for this process's slice
s = 0.0

# Calculate the sum for this process's slice
for i in range(start, stop):
    s += 1.0/(1.0 + ((i - 0.5)/N)**2)

# Reduce the sums across all processes
total_sum = comm.reduce(s, op=MPI.SUM, root=0)

# Print the result for each process
if rank == 0:
    pi = 4.0 * total_sum / N
    print("Approximation of pi: {:.10f}".format(pi))
