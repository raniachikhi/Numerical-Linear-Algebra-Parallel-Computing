from mpi4py import MPI
import math
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Get input value of N from user
if rank == 0:
    N = int(input("Enter a value for N: "))
else:
    N = None

# Broadcast N to all processes
N = comm.bcast(N, root=0)

# Calculate the range of i values to be computed by each process
local_N = math.ceil(N / size)
i_vals = np.linspace(0, N - 1, N, dtype=int)
start_i = rank * local_N
end_i = min(start_i + local_N, N)

# Compute the partial sum for each process's range of i values
s = 0.0
for i in range(start_i, end_i):
    s += 1.0 / (1.0 + ((i + 0.5) / N) ** 2)

# Reduce the sums across all processes
total_sum = comm.reduce(s, op=MPI.SUM, root=0)

# Print the result for each process
if rank == 0:
    pi = 4.0 / N * total_sum
    print("pi = {:.16f}".format(pi))
