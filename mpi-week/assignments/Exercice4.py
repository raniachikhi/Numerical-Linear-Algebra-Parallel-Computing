import numpy as np
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD

# Get number of processes and current process rank
num_procs = comm.Get_size()
rank = comm.Get_rank()

# Define matrix dimensions
n = 8
m = 8

# Create matrix A on process 0
if rank == 0:
#    n = 8
 #   m = 8
    A = np.random.rand(n, m)
else:
    A = None

# Scatter matrix A to all processes using MPI_Scatterv
counts = [n//2 * m, n//2 * m, (n - n//2) * (m - m//2), (n - n//2) * m//2]
displs = [0, n//2 * m, n//2 * m + (n - n//2) * m//2, n//2 * m + (n - n//2) * m//2 + (n - n//2) * m//2]
recvbuf = np.empty(counts[rank], dtype=np.float64)
comm.Scatterv([A, counts, displs, MPI.DOUBLE], recvbuf, root=0)

# Print the part of the matrix received by each process
print(f"Process {rank} received:")
if rank == 1:
    for i in range(n//2):
        for j in range(m//2, m):
            print(f"A({i},{j}) = {recvbuf[i*(m-m//2)+(j-m//2)]}")
elif rank == 2:
    for i in range(n//2, n):
        for j in range(m//2):
            print(f"A({i},{j}) = {recvbuf[(i-n//2)*m//2+j]}")
elif rank == 3:
    for i in range(n//2, n):
        for j in range(m//2, m):
            print(f"A({i},{j}) = {recvbuf[(i-n//2)*(m-m//2)+(j-m//2)]}")
