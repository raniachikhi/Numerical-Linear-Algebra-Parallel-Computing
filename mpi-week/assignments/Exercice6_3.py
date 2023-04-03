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

# Calculate the sum for this process's slice
s = 0.0
for i in range(start, stop):
    s += 1.0/(1.0 + ((i - 0.5)/N)**2)

# Print the partial sum for each process
print("Partial sum for process {} = {}".format(rank, s))

# Send the partial sum to the controller (rank 0)
if rank != 0:
    comm.send(s, dest=0)
else:
    total_sum = s
    for i in range(1, size):
        partial_sum = comm.recv(source=i)
        total_sum += partial_sum


# Print the result for the controller (rank 0)
if rank == 0:
    pi = 4.0 * total_sum / N
    print("Approximation of pi: {:.10f}".format(pi))


# Record the time taken for the computation
t1 = MPI.Wtime()

# Repeat the computation several times to get a sensible time measurement
num_runs = 1000
for i in range(num_runs):
    # Each process calculates its own slice of the sum
    slice_size = N // size
    start = rank * slice_size
    stop = (rank + 1) * slice_size
    if rank == size - 1:
        # Last process also includes the remainder
        stop = N


    # Calculate the sum for this process's slice
    s = 0.0
    for i in range(start, stop):
        s += 1.0/(1.0 + ((i - 0.5)/N)**2)

    # Send the partial sum to the controller (rank 0)
    if rank != 0:
        comm.send(s, dest=0)
    else:
        total_sum = s
        for i in range(1, size):
            partial_sum = comm.recv(source=i)
            total_sum += partial_sum

    # Print the result for the controller (rank 0)
    if rank == 0 and i == num_runs - 1:                                         
        pi = 4.0 * total_sum / N
        print("Approximation of pi: {:.10f}".format(pi))                                                                                                                                                                                                    # Record the time taken for the computation                                     
t2 = MPI.Wtime()                                                                                                                                    
# Print the time taken for the computation                                      
if rank == 0:
    print("Time taken for computation: {:.6f} seconds".format(t2 - t1))

#The time taken to calculate $\pi$ may or may not decrease as we increase the number of processes, depending on the size of N and the efficiency of the parallel implementation.
