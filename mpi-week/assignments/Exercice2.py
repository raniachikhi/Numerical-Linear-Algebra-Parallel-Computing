from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

while True:
    if rank == 0:
        value = int(input("Enter an integer value: "))
        comm.bcast(value, root=0)
        if value < 0:
            break
    else:
        value = comm.bcast(None, root=0)
        if value < 0:
            break
    print(f"Process {rank} got {value}")
    comm.Barrier()
