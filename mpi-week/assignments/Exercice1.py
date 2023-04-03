from mpi4py import MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

if RANK==0:
            print("Hello World, the rank is ",RANK,"and the desired size is ", SIZE)
