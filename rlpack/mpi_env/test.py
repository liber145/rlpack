from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

recv_data = None

if rank == 0:
    send_data = range(4)
    print("process {} scatter data {} to other processes".format(rank, send_data))
else:
    send_data = None

recv_data = comm.scatter(send_data, root=0)
print("process {} recv data {}...".format(rank, recv_data))
