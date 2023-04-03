#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:20:22 2020

@author: kissami
"""
import numpy as np
from scipy.sparse import lil_matrix
from numpy.random import rand, seed
from numba import njit
from mpi4py import MPI


''' This program compute parallel csc matrix vector multiplication using mpi '''

COMM = MPI.COMM_WORLD
nbOfproc = COMM.Get_size()
RANK = COMM.Get_rank()

seed(42)
np.seterr(all='raise')
@njit # using numba jit to speed up the calculation
def matrixVectorMult(A, b, x):
    row, col = A.shape
    for i in range(row):
        a = A[i]
        for j in range(col):
            x[i] += a[j] * b[j]

    return 0

########################initialize matrix A and vector b ######################
#matrix sizes
SIZE = 1000
Local_size = SIZE // nbOfproc 

# create sendcounts array to indicate how many rows each process will get
sendcounts = np.repeat(Local_size, nbOfproc)
# set the last process to handle the remaining rows (if any)
sendcounts[-1] = SIZE - Local_size * (nbOfproc - 1)

# create displacements array to indicate the starting index of each block of rows
displacements = np.zeros(nbOfproc, dtype=int)
displacements[1:] = np.cumsum(sendcounts[:-1])

if RANK == 0:
    # initialize the global matrix and vector
    A = lil_matrix((SIZE, SIZE))
    A[0, :100] = rand(100)
    A[1, 100:200] = A[0, :100]
    A.setdiag(rand(SIZE))
    A = A.toarray()
    b = rand(SIZE)
    print("The condition number of A is", np.linalg.cond(A))
else :
    A = None
    b = None

# scatter b to all processes and scatter A (each process has its own local matrix)
LocalMatrix = np.empty((Local_size, SIZE), dtype=np.double if RANK != 0 else A.dtype)
COMM.Scatterv([A, sendcounts, displacements, MPI.DOUBLE], LocalMatrix, root=0)
b = COMM.bcast(b, root=0)
# compute A*b locally
LocalX = np.zeros(Local_size)
start = MPI.Wtime()
matrixVectorMult(LocalMatrix, b, LocalX)
stop = MPI.Wtime()

# gather the results into X
if RANK == 0:
    print("CPU time of parallel multiplication is ", (stop - start)*1000)

##################Gather te results ###########################################

if RANK == 0:
    X = np.zeros(SIZE, dtype=np.float64)
else:
    X = None
COMM.Gatherv(LocalX, [X, sendcounts, displacements, MPI.DOUBLE], root=0)
##################Print the results ###########################################

if RANK == 0 :
    X_ = A.dot(b)
    print("The result of A*b using dot is :", np.max(X_ - X))
   # print("The result of A*b using parallel version is :", X)
