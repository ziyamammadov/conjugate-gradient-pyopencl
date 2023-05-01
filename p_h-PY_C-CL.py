#!/usr/bin/env python3
"""
Add 4 arguments: subdomain_width number_of_subdomains UseCG [CGMaxIT]
  subdomain width: -- number of subdomain nodes in each direction (without overlap)
  nmber_or_subdomains: -- total number of subdomains is the square of this number
  UseCG: -- the version of the subdomain solver to use:
         0 - Using EXACT SubSolves
         1 - GPGPU PyCL-CG with Single RHS SubSolves
         2 - GPGPU PyCL-CG with Multiple RHS SubSolves
         3 - NumPy-CG SubSolves
         11 - GPGPU C_CL-CG with Single RHS SubSolves
         12 - C_CL-CG with Multiple RHS SubSolves
  CGMaxIT: -- # of CG iterations to perform in each subdomain solve [default 256]

Example of usage:
  $ mpirun -n 1 python3 p_h-PY_C-CL.py 3 64 2 256
"""
import math
import sys
from ctypes import *
from time import time

import numpy as np
from numpy.ctypeslib import ndpointer
import scipy.sparse
import scipy.sparse as sparse
import scipy.io as sio
import scipy.sparse.linalg
from mpi4py import MPI
from numpy import array, random, zeros, ones, arange, dot, vdot, sqrt, real, exp, conjugate, concatenate, empty, ravel, \
    meshgrid
from scipy.sparse.linalg import aslinearoperator
import matplotlib.pyplot as plt
import pandas as pd
from helmFE_var import CG
import cl as pcl

libcg = CDLL("./build/liboclcg.so")
libcg.connect()

def helm_fe(N, k, eps):
    global DomainProc, SubDomain, nprocs, comm, rank, globtag, maxtag
    global GLOBALS, OL  # GLOBALS here assumed in ravelled form
    #  helmFE.m
    #
    #   Discretises the BVP,
    #
    #   -\Delta u - (k^2+i\eps) u = f, on \Omega=(0,1)^2
    #            \partial_{n} u -ik u = 0 , on \partial\Omega
    #
    # using piecewise linear finite elements
    #  Input:
    #
    #   N= number of grid points in each direction
    #   k= wavenumber
    #   eps=damping parameter
    #
    #  output:
    #
    #   S = finite element matrix
    #
    #  Summary:
    #
    # Discretises Helmholtz eqn in a square (using Finite Element method) with
    # Robin conditions on all sides.

    # print '################# creating helm_fe',N,k,eps
    h = 1. / (N - 1.)  # Ivan's correction
    h2 = h ** 2
    k2 = k ** 2
    n_mysubd = SubDomain.shape[0]
    A = list(range(n_mysubd))
    # pdb.set_trace()
    OL_was = OL
    for p in range(n_mysubd):
        ni = SubDomain[p, 2] - SubDomain[p, 1]
        nj = (SubDomain[p, 4] - SubDomain[p, 3])
        nn = ni * nj
        nm = max(ni, nj)
        nnz = nn + 4 * (nm - 1) * nm + 2 * (nm - 1) ** 2
        # nnz=nnz*2 #TODO: better estimate possible?
        # inner matrix
        a_shared = zeros(nnz, dtype=int)  # x's
        b_shared = zeros(nnz, dtype=int)  # y's
        c_shared = zeros(nnz, dtype=complex)  # nz's
        # all the rest (ie. outside nodes' matrix)
        a_own = zeros(nnz, dtype=int)  # x's
        b_own = zeros(nnz, dtype=int)  # y's
        c_own = zeros(nnz, dtype=complex)  # nz's
        pos_shared = 0
        pos_own = 0
        # Determine index limits:
        down_border = SubDomain[p, 1]
        up_border = SubDomain[p, 2]
        left_border = SubDomain[p, 3]
        right_border = SubDomain[p, 4]
        # find which nodes need to be communicated to neighbours:
        shared = zeros(GLOBALS[p].shape, dtype=bool)
        if left_border > 0:
            if down_border > 0:
                strt = 1
            else:
                strt = 0
            if up_border < N:
                endt = -1
            else:
                endt = shared.shape[0]
            shared[strt:endt, 2 * OL] = True
        if right_border < N:
            if down_border > 0:
                strt = 1
            else:
                strt = 0
            if up_border < N:
                endt = -1
            else:
                endt = shared.shape[0]
            shared[strt:endt, -2 * OL - 1] = True
        if down_border > 0:
            if left_border > 0:
                strt = 1
            else:
                strt = 0
            if right_border < N:
                endt = -1
            else:
                endt = shared.shape[1]
            shared[2 * OL, strt:endt] = True
        if up_border < N:
            if left_border > 0:
                strt = 1
            else:
                strt = 0
            if right_border < N:
                endt = -1
            else:
                endt = shared.shape[1]
            shared[-2 * OL - 1, strt:endt] = True
        # pdb.set_trace()
        # form S
        jj = 0
        # if p==1: pdb.set_trace()
        for j in range(SubDomain[p, 3], SubDomain[p, 4]):  # column numbers
            mm = 0
            for m in range(SubDomain[p, 1], SubDomain[p, 2]):  # row numbers
                # Matrix S = K - k^2 M - i*k*B
                # where K= Stiffness matrix, M = Mass matrix and B = Boundary matrix
                ## Diagonal terms
                ## corners
                b_ind = a_ind = m * N + j
                if m == 0 and j == 0:
                    c_val = 1. - (
                            k2 + 1j * eps) * h2 / 6. - 1j * k * 2 * h / 3.  # Ivan second and third term on RHS corrected
                elif m == 0 and j == N - 1:
                    c_val = 1. - (k2 + 1j * eps) * h2 / 12. - 1j * k * 2 * h / 3.  # Ivan third term on RHS corrected
                elif m == N - 1 and j == 0:
                    c_val = 1. - (k2 + 1j * eps) * h2 / 12. - 1j * k * 2 * h / 3.  # Ivan third term on RHS corrected
                elif m == N - 1 and j == N - 1:
                    c_val = 1. - (
                            k2 + 1j * eps) * h2 / 6. - 1j * k * 2 * h / 3.  # Ivan second and third term on RHS corrected
                ## Diagonal terms on boundary (edge)
                elif m == 0 and j > 0 and j < N - 1:
                    c_val = 2. - (k2 + 1j * eps) * h2 / 4. - 2. * 1j * k * h / 3.
                elif m == N - 1 and j > 0 and j < N - 1:
                    c_val = 2. - (k2 + 1j * eps) * h2 / 4. - 2. * 1j * k * h / 3.
                elif m > 0 and m < N - 1 and j > 0 and j < N - 1:
                    c_val = 4. - (k2 + 1j * eps) * h2 / 2.
                elif j == 0 and m > 0 and m < N - 1:
                    c_val = 2. - (k2 + 1j * eps) * h2 / 4. - 2. * 1j * k * h / 3.
                elif j == N - 1 and m > 0 and m < N - 1:
                    c_val = 2. - (k2 + 1j * eps) * h2 / 4. - 2. * 1j * k * h / 3.
                # add the value to an array:
                if shared[mm, jj]:
                    a_shared[pos_shared] = a_ind
                    b_shared[pos_shared] = b_ind
                    c_shared[pos_shared] = c_val
                    pos_shared = pos_shared + 1
                else:
                    a_own[pos_own] = b_own[pos_own] = a_ind
                    c_own[pos_own] = c_val
                    pos_own = pos_own + 1

                ## Off diagonal On bottom and top boundary % Ivan
                if j < N - 1 and m == 0:
                    a_ind = m * N + j
                    b_ind = m * N + j + 1  # (m,j+1) case
                    c_val = -1. / 2. - (k2 + 1j * eps) * h2 / 24. - 1j * k * h / 6.
                    if j + 1 < right_border:  # not from outside
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1

                # following four lines added by Ivan % off diagonal on bottom
                if j < N - 1 and j > 0 and m == 0:
                    a_ind = j
                    b_ind = j + N + 1
                    c_val = -(k2 + 1j * eps) * h2 / 12.
                    if m + 1 < up_border and j + 1 < right_border:  # not from outside
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1

                if j < N - 1 and m == N - 1:
                    a_ind = m * N + j
                    b_ind = m * N + j + 1
                    c_val = -1. / 2. - (k2 + 1j * eps) * h2 / 24. - 1j * k * h / 6.
                    if j + 1 < right_border:  # not from outside
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1

                if j > 0 and m == 0:
                    a_ind = m * N + j
                    b_ind = m * N + j - 1
                    c_val = -1. / 2. - (k2 + 1j * eps) * h2 / 24. - 1j * k * h / 6.
                    if j - 1 >= left_border:  # not from outside
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1

                if j > 0 and m == N - 1:
                    a_ind = m * N + j
                    b_ind = m * N + j - 1
                    c_val = -1. / 2. - (k2 + 1j * eps) * h2 / 24. - 1j * k * h / 6.
                    if j - 1 >= left_border:  # not from outside
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1

                # following four lines added by Ivan % off diagonal on top
                if j > 0 and j < N - 1 and m == N - 1:
                    a_ind = m * N + j
                    b_ind = m * N + j - N - 1  # (m-1,j-1) case
                    c_val = -(k2 + 1j * eps) * h2 / 12.
                    if m - 1 >= down_border and j - 1 >= left_border:  # not from outside
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1

                ## Off diagonal on left and right boundary % Ivan
                if m < N - 1 and j == 0:
                    a_ind = m * N + j
                    b_ind = (m + 1) * N + j
                    c_val = -1. / 2. - (k2 + 1j * eps) * h2 / 24. - 1j * k * h / 6.
                    if m + 1 < up_border:  # not from outside
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1

                    # following four lines added by Ivan % off diagonal left boundary
                    a_ind = m * N + j
                    b_ind = (m + 1) * N + j + 1
                    c_val = -(k2 + 1j * eps) * h2 / 12.
                    if m + 1 < up_border:  # not from outside
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1

                if m < N - 1 and j == N - 1:
                    a_ind = m * N + j
                    b_ind = (m + 1) * N + j
                    c_val = -1. / 2. - (k2 + 1j * eps) * h2 / 24. - 1j * k * h / 6.
                    if m + 1 < up_border:  # not from outside
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1

                if m > 0 and j == 0:
                    a_ind = m * N + j
                    b_ind = (m - 1) * N + j
                    c_val = -1. / 2. - (k2 + 1j * eps) * h2 / 24. - 1j * k * h / 6.
                    if m - 1 >= down_border:  # not from outside
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1

                if m > 0 and j == N - 1:
                    a_ind = m * N + j
                    b_ind = (m - 1) * N + j
                    c_val = -1. / 2. - (k2 + 1j * eps) * h2 / 24. - 1j * k * h / 6.
                    if m - 1 >= down_border:  # not from outside
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1

                    # following four lines added by Ivan % off diagonal right boundaary
                    a_ind = m * N + j
                    b_ind = m * N + j - N - 1
                    c_val = -(k2 + 1j * eps) * h2 / 12.
                    if m - 1 >= down_border:  # and j-1>=left_border ok anyway: # not from outside
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1

                ## In the interior
                if j < N - 1 and m > 0 and m < N - 1:
                    a_ind = m * N + j
                    b_ind = m * N + j + 1
                    c_val = -1. - (k2 + 1j * eps) * h2 / 12.
                    if j + 1 < right_border:  # not from outside
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1

                if j > 0 and m > 0 and m < N - 1:
                    a_ind = m * N + j
                    b_ind = m * N + j - 1
                    c_val = -1. - (k2 + 1j * eps) * h2 / 12.
                    if j - 1 >= left_border:  # not from outside
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1

                if m < N - 1 and j > 0 and j < N - 1:
                    a_ind = m * N + j
                    b_ind = (m + 1) * N + j
                    c_val = -1. - (k2 + 1j * eps) * h2 / 12.
                    if m + 1 < up_border:  # not from outside
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1

                if m > 0 and j > 0 and j < N - 1:
                    a_ind = m * N + j
                    b_ind = (m - 1) * N + j
                    c_val = -1. - (k2 + 1j * eps) * h2 / 12.
                    if m - 1 >= down_border:  # not from outside
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1

                # Off diagonal term, zero when k=0
                if j > 0 and j < N - 1 and m > 0 and m < N - 1:
                    a_ind = m * N + j
                    b_ind = (m - 1) * N + j - 1
                    c_val = -((k2 + 1j * eps) * h2) / 12.
                    if m - 1 >= down_border and j - 1 >= left_border:  # not from outside
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1

                if j > 0 and j < N - 1 and m > 0 and m < N - 1:
                    a_ind = m * N + j
                    b_ind = (m + 1) * N + j + 1
                    c_val = -((k2 + 1j * eps) * h2) / 12.
                    if m + 1 < up_border and j + 1 < right_border:  # not from outside
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                mm = mm + 1
            jj = jj + 1
        glo2loc = {}
        ii = 0
        for ind in GLOBALS[p].flatten():
            glo2loc[ind] = ii
            ii = ii + 1
        count = 0
        for ind in a_shared[:pos_shared]:
            if ind in glo2loc:
                a_shared[count] = glo2loc[ind]
            else:
                a_shared[count] = ii
                glo2loc[ind] = ii
                ii = ii + 1
                print('SubDomain:', rank, p, ':', SubDomain[p])
                print('GLOBALS are:', GLOBALS[p].reshape((ni, nj)))
                print(rank, p, 'WARNING a_shared glo2loc adding:', ind)
                print('a_shared::', a_shared[:pos_shared])
                print('N=', N)
                exit(0)
            count = count + 1
        count = 0
        for ind in a_own[:pos_own]:
            if ind in glo2loc:
                a_own[count] = glo2loc[ind]
            else:
                a_own[count] = ii
                glo2loc[ind] = ii
                ii = ii + 1
                print(rank, p, 'WARNING a_own: glo2loc adding:', ind)
                exit(0)
            count = count + 1
        count = 0
        for ind in b_shared[:pos_shared]:
            # print p,pos_shared,'bbbb ind is:',ind,b_shared[:5]
            if ind in glo2loc:
                b_shared[count] = glo2loc[ind]
            else:
                b_shared[count] = ii
                glo2loc[ind] = ii
                ii = ii + 1
                print('ind,N,j,m,jj,mm,count=', ind, N, j, m, jj, mm, count)
                exit(0)
            count = count + 1
        count = 0
        for ind in b_own[:pos_own]:
            if ind in glo2loc:
                b_own[count] = glo2loc[ind]
            else:
                b_own[count] = ii
                glo2loc[ind] = ii
                ii = ii + 1
                print('SubDomain:', rank, p, ':', SubDomain[p])
                print('GLOBALS are:', GLOBALS[p].reshape((ni, nj)))
                print(rank, p, 'WARNING b_own: glo2loc adding:', ind)
                exit(0)
            count = count + 1
        A[p] = list(range(3))
        A[p][0] = scipy.sparse.csr_matrix((c_shared[:pos_shared], (a_shared[:pos_shared], b_shared[:pos_shared])) \
                                          , shape=(GLOBALS[p].size, GLOBALS[p].size))
        A[p][1] = scipy.sparse.csr_matrix((c_own[:pos_own], (a_own[:pos_own], b_own[:pos_own]))
                                          , shape=(GLOBALS[p].size, GLOBALS[p].size))
        a_all = concatenate((a_shared[:pos_shared], a_own[:pos_own]))
        b_all = concatenate((b_shared[:pos_shared], b_own[:pos_own]))
        c_all = concatenate((c_shared[:pos_shared], c_own[:pos_own]))
        A[p][2] = scipy.sparse.csr_matrix((c_all, (a_all, b_all)) \
                                          , shape=(GLOBALS[p].size, GLOBALS[p].size))

    OL = OL_was
    return A


def helm_fe_var(N, k, C, rho):
    global DomainProc, SubDomain, nprocs, comm, rank, globtag, maxtag
    global GLOBALS, OL  # GLOBALS here assumed in ravelled form
    #  helmFE.m
    #
    #   Discretises the BVP,
    #
    #   -\Delta u - (1+i\rho)(k^2) u = f, on \, Omega=(0,1)^2
    #            \partial_{n} u -ik u = 0 , on \partial\Omega
    #
    # where k = omega/c, and omega is the (constant) frequency while c is the (variable) wave spped
    # using piecewise linear finite elements
    #  Input:
    #
    #   N= number of grid points in each direction
    #   k= wavenumber
    #
    #  output:
    #
    #   S = finite element matrix
    #
    #  Summary:
    #
    # Discretises Helmholtz eqn in a square (using Finite Element method) with
    # Robin conditions on all sides.

    # print '################# creating helm_fe',N,k,rho
    h = 1. / (N - 1.)  # Ivan's correction
    h2 = h ** 2
    k2 = k ** 2
    n_mysubd = SubDomain.shape[0]
    A = list(range(n_mysubd))
    # pdb.set_trace()
    OL_was = OL
    for p in range(n_mysubd):
        ni = SubDomain[p, 2] - SubDomain[p, 1]
        nj = (SubDomain[p, 4] - SubDomain[p, 3])
        nn = ni * nj
        nm = max(ni, nj)
        nnz = nn + 4 * (nm - 1) * nm + 2 * (nm - 1) ** 2
        # nnz=nnz*2 #TODO: better estimate possible?
        # inner matrix
        a_shared = zeros(nnz, dtype=int)  # x's
        b_shared = zeros(nnz, dtype=int)  # y's
        c_shared = zeros(nnz, dtype=complex)  # nz's
        # all the rest (ie. outside nodes' matrix)
        a_own = zeros(nnz, dtype=int)  # x's
        b_own = zeros(nnz, dtype=int)  # y's
        c_own = zeros(nnz, dtype=complex)  # nz's
        pos_shared = 0
        pos_own = 0
        # Determine index limits:
        down_border = SubDomain[p, 1]
        up_border = SubDomain[p, 2]
        left_border = SubDomain[p, 3]
        right_border = SubDomain[p, 4]
        # find which nodes need to be communicated to neighbours:
        shared = zeros(GLOBALS[p].shape, dtype=bool)
        if left_border > 0:
            if down_border > 0:
                strt = 1
            else:
                strt = 0
            if up_border < N:
                endt = -1
            else:
                endt = shared.shape[0]
            shared[strt:endt, 2 * OL] = True
        if right_border < N:
            if down_border > 0:
                strt = 1
            else:
                strt = 0
            if up_border < N:
                endt = -1
            else:
                endt = shared.shape[0]
            shared[strt:endt, -2 * OL - 1] = True
        if down_border > 0:
            if left_border > 0:
                strt = 1
            else:
                strt = 0
            if right_border < N:
                endt = -1
            else:
                endt = shared.shape[1]
            shared[2 * OL, strt:endt] = True
        if up_border < N:
            if left_border > 0:
                strt = 1
            else:
                strt = 0
            if right_border < N:
                endt = -1
            else:
                endt = shared.shape[1]
            shared[-2 * OL - 1, strt:endt] = True
        # pdb.set_trace()
        # form S
        jj = 0
        # if p==1: pdb.set_trace()
        for j in range(SubDomain[p, 3], SubDomain[p, 4]):  # column numbers
            mm = 0
            for m in range(SubDomain[p, 1], SubDomain[p, 2]):  # row numbers
                # Matrix S = K - k^2 M - i*k*B
                # where K= Stiffness matrix, M = Mass matrix and B = Boundary matrix
                ## Diagonal terms
                ## corners
                b_ind = a_ind = m * N + j
                if m == 0 and j == 0:
                    kC = k / C[0, 0]
                    c_val = 1. - (
                            1. + 1j * rho) * kC ** 2 * h2 / 6. - 1j * kC * 2 * h / 3.  # Ivan second and third term on RHS corrected
                elif m == 0 and j == N - 1:
                    kC = C[0, N - 2]
                    c_val = 1. - (
                            1. + 1j * rho) * kC ** 2 * h2 / 12. - 1j * kC * 2 * h / 3.  # Ivan third term on RHS corrected
                elif m == N - 1 and j == 0:
                    kC = C[N - 2, 0]
                    c_val = 1. - (
                            1. + 1j * rho) * kC ** 2 * h2 / 12. - 1j * kC * 2 * h / 3.  # Ivan third term on RHS corrected
                elif m == N - 1 and j == N - 1:
                    kC = C[N - 2, N - 2]
                    c_val = 1. - (
                            1. + 1j * rho) * kC ** 2 * h2 / 6. - 1j * kC * 2 * h / 3.  # Ivan second and third term on RHS corrected
                ## Diagonal terms on boundary (edge)
                elif m == 0 and j > 0 and j < N - 1:  # bottom edge
                    klC = k / C[0, j - 1]
                    krC = k / C[0, j]  # contributions from squares to left and right
                    c_val = 2. - (1. + 1j * rho) * (klC ** 2 + 2. * krC ** 2) * h2 / 12. - 1j * (klC + krC) * h / 3.
                elif m == N - 1 and j > 0 and j < N - 1:  # top edge
                    klC = k / C[N - 2, j - 1]
                    krC = k / C[N - 2, j]
                    c_val = 2. - (1. + 1j * rho) * (2. * klC ** 2 + krC ** 2) * h2 / 12. - 1j * (klC + krC) * h / 3.
                elif m > 0 and m < N - 1 and j > 0 and j < N - 1:  # interior
                    knwC = k / C[m, j - 1]
                    kswC = k / C[m - 1, j - 1]
                    kneC = k / C[m, j]
                    kseC = k / C[m - 1, j]  # contributions from 4 surrounding squares
                    c_val = 4. - (1. + 1j * rho) * (knwC ** 2 + 2. * kswC ** 2 + 2. * kneC ** 2 + kseC ** 2) * h2 / 12.
                elif j == 0 and m > 0 and m < N - 1:  # Left hand edge
                    ktC = k / C[m, 0]
                    kbC = k / C[m - 1, 0]  # contributions from squares above and below
                    c_val = 2. - (1 + 1j * rho) * (2. * ktC ** 2 + kbC ** 2) * h2 / 12. - 1j * (ktC + kbC) * h / 3.
                elif j == N - 1 and m > 0 and m < N - 1:  # Right hand edge
                    ktC = k / C[m, N - 2]
                    kbC = k / C[m - 1, N - 2]  # contributions from squares above and below
                    c_val = 2. - (1. + 1j * rho) * (ktC ** 2 + 2. * kbC ** 2) * h2 / 12. - 1j * (ktC + kbC) * h / 3.
                # add the value to an array:
                if shared[mm, jj]:
                    a_shared[pos_shared] = a_ind
                    b_shared[pos_shared] = b_ind
                    c_shared[pos_shared] = c_val
                    pos_shared = pos_shared + 1
                else:
                    a_own[pos_own] = a_ind
                    b_own[pos_own] = b_ind
                    c_own[pos_own] = c_val
                    pos_own = pos_own + 1

                ## Off diagonal corner nodes
                if j == 0 and m == 0:  # bottom LH corner
                    kC = k / C[0, 0]
                    a_ind = 0
                    b_ind = 1  # node to the right
                    c_val = -0.5 - (1. + 1j * rho) * kC ** 2 * h2 / 24. - 1j * kC * h / 6.
                    # add the value to an array:
                    if shared[mm, jj]:
                        a_shared[pos_shared] = a_ind
                        b_shared[pos_shared] = b_ind
                        c_shared[pos_shared] = c_val
                        pos_shared = pos_shared + 1
                    else:
                        a_own[pos_own] = a_ind
                        b_own[pos_own] = b_ind
                        c_own[pos_own] = c_val
                        pos_own = pos_own + 1
                    b_ind = N
                    c_val = -0.5 - (1. + 1j * rho) * kC ** 2 * h2 / 24. - 1j * kC * h / 6.
                    # add the value to an array:
                    if shared[mm, jj]:
                        a_shared[pos_shared] = a_ind
                        b_shared[pos_shared] = b_ind
                        c_shared[pos_shared] = c_val
                        pos_shared = pos_shared + 1
                    else:
                        a_own[pos_own] = a_ind
                        b_own[pos_own] = b_ind
                        c_own[pos_own] = c_val
                        pos_own = pos_own + 1
                    b_ind = N + 1  # node above to right
                    c_val = - (1. + 1j * rho) * kC ** 2 * h2 / 12.
                    # add the value to an array:
                    if shared[mm, jj]:
                        a_shared[pos_shared] = a_ind
                        b_shared[pos_shared] = b_ind
                        c_shared[pos_shared] = c_val
                        pos_shared = pos_shared + 1
                    else:
                        a_own[pos_own] = a_ind
                        b_own[pos_own] = b_ind
                        c_own[pos_own] = c_val
                        pos_own = pos_own + 1
                if j == N - 1 and m == N - 1:  # Top RH corner
                    kC = k / C[N - 2, N - 2]
                    a_ind = N * N - 1
                    b_ind = a_ind - 1  # node to the left
                    c_val = -0.5 - (1. + 1j * rho) * kC ** 2 * h2 / 24. - 1j * kC * h / 6.
                    if shared[mm, jj]:
                        a_shared[pos_shared] = a_ind
                        b_shared[pos_shared] = b_ind
                        c_shared[pos_shared] = c_val
                        pos_shared = pos_shared + 1
                    else:
                        a_own[pos_own] = a_ind
                        b_own[pos_own] = b_ind
                        c_own[pos_own] = c_val
                        pos_own = pos_own + 1
                    b_ind = a_ind - N  # node below
                    c_val = -0.5 - (1. + 1j * rho) * kC ** 2 * h2 / 24. - 1j * kC * h / 6.
                    if shared[mm, jj]:
                        a_shared[pos_shared] = a_ind
                        b_shared[pos_shared] = b_ind
                        c_shared[pos_shared] = c_val
                        pos_shared = pos_shared + 1
                    else:
                        a_own[pos_own] = a_ind
                        b_own[pos_own] = b_ind
                        c_own[pos_own] = c_val
                        pos_own = pos_own + 1
                    b_ind = a_ind - N - 1  # node below to left
                    c_val = - (1. + 1j * rho) * kC ** 2 * h2 / 12.
                    if shared[mm, jj]:
                        a_shared[pos_shared] = a_ind
                        b_shared[pos_shared] = b_ind
                        c_shared[pos_shared] = c_val
                        pos_shared = pos_shared + 1
                    else:
                        a_own[pos_own] = a_ind
                        b_own[pos_own] = b_ind
                        c_own[pos_own] = c_val
                        pos_own = pos_own + 1
                if j == N - 1 and m == 0:  # Bottom RH corner
                    kC = k / C[0, N - 2]
                    a_ind = N - 1
                    b_ind = a_ind - 1  # node to the left
                    c_val = -0.5 - (1. + 1j * rho) * kC ** 2 * h2 / 24. - 1j * kC * h / 6.
                    if shared[mm, jj]:
                        a_shared[pos_shared] = a_ind
                        b_shared[pos_shared] = b_ind
                        c_shared[pos_shared] = c_val
                        pos_shared = pos_shared + 1
                    else:
                        a_own[pos_own] = a_ind
                        b_own[pos_own] = b_ind
                        c_own[pos_own] = c_val
                        pos_own = pos_own + 1
                    b_ind = a_ind + N  # node above
                    c_val = -0.5 - (1. + 1j * rho) * kC ** 2 * h2 / 24. - 1j * kC * h / 6.
                    if shared[mm, jj]:
                        a_shared[pos_shared] = a_ind
                        b_shared[pos_shared] = b_ind
                        c_shared[pos_shared] = c_val
                        pos_shared = pos_shared + 1
                    else:
                        a_own[pos_own] = a_ind
                        b_own[pos_own] = b_ind
                        c_own[pos_own] = c_val
                        pos_own = pos_own + 1
                if j == 0 and m == N - 1:  # Top LH corner
                    kC = k / C[N - 2, 0]
                    a_ind = (N - 1) * N
                    b_ind = a_ind + 1  # node to the right
                    c_val = -0.5 - (1. + 1j * rho) * kC ** 2 * h2 / 24. - 1j * kC * h / 6.
                    if shared[mm, jj]:
                        a_shared[pos_shared] = a_ind
                        b_shared[pos_shared] = b_ind
                        c_shared[pos_shared] = c_val
                        pos_shared = pos_shared + 1
                    else:
                        a_own[pos_own] = a_ind
                        b_own[pos_own] = b_ind
                        c_own[pos_own] = c_val
                        pos_own = pos_own + 1
                    b_ind = a_ind - N  # node below
                    c_val = -0.5 - (1. + 1j * rho) * kC ** 2 * h2 / 24. - 1j * kC * h / 6.
                    if shared[mm, jj]:
                        a_shared[pos_shared] = a_ind
                        b_shared[pos_shared] = b_ind
                        c_shared[pos_shared] = c_val
                        pos_shared = pos_shared + 1
                    else:
                        a_own[pos_own] = a_ind
                        b_own[pos_own] = b_ind
                        c_own[pos_own] = c_val
                        pos_own = pos_own + 1

                ## off diagonal - bottom boundary  nodes, not corners : node has 4 neighbours
                if j > 0 and j < N - 1 and m == 0:
                    klC = k / C[0, j - 1]  # value of k on left-hand square
                    krC = k / C[0, j]  # value of k on right-hand square
                    if j + 1 < right_border:  # not from outside
                        a_ind = j
                        b_ind = j + 1  # (m,j+1) case   # node to right
                        c_val = -0.5 - (1. + 1j * rho) * krC ** 2 * h2 / 24. - 1j * krC * h / 6.
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                    if m + 1 < up_border and j + 1 < right_border:  # not from outside
                        a_ind = j
                        b_ind = j + N + 1  # node above and to right
                        c_val = -(1 + 1j * rho) * krC ** 2 * h2 / 12.  # should be right now
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                    if m + 1 < up_border:  # not from outside
                        a_ind = j
                        b_ind = j + N  # node directly above
                        c_val = -1. - (1. + 1j * rho) * (klC ** 2 + krC ** 2) * h2 / 24.
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                    if j - 1 >= left_border:  # not from outside
                        a_ind = j
                        b_ind = j - 1  # node to the left
                        c_val = -0.5 - (1. + 1j * rho) * klC ** 2 * h2 / 24. - 1j * klC * h / 6.
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                ## off diagonal - top boundary  nodes, not corners : node has 4 neighbours
                if j > 0 and j < N - 1 and m == N - 1:
                    klC = k / C[N - 2, j - 1]  # value of k on left-hand square
                    krC = k / C[N - 2, j]  # value of k on right-hand square
                    if j + 1 < right_border:  # not from outside
                        a_ind = m * N + j
                        b_ind = m * N + j + 1  # node to the right
                        c_val = -0.5 - (1. + 1j * rho) * krC ** 2 * h2 / 24. - 1j * krC * h / 6.
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                    if j - 1 >= left_border:  # not from outside
                        a_ind = m * N + j
                        b_ind = m * N + j - 1  # node to the left
                        c_val = -0.5 - (1. + 1j * rho) * klC * h2 / 24. - 1j * klC * h / 6.
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                    if m - 1 >= down_border:  # not from outside
                        a_ind = m * N + j
                        b_ind = (m - 1) * N + j  # node directly below
                        c_val = -1. - (1. + 1j * rho) * (klC ** 2 + krC ** 2) * h2 / 24.
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                    if m - 1 >= down_border and j - 1 >= left_border:  # not from outside
                        a_ind = m * N + j
                        b_ind = (m - 1) * N + j - 1  # node below, diagonal to left
                        c_val = -(1. + 1j * rho) * klC ** 2 * h2 / 12.
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                ## Off diagonal on left and right boundary % Ivan
                if m > 0 and m < N - 1 and j == N - 1:  # on right boundary, not corners
                    kbC = k / C[m - 1, N - 2]  # value of k on bottom square
                    ktC = k / C[m, N - 2]  # value of k on top square
                    if m + 1 < up_border:  # not from outside
                        a_ind = m * N + j
                        b_ind = (m + 1) * N + j  # node above
                        c_val = -0.5 - (1. + 1j * rho) * ktC ** 2 * h2 / 24. - 1j * ktC * h / 6.
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                    if m - 1 >= down_border:  # not from outside
                        a_ind = m * N + j
                        b_ind = (m - 1) * N + j  # node below
                        c_val = -0.5 - (1. + 1j * rho) * kbC ** 2 * h2 / 24. - 1j * kbC * h / 6.
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                    a_ind = m * N + j
                    b_ind = m * N + j - 1  # node directly to the left
                    c_val = -1. - (1. + 1j * rho) * (kbC ** 2 + ktC ** 2) * h2 / 24.
                    # add the value to an array:
                    if shared[mm, jj]:
                        a_shared[pos_shared] = a_ind
                        b_shared[pos_shared] = b_ind
                        c_shared[pos_shared] = c_val
                        pos_shared = pos_shared + 1
                    else:
                        a_own[pos_own] = a_ind
                        b_own[pos_own] = b_ind
                        c_own[pos_own] = c_val
                        pos_own = pos_own + 1
                    if m - 1 >= down_border:  # not from outside
                        a_ind = m * N + j
                        b_ind = (m - 1) * N + j - 1  # node below, diagonal to left
                        c_val = -(1. + 1j * rho) * kbC ** 2 * h2 / 12.
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                if m > 0 and m < N - 1 and j == 0:  # on left boundary, not corners
                    kbC = k / C[m - 1, 0]  # value of k on bottom square
                    ktC = k / C[m, 0]  # value of k on top square
                    if m + 1 < up_border:  # not from outside
                        a_ind = m * N
                        b_ind = (m + 1) * N  # node above
                        c_val = -0.5 - (1. + 1j * rho) * ktC ** 2 * h2 / 24. - 1j * ktC * h / 6.
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                    if m - 1 >= down_border:  # not from outside
                        a_ind = m * N
                        b_ind = (m - 1) * N  # node below
                        c_val = -0.5 - (1. + 1j * rho) * kbC ** 2 * h2 / 24. - 1j * kbC * h / 6.
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                    a_ind = m * N
                    b_ind = m * N + 1  # node directly to the right
                    c_val = -1. - (1. + 1j * rho) * (kbC ** 2 + ktC ** 2) * h2 / 24.
                    # add the value to an array:
                    if shared[mm, jj]:
                        a_shared[pos_shared] = a_ind
                        b_shared[pos_shared] = b_ind
                        c_shared[pos_shared] = c_val
                        pos_shared = pos_shared + 1
                    else:
                        a_own[pos_own] = a_ind
                        b_own[pos_own] = b_ind
                        c_own[pos_own] = c_val
                        pos_own = pos_own + 1
                    if m + 1 < up_border:  # not from outside
                        a_ind = m * N
                        b_ind = (m + 1) * N + 1  # node above, diagonal to right
                        c_val = -(1. + 1j * rho) * ktC ** 2 * h2 / 12.
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1

                ## In the interior
                if j > 0 and j < N - 1 and m > 0 and m < N - 1:
                    knwC = k / C[m, j - 1]
                    kswC = k / C[m - 1, j - 1]
                    kneC = k / C[m, j]
                    kseC = k / C[m - 1, j]  # contributions from 4 surrounding squares
                    a_ind = m * N + j
                    if j + 1 < right_border:  # not from outside
                        b_ind = m * N + j + 1  # node directly at right
                        c_val = -1. - (1. + 1j * rho) * (kneC ** 2 + kseC ** 2) * h2 / 24.
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                    if j - 1 >= left_border:  # not from outside
                        b_ind = m * N + j - 1  # node directly to the left
                        c_val = -1. - (1. + 1j * rho) * (knwC ** 2 + kswC ** 2) * h2 / 24.
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                    if m + 1 < up_border:  # not from outside
                        b_ind = (m + 1) * N + j  # node directly above
                        c_val = -1. - (1. + 1j * rho) * (knwC ** 2 + kneC ** 2) * h2 / 24.
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                    if m - 1 >= down_border:  # not from outside
                        b_ind = (m - 1) * N + j  # node directly below
                        c_val = -1. - (1. + 1j * rho) * (kswC ** 2 + kseC ** 2) * h2 / 24.
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                    if m - 1 >= down_border and j - 1 >= left_border:  # not from outside
                        b_ind = (m - 1) * N + j - 1  # diagonal below to the left
                        c_val = - (1. + 1j * rho) * kswC ** 2 * h2 / 12.
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                    if m + 1 < up_border and j + 1 < right_border:  # not from outside
                        b_ind = (m + 1) * N + j + 1  # diagonal above to the right
                        c_val = - (1. + 1j * rho) * kneC ** 2 * h2 / 12.
                        # add the value to an array:
                        if shared[mm, jj]:
                            a_shared[pos_shared] = a_ind
                            b_shared[pos_shared] = b_ind
                            c_shared[pos_shared] = c_val
                            pos_shared = pos_shared + 1
                        else:
                            a_own[pos_own] = a_ind
                            b_own[pos_own] = b_ind
                            c_own[pos_own] = c_val
                            pos_own = pos_own + 1
                mm = mm + 1
            jj = jj + 1
        glo2loc = {}
        ii = 0
        for ind in GLOBALS[p].flatten():
            glo2loc[ind] = ii
            ii = ii + 1
        count = 0
        for ind in a_shared[:pos_shared]:
            if ind in glo2loc:
                a_shared[count] = glo2loc[ind]
            else:
                a_shared[count] = ii
                glo2loc[ind] = ii
                ii = ii + 1
                print('SubDomain:', rank, p, ':', SubDomain[p])
                print('GLOBALS are:', GLOBALS[p].reshape((ni, nj)))
                print(rank, p, 'WARNING a_shared glo2loc adding:', ind)
                print('a_shared::', a_shared[:pos_shared])
                print('N=', N)
                exit(0)
            count = count + 1
        count = 0
        for ind in a_own[:pos_own]:
            if ind in glo2loc:
                a_own[count] = glo2loc[ind]
            else:
                a_own[count] = ii
                glo2loc[ind] = ii
                ii = ii + 1
                print('SubDomain:', rank, p, ':', SubDomain[p])
                print('GLOBALS are:', GLOBALS[p].reshape((ni, nj)))
                print(rank, p, 'WARNING a_own: glo2loc adding:', ind)
                exit(0)
            count = count + 1
        count = 0
        for ind in b_shared[:pos_shared]:
            # print p,pos_shared,'bbbb ind is:',ind,b_shared[:5]
            if ind in glo2loc:
                b_shared[count] = glo2loc[ind]
            else:
                b_shared[count] = ii
                glo2loc[ind] = ii
                ii = ii + 1
                print('ind,N,j,m,jj,mm,count=', ind, N, j, m, jj, mm, count)
                exit(0)
            count = count + 1
        count = 0
        for ind in b_own[:pos_own]:
            if ind in glo2loc:
                b_own[count] = glo2loc[ind]
            else:
                b_own[count] = ii
                glo2loc[ind] = ii
                ii = ii + 1
                print('SubDomain:', rank, p, ':', SubDomain[p])
                print('GLOBALS are:', GLOBALS[p].reshape((ni, nj)))
                print(rank, p, 'W WARNING b_own: glo2loc adding:', ind)
                exit(0)
            count = count + 1
        A[p] = list(range(3))
        A[p][0] = scipy.sparse.csr_matrix((c_shared[:pos_shared], (a_shared[:pos_shared], b_shared[:pos_shared])) \
                                          , shape=(GLOBALS[p].size, GLOBALS[p].size))
        A[p][1] = scipy.sparse.csr_matrix((c_own[:pos_own], (a_own[:pos_own], b_own[:pos_own]))
                                          , shape=(GLOBALS[p].size, GLOBALS[p].size))
        a_all = concatenate((a_shared[:pos_shared], a_own[:pos_own]))
        b_all = concatenate((b_shared[:pos_shared], b_own[:pos_own]))
        c_all = concatenate((c_shared[:pos_shared], c_own[:pos_own]))
        A[p][2] = scipy.sparse.csr_matrix((c_all, (a_all, b_all)) \
                                          , shape=(GLOBALS[p].size, GLOBALS[p].size))

    OL = OL_was
    return A


def rhs(N, k):  # special RHS from Ivan
    global DomainProc, SubDomain, nprocs, comm, rank, globtag, maxtag
    global OshapeD, InactiveNodes
    aa = [1. / math.sqrt(2.), 1. / math.sqrt(2.)]
    aaa = array(aa).transpose()
    h = 1. / (N - 1.)
    nlocsub = SubDomain.shape[0]
    b = list(range(nlocsub))
    for p in range(nlocsub):
        b[p] = zeros(GLOBALS[p].shape, dtype=complex)
    # x = (0:M-1)/(M-1)  % grid points
    x = arange(0.0, 1.00001, 1.0 / (N - 1))
    # y = (x(2:M) + x(1:M-1))/2   % mid points
    y = (x[1:] + x[:-1]) / 2.0
    # % multipliers i*k*(a.n-1) computed on each side of boundary
    multbot = 1.j * k * (-aa[1] - 1.)
    multtop = 1.j * k * (aa[1] - 1.)
    multleft = 1.j * k * (-aa[0] - 1.)
    multright = 1.j * k * (aa[0] - 1.)
    for p in range(nlocsub):
        for j in range(SubDomain[p, 1], SubDomain[p, 2]):  # global rownumbers
            for m in range(SubDomain[p, 3], SubDomain[p, 4]):  # global colnumbers
                if j == 0 and m > 0 and m < N - 1:
                    points = array([[y[m - 1], 0.], [x[m], 0.], [y[m], 0.]])  # interior of bottom boundary
                    b[p][j - SubDomain[p, 1], m - SubDomain[p, 3]] = (h / 3.) * multbot * sum(
                        exp(1. * 1.j * k * (dot(points, aaa))))
                if j == N - 1 and m > 0 and m < N - 1:
                    points = array([[y[m - 1], 1.], [x[m], 1.], [y[m], 1.]])  # interior of top boundary
                    b[p][j - SubDomain[p, 1], m - SubDomain[p, 3]] = (h / 3.) * multtop * sum(
                        exp(1. * 1.j * k * (dot(points, aaa))))
                if m == 0 and j > 0 and j < N - 1:
                    points = array([[0., y[j - 1]], [0., x[j]], [0, y[j]]])  # interior of left boundary
                    b[p][j - SubDomain[p, 1], m - SubDomain[p, 3]] = (h / 3.) * multleft * sum(
                        exp(1. * 1.j * k * (dot(points, aaa))))
                if m == N - 1 and j > 0 and j < N - 1:
                    points = array([[y[j - 1], 1.], [x[j], 1.], [y[j], 1.]])  # interior of right boundary
                    b[p][j - SubDomain[p, 1], m - SubDomain[p, 3]] = (h / 3.) * multright * sum(
                        exp(1. * 1.j * k * (dot(points, aaa))))
        if SubDomain[p, 3] == 0 and SubDomain[p, 1] == 0:  # bottom left corner
            points = array([[0., y[0]], [0., 0.], [y[0], 0.]])
            b[p][0, 0] = (h / 6.) * multleft * (
                    2. * exp(1. * 1.j * k * (dot(points[0, :], aaa))) + exp(1. * 1.j * k * dot(points[1, :], aaa)))
            b[p][0, 0] = b[p][0, 0] + (h / 6.) * multbot * (
                    2. * exp(1. * 1.j * k * dot(points[2, :], aaa)) + exp(1. * 1.j * k * dot(points[1, :], aaa)))
        if SubDomain[p, 4] == N and SubDomain[p, 1] == 0:  # bottom right corner
            points = array([[y[N - 2], 0.], [1., 0.], [1., y[0]]])  # bottom right corner
            b[p][0, -1] = (h / 6.) * multbot * (
                    2. * exp(1. * 1.j * k * dot(points[0, :], aaa)) + exp(1. * 1.j * k * dot(points[1, :], aaa)))
            b[p][0, -1] = b[p][0, -1] + (h / 6.) * multright * (
                    2. * exp(1. * 1.j * k * dot(points[2, :], aaa)) + exp(1. * 1.j * k * dot(points[1, :], aaa)))
        if SubDomain[p, 3] == 0 and SubDomain[p, 2] == N:  # top left corner
            points = array([[0., y[N - 2]], [0., 1.], [y[0], 1.]])
            b[p][-1, 0] = (h / 6.) * multleft * (
                    2. * exp(1. * 1.j * k * dot(points[0, :], aaa)) + exp(1. * 1.j * k * dot(points[1, :], aaa)))
            b[p][-1, 0] = b[p][-1, 0] + (h / 6.) * multtop * (
                    2. * exp(1. * 1.j * k * dot(points[2, :], aaa)) + exp(1. * 1.j * k * dot(points[1, :], aaa)))
        if SubDomain[p, 4] == N and SubDomain[p, 2] == N:  # top right corner
            points = array([[y[N - 2], 1.], [1., 1.], [1., y[N - 2]]])
            b[p][-1, -1] = (h / 6.) * multtop * (
                    2. * exp(1. * 1.j * k * dot(points[0, :], aaa)) + exp(1. * 1.j * k * dot(points[1, :], aaa)))
            b[p][-1, -1] = b[p][-1, -1] + (h / 6.) * multright * (
                    2. * exp(1. * 1.j * k * dot(points[2, :], aaa)) + exp(1. * 1.j * k * dot(points[1, :], aaa)))
        if OshapeD:  # switch the values in inactive nodes to 0.0
            b[p] = b[p] * InactiveNodes[p]
    return b


def local_rect(N, k, eps, eta, L, Nhoriz, Nvert):
    global DomainProc, SubDomain, nprocs, comm, rank, globtag, maxtag
    #   Discretises the BVP,
    #
    #   -\Delta u - (k^2+i\epsilon) u = f, on \Omega=(0,1)^2
    #            \partial_{n} u -i\eta u = 0 , on \partial\Omega
    #
    # on a domain of size
    #
    #  (Nhoriz - 1)*h      (in the horizontal direction)
    # and
    #  (Nvert - 1)*h       (in the vertical direction)
    # where h = L/(N-1)
    # The mesh is uniform in each direction with Nhoriz mesh points in the
    # horizontal direction and Nvert mesh points in the vertical direction
    # including the boundary points

    # The discretization uses piecewise linear finite elements on a uniform mesh with N grid
    # points in each coordinate direction (including the end points)
    # The nodes are ordered lexicographically wigh j being the horizontal index
    # and m being the vertical index
    # %  Input:
    #
    #   N= number of grid points in each direction in the global PDE problem
    #   k= wavenumber
    #   epsilon=damping parameter
    #   eta = impedance parameter
    #   Nhoriz = number of grid points in  the horizonal direction in the subproblem
    #   Nvert =  number of grid points in the vertical direction in the subproblem
    #
    # %  output:
    #
    #   S = finite element matrix on the subproblem
    #

    h = L * 1.0 / (N - 1.)
    h2 = h ** 2
    k2 = k ** 2

    nn = Nhoriz * Nvert
    nnz = N ** 2 + 4 * (N - 1) * N + 2 * (N - 1) ** 2

    a = zeros(nnz, dtype=int)  # x's
    b = zeros(nnz, dtype=int)  # y's
    c = zeros(nnz, dtype=complex)  # nz's
    pos = 0
    # form S
    for j in range(Nhoriz):
        for m in range(Nvert):
            # Matrix S = K - k^2 Mass - i*k*B
            # where K= Stiffness matrix, Mass = Domain mass matrix and B =
            # Boundary mass matrix

            ## Diagonal terms

            ## corners
            if m == 0 and j == 0:
                a[pos] = m * Nhoriz + j
                b[pos] = m * Nhoriz + j
                c[pos] = 1. - (
                        k2 + 1j * eps) * h2 / 6. - 1j * eta * 2 * h / 3.  # Ivan second and third term on RHS corrected
                pos = pos + 1
            if m == 0 and j == Nhoriz - 1:
                a[pos] = m * Nhoriz + j
                b[pos] = m * Nhoriz + j
                c[pos] = 1. - (k2 + 1j * eps) * h2 / 12. - 1j * eta * 2 * h / 3.  # Ivan third term on RHS corrected
                pos = pos + 1
            if m == Nvert - 1 and j == 0:
                a[pos] = m * Nhoriz + j
                b[pos] = m * Nhoriz + j
                c[pos] = 1. - (k2 + 1j * eps) * h2 / 12. - 1j * eta * 2 * h / 3.  # Ivan third term on RHS corrected
                pos = pos + 1
            if m == Nvert - 1 and j == Nhoriz - 1:
                a[pos] = m * Nhoriz + j
                b[pos] = m * Nhoriz + j
                c[pos] = 1. - (
                        k2 + 1j * eps) * h2 / 6. - 1j * eta * 2 * h / 3.  # Ivan second and third term on RHS corrected
                pos = pos + 1
            ## Diagonal terms on boundary (edge)
            if m == 0 and j > 0 and j < Nhoriz - 1:
                a[pos] = m * Nhoriz + j
                b[pos] = m * Nhoriz + j
                c[pos] = 2. - (k2 + 1j * eps) * h2 / 4. - 2. * 1j * eta * h / 3.
                pos = pos + 1
            if m == Nvert - 1 and j > 0 and j < Nhoriz - 1:
                a[pos] = m * Nhoriz + j
                b[pos] = m * Nhoriz + j
                c[pos] = 2. - (k2 + 1j * eps) * h2 / 4. - 2. * 1j * eta * h / 3.
                pos = pos + 1
            if m > 0 and m < Nvert - 1 and j > 0 and j < Nhoriz - 1:
                a[pos] = m * Nhoriz + j
                b[pos] = m * Nhoriz + j
                c[pos] = 4. - (k2 + 1j * eps) * h2 / 2.
                pos = pos + 1
            if j == 0 and m > 0 and m < Nvert - 1:
                a[pos] = m * Nhoriz + j
                b[pos] = m * Nhoriz + j
                c[pos] = 2. - (k2 + 1j * eps) * h2 / 4. - 2. * 1j * eta * h / 3.
                pos = pos + 1
            if j == Nhoriz - 1 and m > 0 and m < Nvert - 1:
                a[pos] = m * Nhoriz + j
                b[pos] = m * Nhoriz + j
                c[pos] = 2. - (k2 + 1j * eps) * h2 / 4. - 2. * 1j * eta * h / 3.
                pos = pos + 1
            ## Off diagonal On bottom and top boundary % Ivan
            if j < Nhoriz - 1 and m == 0:
                a[pos] = m * Nhoriz + j
                b[pos] = m * Nhoriz + j + 1
                c[pos] = -1. / 2. - (k2 + 1j * eps) * h2 / 24. - 1j * eta * h / 6.
                pos = pos + 1
                # following four lines added by Ivan % off diagonal on bottom
            if j < Nhoriz - 1 and j > 0 and m == 0:
                a[pos] = j  # a[pos]=(m-1)*(Mhoriz)+j -- in matlab, but m==0
                b[pos] = j + Nhoriz + 1  # b[pos]=(m-1)*(Mhoriz)+j+Mhoriz+1 -- in matlab, but m==0
                c[pos] = -(k2 + 1j * eps) * h2 / 12.
                pos = pos + 1
            if j < Nhoriz - 1 and m == Nvert - 1:
                a[pos] = m * Nhoriz + j
                b[pos] = m * Nhoriz + j + 1
                c[pos] = -1. / 2. - (k2 + 1j * eps) * h2 / 24. - 1j * eta * h / 6.
                pos = pos + 1
            if j > 0 and m == 0:
                a[pos] = m * Nhoriz + j
                b[pos] = m * Nhoriz + j - 1
                c[pos] = -1. / 2. - (k2 + 1j * eps) * h2 / 24. - 1j * eta * h / 6.
                pos = pos + 1
            if j > 0 and m == Nvert - 1:
                a[pos] = m * Nhoriz + j
                b[pos] = m * Nhoriz + j - 1
                c[pos] = -1. / 2. - (k2 + 1j * eps) * h2 / 24. - 1j * eta * h / 6.
                pos = pos + 1
                # following four lines added by Ivan % off diagonal on top
            if j > 0 and j < Nhoriz - 1 and m == Nvert - 1:
                a[pos] = m * Nhoriz + j
                b[pos] = m * Nhoriz + j - Nhoriz - 1
                c[pos] = -(k2 + 1j * eps) * h2 / 12.
                pos = pos + 1
            ## Off diagonal on left and right boundary % Ivan
            if m < Nvert - 1 and j == 0:
                a[pos] = m * Nhoriz + j
                b[pos] = (m + 1) * Nhoriz + j
                c[pos] = -1. / 2. - (k2 + 1j * eps) * h2 / 24. - 1j * eta * h / 6.
                pos = pos + 1
                # following four lines added by Ivan % off diagonal left boundary
                a[pos] = m * Nhoriz + j
                b[pos] = (m + 1) * Nhoriz + j + 1
                c[pos] = -(k2 + 1j * eps) * h2 / 12.
                pos = pos + 1
            if m < Nvert - 1 and j == Nhoriz - 1:
                a[pos] = m * Nhoriz + j
                b[pos] = (m + 1) * Nhoriz + j
                c[pos] = -1. / 2. - (k2 + 1j * eps) * h2 / 24. - 1j * eta * h / 6.
                pos = pos + 1
            if m > 0 and j == 0:
                a[pos] = m * Nhoriz + j
                b[pos] = (m - 1) * Nhoriz + j
                c[pos] = -1. / 2. - (k2 + 1j * eps) * h2 / 24. - 1j * eta * h / 6.
                pos = pos + 1
            if m > 0 and j == Nhoriz - 1:
                a[pos] = m * Nhoriz + j
                b[pos] = (m - 1) * Nhoriz + j
                c[pos] = -1. / 2. - (k2 + 1j * eps) * h2 / 24. - 1j * eta * h / 6.
                pos = pos + 1
                # following four lines added by Ivan % off diagonal right boundaary
                a[pos] = m * Nhoriz + j
                b[pos] = m * Nhoriz + j - Nhoriz - 1
                c[pos] = -(k2 + 1j * eps) * h2 / 12.
                pos = pos + 1
            ## In the interior
            if j < Nhoriz - 1 and m > 0 and m < Nvert - 1:
                a[pos] = m * Nhoriz + j
                b[pos] = m * Nhoriz + j + 1
                c[pos] = -1. - (k2 + 1j * eps) * h2 / 12.
                pos = pos + 1
            if j > 0 and m > 0 and m < Nvert - 1:
                a[pos] = m * Nhoriz + j
                b[pos] = m * Nhoriz + j - 1
                c[pos] = -1. - (k2 + 1j * eps) * h2 / 12.
                pos = pos + 1
            if m < Nvert - 1 and j > 0 and j < Nhoriz - 1:
                a[pos] = m * Nhoriz + j
                b[pos] = (m + 1) * Nhoriz + j
                c[pos] = -1. - (k2 + 1j * eps) * h2 / 12.
                pos = pos + 1
            if m > 0 and j > 0 and j < Nhoriz - 1:
                a[pos] = m * Nhoriz + j
                b[pos] = (m - 1) * Nhoriz + j
                c[pos] = -1. - (k2 + 1j * eps) * h2 / 12.
                pos = pos + 1
            # Off diagonal term, zero when k=0
            if j > 0 and j < Nhoriz - 1 and m > 0 and m < Nvert - 1:
                a[pos] = m * Nhoriz + j
                b[pos] = (m - 1) * Nhoriz + j - 1
                c[pos] = -((k2 + 1j * eps) * h2) / 12.
                pos = pos + 1
            if j > 0 and j < Nhoriz - 1 and m > 0 and m < Nvert - 1:
                a[pos] = m * Nhoriz + j
                b[pos] = (m + 1) * Nhoriz + j + 1
                c[pos] = -((k2 + 1j * eps) * h2) / 12.
                pos = pos + 1
    return scipy.sparse.csr_matrix((c[:pos], (a[:pos], b[:pos])), shape=(nn, nn))


def Poisson(N):
    global DomainProc, SubDomain, nprocs, comm, rank, globtag, maxtag
    #  Poisson matrix from FD
    nn = N ** 2
    nnz = N ** 2 + 4 * (N - 1) * N
    a = zeros(nnz, dtype=int)  # x's
    b = zeros(nnz, dtype=int)  # y's
    c = zeros(nnz, dtype=float)  # nz's
    pos = 0
    # form S
    for i in range(N):
        for j in range(N):
            ## Diagonal
            a[pos] = i * N + j
            b[pos] = i * N + j
            c[pos] = 4.0
            pos = pos + 1
            ## Subdiagonal
            if j > 0:
                a[pos] = i * N + j
                b[pos] = i * N + j - 1
                c[pos] = -1.0
                pos = pos + 1
            ## Updiagonal
            if j < N - 1:
                a[pos] = i * N + j
                b[pos] = i * N + j + 1
                c[pos] = -1.0
                pos = pos + 1
            ## Off diagonas
            if i > 0:
                a[pos] = (i - 1) * N + j
                b[pos] = i * N + j
                c[pos] = -1.0
                pos = pos + 1
            if i < N - 1:
                a[pos] = (i + 1) * N + j
                b[pos] = i * N + j
                c[pos] = -1.0
                pos = pos + 1
    return scipy.sparse.csr_matrix((c, (a, b)), shape=(nn, nn))


def create_varsize_subdomain_indeces(M, N, OLP):
    '''
    Indeces of M*M subdomains of N*N region
    '''
    global Mhoriz, Mvert, Mhoriz2, Mvert2
    global DomainProc, SubDomain, nprocs, comm, rank, globtag, maxtag

    n_mysubd = SubDomain.shape[0]
    G = []
    Mhoriz = zeros(n_mysubd, dtype=int)
    Mvert = zeros(n_mysubd, dtype=int)
    n = N - 1
    sdw = ones(M, dtype=int) * n // M
    rest = n - n // M * M
    sdw[-1:-rest - 1:-1] = sdw[-1:-rest - 1:-1] + 1  # make the last subdomains larger
    if rank == 0: print('*** sdW:', sdw)
    if rank == 0: print('***   N,sum(sdw):', N, sum(sdw))
    p = 0
    for i in range(M):
        for j in range(M):
            if DomainProc[p] < 0:  # my domain
                pp = -DomainProc[p] - 1  # get the pointer in SubDomain...
                DomainProc[p] = rank
                GG = []
                Lis = 0
                if i > 0:
                    Lis = OLP
                Lie = 0
                if i < M - 1:
                    Lie = OLP
                Ljs = 0
                if j > 0:
                    Ljs = OLP
                Lje = 0
                if j < M - 1:
                    Lje = OLP
                SubDomain[pp, 1] = sum(sdw[:i]) - Lis  # row start
                SubDomain[pp, 3] = sum(sdw[:j]) - Ljs  # col start
                SubDomain[pp, 5] = Lis  # local unique row start
                SubDomain[pp, 7] = Ljs  # local unique col start
                dstart = SubDomain[pp, 1] * N + SubDomain[pp, 3]
                for v in range(sdw[i] + 1 + Lis + Lie):  # number of rows
                    start = dstart + v * N
                    GG.append(list(range(start, start + sdw[j] + 1 + Ljs + Lje)))
                dstart_0 = sum(sdw[:i]) * N + sum(sdw[:j])
                for v in range(sdw[i] + 1):  # number of rows
                    start_0 = dstart_0 + v * N
                Mvert[pp] = Lis + sdw[i] + Lie + 1
                SubDomain[pp, 2] = SubDomain[pp, 1] + Mvert[pp]
                if i == M - 1:
                    SubDomain[pp, 6] = SubDomain[pp, 5] + sdw[i] + 1  # last local row
                else:
                    SubDomain[pp, 6] = SubDomain[pp, 5] + sdw[i]  # local unique row end
                Mhoriz[pp] = Ljs + sdw[j] + Lje + 1
                SubDomain[pp, 4] = SubDomain[pp, 3] + Mhoriz[pp]
                if j == M - 1:
                    SubDomain[pp, 8] = SubDomain[pp, 7] + sdw[j] + 1  # last local col
                else:
                    SubDomain[pp, 8] = SubDomain[pp, 7] + sdw[j]  # local unique col end
                SubDomain[pp, 9] = (SubDomain[pp, 2] - SubDomain[pp, 1]) * \
                                   (SubDomain[pp, 4] - SubDomain[pp, 3])  # subdomain size
                G.append(array(GG, dtype=int))
            p = p + 1
    return G


def create_eqsize_subdomain_indeces(M, N, OLP):
    '''
    Indeces of M*M subdomains of N*N region
    '''
    global Mhoriz, Mvert, Mhoriz2, Mvert2
    global DomainProc, SubDomain, nprocs, comm, rank, globtag, maxtag

    n_mysubd = SubDomain.shape[0]
    G = []
    Mhoriz = zeros(n_mysubd, dtype=int)
    Mvert = zeros(n_mysubd, dtype=int)
    n = N - 2 * OLP - 1
    short_w = n // M  # subd. width w/o OL
    sdsz = short_w + 2 * OLP + 1  # subdomain size with OLs
    # reference subd. numbers
    sd00 = zeros((sdsz, sdsz), dtype=int)
    for i in range(0, sdsz):
        k = i * N
        sd00[i, :] = arange(k, k + sdsz, dtype=int)
    p = 0
    for i in range(M):
        for j in range(M):
            if DomainProc[p] < 0:  # my domain
                pp = -DomainProc[p] - 1  # get the pointer in SubDomain...
                DomainProc[p] = rank
                Lis = 0
                if i > 0:
                    Lis = OLP
                Lie = 0
                if i < M - 1:
                    Lie = OLP
                Ljs = 0
                if j > 0:
                    Ljs = OLP
                Lje = 0
                if j < M - 1:
                    Lje = OLP
                SubDomain[pp, 1] = i * short_w  # row start
                SubDomain[pp, 3] = j * short_w  # col start
                SubDomain[pp, 5] = Lis  # local unique row start
                SubDomain[pp, 7] = Ljs  # local unique col start
                Mvert[pp] = sdsz
                SubDomain[pp, 2] = SubDomain[pp, 1] + Mvert[pp]
                if i == M - 1:
                    SubDomain[pp, 6] = SubDomain[pp, 5] + short_w + OLP  # last local row
                else:
                    SubDomain[pp, 6] = SubDomain[pp, 5] + short_w  # local unique row end
                Mhoriz[pp] = sdsz
                SubDomain[pp, 4] = SubDomain[pp, 3] + Mhoriz[pp]
                if j == M - 1:
                    SubDomain[pp, 8] = SubDomain[pp, 7] + short_w + OLP  # last local col
                else:
                    SubDomain[pp, 8] = SubDomain[pp, 7] + short_w  # local unique col end
                SubDomain[pp, 9] = (SubDomain[pp, 2] - SubDomain[pp, 1]) * \
                                   (SubDomain[pp, 4] - SubDomain[pp, 3])  # subdomain size
                startind = SubDomain[pp, 1] * N + SubDomain[pp, 3]
                sd = sd00 + startind
                G.append(sd)
            p = p + 1
    return G

def counter(z):
    global it, verbose, current_norm, time_per_it, time__
    global DomainProc, SubDomain, nprocs, comm, rank, globtag, maxtag
    it = it + 1
    if verbose == 10:
        if it>=1:
            t=time() - time__
            time_per_it = time_per_it + t
            if rank == 0: print(it, '--', z, ' ', t, 's')
        else:
            if rank == 0: print(it, '--', z)
        sys.stdout.flush()
        current_norm = z
    elif verbose > 0:
        if rank == 0: print('\r', it, '-->', z, end=' ')
        sys.stdout.flush()
        current_norm = z
    if it == 1:
        time_per_it = 0.0
    time__ = time()
    return z

def precd(z):
    return z / A.diagonal()


def prec_cp(z):
    return z


def as_prec(z):  # 1-level Additive Schwarz Preconditioner
    global GLOBALS, A, P, M_coarse, N, AVERAGER, Use_Poisson, n, epsilon, k, eps_prec1
    global A_eps, OL, Restricted_AS, CHOOSER, GLOBALS_0, M_subd, Robin, Averaging
    global Mhoriz, Mvert, Mhoriz2, Mvert2
    global DomainProc, SubDomain, nprocs, comm, rank, globtag, maxtag, time_per_it, time__
    global VarCoeff, Morig, Marmousi, Marmousi_c, UseCG, CGtol, CGMaxIT
    global it

    n_my = SubDomain.shape[0]
    if type(P[0]) == int:  # prepare the Additive Schwarz preconditioner
        if Restricted_AS:
            if rank == 0: print('    *** Restricted Additive Schwarz ***')
        if Robin == 0:
            if rank == 0: print('   *** Dirichlet BC on subdomain matrices ***')
        elif Robin == 1:
            if rank == 0: print('   *** Impedance BC on subdomain matrices ***')
        if Averaging:
            if Averaging == 2:
                if rank == 0: print('    +++ Averaging geometrically on overlaps       +++')
            else:
                if rank == 0: print('    +++ Averaging on overlaps       +++')
        if epsilon == eps_prec1 or Use_Poisson:
            for p in range(n_my):
                if Robin == 1:
                    Lv = 1.0
                    # Worth checking best eta value -- in case eps=k**2 eta=k
                    #                               much better than eta=k**2
                    #  --  seems like eta=k is the best value to use!!!
                    if p == 0:
                        if rank == 0: print('--- calling local_rect')
                    if VarCoeff:
                        P[p] = helmFE_var(N, k, \
                                          C=Marmousi[SubDomain[p, 1]:SubDomain[p, 2] - 1, \
                                            SubDomain[p, 3]:SubDomain[p, 4] - 1], \
                                          rho=eps_prec1, \
                                          Nhoriz=Mhoriz[p], Nvert=Mvert[p])
                    else:
                        P[p] = local_rect(N, k=k, eps=eps_prec1, eta=k, L=Lv, \
                                          Nhoriz=Mhoriz[p], Nvert=Mvert[p])
                else:
                    if p == 0:
                        if rank == 0: print('--- Using A for solves')
                    P[p] = A[p][2]
        else:
            if Robin == 0:  # and A_eps.shape[0]<=1:
                if rank == 0: print('    creating fine matrix with eps_prec1', eps_prec1)
                if VarCoeff:
                    A_eps = helm_fe_var(N, k, C=Marmousi, rho=eps_prec1)
                else:
                    A_eps = helm_fe(N, k, eps_prec1)
            for p in range(n_my):
                if Robin == 1:
                    Lv = 1.0
                    if p == 0:
                        if rank == 0: print('--- calling local_rect')
                    if VarCoeff:
                        P[p] = helmFE_var(N, k, \
                                          C=Marmousi[SubDomain[p, 1]:SubDomain[p, 2] - 1, \
                                            SubDomain[p, 3]:SubDomain[p, 4] - 1], \
                                          rho=eps_prec1, \
                                          Nhoriz=Mhoriz[p], Nvert=Mvert[p])
                    else:
                        P[p] = local_rect(N, k=k, eps=eps_prec1, eta=k, L=Lv, \
                                          Nhoriz=Mhoriz[p], Nvert=Mvert[p])
                else:
                    if p == 0:
                        if rank == 0: print('--- Using A_eps for solves')
                    P[p] = A_eps[p][2]
    r = list(range(n_my))
    time__=time()
    if 'time_per_it' not in globals():
        time_per_it=0.0
    if UseCG == 3:
        for p in range(n_my):
            if it<=1:
                t=time()
            r[p] = CG(P[0], z[p].ravel(), tol=CGtol, maxit=CGMaxIT)  # replace with GPGPU solver
            r[p] = r[p].reshape(GLOBALS[p].shape)
            if it<=1:
                if rank==0: print('  subsolve time:',p,time()-t)
    elif UseCG == 2:
        size = P[0].shape[0]
        x = np.ascontiguousarray(np.zeros(size*n_my), dtype=np.csingle)
        b_values = np.zeros(size*n_my, dtype=np.csingle)
        for p in range(n_my):
            b_values[p*size:(p+1)*size] = z[p][:].ravel()
        a_values = np.array(P[0].data, dtype=np.csingle)
        row_ptr = np.array(P[0].indptr, dtype=np.intc)
        col_idx = np.array(P[0].indices, dtype=np.intc)
        
        x = pcl.CG(size, P[0].nnz, a_values, b_values,
                     row_ptr, col_idx, x, n_my, CGMaxIT)
        
        for p in range(n_my):
            r[p] = x[p*size:(p+1)*size].astype(complex)
            r[p] = r[p].reshape(GLOBALS[p].shape)
    elif UseCG == 12:
        size = P[0].shape[0]
        x = np.ascontiguousarray(np.zeros(size*n_my), dtype=np.csingle)
        b_vals = zeros(size*n_my, dtype=z[0].dtype)
        for p in range(n_my):
            b_vals[p*size:(p+1)*size] = z[p][:].ravel()
            # print(p,' -- z=',z[p][:].ravel())
        # print('bvals=',b_vals)
        b_values = np.array(b_vals, dtype=np.csingle)
        a_values = np.array(P[0].data, dtype=np.csingle)
        row_ptr = np.array(P[0].indptr, dtype=np.intc)
        col_idx = np.array(P[0].indices, dtype=np.intc)
        libcg.cg.argtypes=[c_int, c_int, ndpointer(dtype=np.csingle,ndim=1,flags='C'), ndpointer(dtype=np.csingle,ndim=1,flags='C'), ndpointer(dtype=np.intc,ndim=1,flags='C'),
        ndpointer(dtype=np.intc,ndim=1,flags='C'), ndpointer(dtype=np.csingle,ndim=1,flags='C'), c_int, c_int, c_int]
        libcg.cg(size, P[0].nnz, a_values, b_values, row_ptr, col_idx, x, n_my, CGMaxIT, 1)
        for p in range(n_my):
            r[p] = x[p*size:(p+1)*size].astype(complex)
            r[p] = r[p].reshape(GLOBALS[p].shape)
    else:
        for p in range(n_my):
            if UseCG == 1:
                if it<=1:
                    t=time()
                size = P[0].shape[0]
                a_values=np.array(P[0].data,dtype=np.csingle)
                row_ptr=np.array(P[0].indptr,dtype=np.intc)
                col_idx=np.array(P[0].indices,dtype=np.intc)
                x=np.ascontiguousarray(np.zeros(size),dtype=np.csingle)
                b_values=np.array(z[p].ravel(),dtype=np.csingle)
                
                x = pcl.CG(size, P[p].nnz, a_values, b_values,
                     row_ptr, col_idx, x, 1, CGMaxIT)
                r[p] = x.astype(complex)     
                if it<=1:
                    if rank==0: print('  subsolve time:',p,time()-t)
            elif UseCG==11:
                if it<=1:
                    t=time()
                size = P[0].shape[0]
                a_values=np.array(P[0].data,dtype=np.csingle)
                row_ptr=np.array(P[0].indptr,dtype=np.intc)
                col_idx=np.array(P[0].indices,dtype=np.intc)
                x=np.ascontiguousarray(np.zeros(size),dtype=np.csingle)
                b_values=np.array(z[p].ravel(),dtype=np.csingle)
                libcg.cg.argtypes=[c_int, c_int, ndpointer(dtype=np.csingle,ndim=1,flags='C'), ndpointer(dtype=np.csingle,ndim=1,flags='C'), ndpointer(dtype=np.intc,ndim=1,flags='C'),
                ndpointer(dtype=np.intc,ndim=1,flags='C'), ndpointer(dtype=np.csingle,ndim=1,flags='C'), c_int, c_int, c_int]
                
                libcg.cg(size, P[0].nnz, a_values, b_values, row_ptr, col_idx, x, 1, CGMaxIT, 1)
                
                r[p] = x.astype(complex)
                if it<=1:
                    if rank==0: print('  subsolve time:',p,time()-t)
            else: # UseCG==0
                if it<=1:
                    t=time()
                r[p] = scipy.sparse.linalg.spsolve(P[p], z[p].ravel())
                if it<=1:
                    if rank==0: print('  subsolve time:',p,time()-t)
            r[p] = r[p].reshape(GLOBALS[p].shape)
    r = OL_update(r)
    return r

def check_nd_print_global_vec(v, txt):  # comm not done yet...
    global N, M_coarse, R, RT, A, A_c, n, m, scale_int, Explicit_Acoarse, k, epsilon
    global DomainProc, SubDomain, nprocs, comm, rank
    n_my = SubDomain.shape[0]
    v_tst = zeros((N, N), dtype=complex)
    for p in range(n_my):
        v_resh = v[p].reshape((SubDomain[p, 2] - SubDomain[p, 1], SubDomain[p, 4] - SubDomain[p, 3]))
        i1 = 0
        for k1 in range(SubDomain[p, 1], SubDomain[p, 2]):
            i2 = 0
            # for k2 in range(SubDomain[p,1],SubDomain[p,2]):
            for k2 in range(SubDomain[p, 3], SubDomain[p, 4]):
                if v_tst[k1, k2] != 0j:
                    if abs(v_tst[k1, k2] - v_resh[i1, i2]) > 1e-15:
                        print(p, k1, k2, '::inconsistency:', v_resh[i1, i2], v_tst[k1, k2])
                else:
                    v_tst[k1, k2] = v_resh[i1, i2]
                i2 = i2 + 1
            i1 = i1 + 1
    print(txt, '=', v_tst)


def check_nd_plot_global_vec(v, txt):  # comm not done yet...
    global N, M_coarse, R, RT, A, A_c, n, m, scale_int, Explicit_Acoarse, k, epsilon
    global DomainProc, SubDomain, nprocs, comm, rank
    import matplotlib.pyplot as plt
    x = arange(0.0, 1.00001, 1.0 / (N - 1))
    y = arange(0.0, 1.00001, 1.0 / (N - 1))
    n_my = SubDomain.shape[0]
    v_tst = zeros((N, N), dtype=complex)
    for p in range(n_my):
        v_resh = v[p].reshape((SubDomain[p, 2] - SubDomain[p, 1], SubDomain[p, 4] - SubDomain[p, 3]))
        i1 = 0
        for k1 in range(SubDomain[p, 1], SubDomain[p, 2]):
            i2 = 0
            # for k2 in range(SubDomain[p,1],SubDomain[p,2]):
            for k2 in range(SubDomain[p, 3], SubDomain[p, 4]):
                if v_tst[k1, k2] != 0j:
                    if abs(v_tst[k1, k2] - v_resh[i1, i2]) > 1e-12:
                        print(p, k1, k2, '::inconsistency:', v_resh[i1, i2], v_tst[k1, k2])
                else:
                    v_tst[k1, k2] = v_resh[i1, i2]
                i2 = i2 + 1
            i1 = i1 + 1
    plt.figure()
    xx, yy = meshgrid(x, y)
    v = (v_tst.real + v_tst.imag) / 2.0
    plt.pcolor(xx, yy, v)
    plt.colorbar()
    plt.title(txt)
    plt.show()


def check_nd_plot3d_global_vec(v, txt):  # comm not done yet...
    global N, M_coarse, R, RT, A, A_c, n, m, scale_int, Explicit_Acoarse, k, epsilon
    global DomainProc, SubDomain, nprocs, comm, rank
    x = arange(0.0, 1.00001, 1.0 / (N - 1))
    y = arange(0.0, 1.00001, 1.0 / (N - 1))
    n_my = SubDomain.shape[0]
    v_tst = zeros((N, N), dtype=complex)
    for p in range(n_my):
        v_resh = v[p].reshape((SubDomain[p, 2] - SubDomain[p, 1], SubDomain[p, 4] - SubDomain[p, 3]))
        i1 = 0
        for k1 in range(SubDomain[p, 1], SubDomain[p, 2]):
            i2 = 0
            # for k2 in range(SubDomain[p,1],SubDomain[p,2]):
            for k2 in range(SubDomain[p, 3], SubDomain[p, 4]):
                if v_tst[k1, k2] != 0j:
                    if abs(v_tst[k1, k2] - v_resh[i1, i2]) > 1e-7 * max(1.0, abs(v_resh[i1, i2])):
                        print(p, k1, k2, '::inconsistency:', v_resh[i1, i2], v_tst[k1, k2])
                else:
                    v_tst[k1, k2] = v_resh[i1, i2]
                i2 = i2 + 1
            i1 = i1 + 1
    # plot3d(v_tst.real,txt+'(real)')
    # plot3d(v_tst.imag,txt+'(imag)')
    v = (v_tst.real + v_tst.imag) / 2.0
    plot3d(v, txt + '(aver(real,imag))')


def plot3d(f, label):  # for plotting 1D functions
    import Gnuplot.funcutils
    g = Gnuplot.Gnuplot(debug=0, persist=1)
    g.clear()
    x = arange(len(f[:, 0]))
    y = arange(len(f[0, :]))
    g('set parametric')
    # g('set dgrid3d')
    g('set style data lines')
    # g('set style data pm3d')
    # g('set hidden')
    g('set contour base')
    g.title(label)
    g.xlabel('x')
    g.ylabel('y')
    g.splot(Gnuplot.GridData(f, x, y, binary=0))
    input("click enter <--' ")  # needed to be able to rotate the fig.


def norm(zz, Dk=None):
    global N
    global DomainProc, SubDomain, nprocs, comm, rank, OL, M_subd, globtag, maxtag
    n_mysubd = SubDomain.shape[0]
    nrm = 0.0
    if Dk == None:
        for p in range(n_mysubd):
            nrm = nrm + real(vdot(zz[p][SubDomain[p, 5]:SubDomain[p, 6], SubDomain[p, 7]:SubDomain[p, 8]], \
                                  zz[p][SubDomain[p, 5]:SubDomain[p, 6], SubDomain[p, 7]:SubDomain[p, 8]]))
    else:
        print('Dk case not implemented yet...')
        exit(0)
    nrm2 = comm.allreduce(nrm)
    return sqrt(nrm2)


def OL_update(x, Force_Averaging=False, DEBA=False):
    global N
    global DomainProc, SubDomain, nprocs, comm, rank, OL, M_subd, globtag, maxtag
    global Averaging
    if Force_Averaging == False:
        Force_Averaging = Averaging
    n_mysubd = SubDomain.shape[0]
    if DEBA: print(rank, ' -- I have DomainProc:', DomainProc)
    if DEBA: print(rank, ' -- my subdomains:', SubDomain[:, 0])
    if DEBA:
        for p in range(n_mysubd):
            print(rank, p, SubDomain[p])
    nsubd = M_subd ** 2
    if DEBA: print(rank, ' ++ globtag=', globtag, ' nsubd=', nsubd)
    sreqs = []
    rreqs = []
    sbuf = []
    rbuf = []
    rq = 0
    sq = 0
    for p in range(n_mysubd):
        csd = SubDomain[p, 0]  # Current SubDomain
        # start comm nearest neighbour communication
        if SubDomain[p, 3] > 0:  # not the leftmost subdomain W
            if SubDomain[p, 1] > 0:  # not the downmost subdomain S
                strt = OL
            else:
                strt = 0
            if SubDomain[p, 2] < N:  # not the upmost subdomain N
                endt = x[p].shape[0] - OL
            else:
                endt = x[p].shape[0]
            rbuf.append(empty((endt - strt) * (OL + 1) + 16, dtype=complex))
            nsd = csd - 1  # Neighbour SubDomain #
            nown = DomainProc[nsd]  # Neighbour's OWNer process #
            rtag = (globtag + nsd * nsubd + csd) % maxtag  # Senders number *nsubd + Receivers number
            stag = (globtag + csd * nsubd + nsd) % maxtag  # Senders number *nsubd + Receivers number
            rreqs.append(comm.Irecv([rbuf[rq], MPI.COMPLEX], source=nown, tag=rtag))
            rq = rq + 1
            sbuf.append(x[p][strt:endt, OL:2 * OL + 1].flatten())
            sreqs.append(comm.Isend([sbuf[sq], MPI.COMPLEX], dest=nown, tag=stag))
            sq = sq + 1
            if DEBA: print(rank, p, SubDomain[p, 0], rq - 1, sq - 1, 'sent to W', nown, rtag, stag)
        if SubDomain[p, 4] < N:  # not the rightmost subdomain E
            if SubDomain[p, 1] > 0:  # not the downmost subdomain S
                strt = OL
            else:
                strt = 0
            if SubDomain[p, 2] < N:  # not the upmost subdomain N
                endt = x[p].shape[0] - OL
            else:
                endt = x[p].shape[0]
            rbuf.append(empty((endt - strt) * (OL + 1) + 16, dtype=complex))
            nsd = csd + 1  # Neighbour SubDomain #
            nown = DomainProc[nsd]  # Neighbour's OWNer process #
            rtag = (globtag + nsd * nsubd + csd) % maxtag  # Senders number *nsubd + Receivers number
            stag = (globtag + csd * nsubd + nsd) % maxtag  # Senders number *nsubd + Receivers number
            rreqs.append(comm.Irecv([rbuf[rq], MPI.COMPLEX], source=nown, tag=rtag))
            rq = rq + 1
            sbuf.append(x[p][strt:endt, -2 * OL - 1:-OL].flatten())
            sreqs.append(comm.Isend([sbuf[sq], MPI.COMPLEX], dest=nown, tag=stag))
            sq = sq + 1
            if DEBA: print(rank, p, SubDomain[p, 0], rq - 1, sq - 1, 'sent to E', nown, rtag, stag)
        if SubDomain[p, 1] > 0:  # not the downmost subdomain S
            if SubDomain[p, 3] > 0:  # not the leftmost subdomain W
                strt = OL
            else:
                strt = 0
            if SubDomain[p, 4] < N:  # not the rightmost subdomain E
                endt = x[p].shape[1] - OL
            else:
                endt = x[p].shape[1]
            rbuf.append(empty((OL + 1) * (endt - strt) + 16, dtype=complex))
            nsd = csd - M_subd  # Neighbour SubDomain #
            nown = DomainProc[nsd]  # Neighbour's OWNer process #
            rtag = (globtag + nsd * nsubd + csd) % maxtag  # Senders number *nsubd + Receivers number
            stag = (globtag + csd * nsubd + nsd) % maxtag  # Senders number *nsubd + Receivers number
            rreqs.append(comm.Irecv([rbuf[rq], MPI.COMPLEX], source=nown, tag=rtag))
            rq = rq + 1
            sbuf.append(x[p][OL:2 * OL + 1, strt:endt].flatten())
            sreqs.append(comm.Isend([sbuf[sq], MPI.COMPLEX], dest=nown, tag=stag))
            sq = sq + 1
            if DEBA: print(rank, p, SubDomain[p, 0], rq - 1, sq - 1, 'sent to S', nown, rtag, stag)
        if SubDomain[p, 2] < N:  # not the upmost subdomain N
            if SubDomain[p, 3] > 0:  # not the leftmost subdomain W
                strt = OL
            else:
                strt = 0
            if SubDomain[p, 4] < N:  # not the rightmost subdomain E
                endt = x[p].shape[1] - OL
            else:
                endt = x[p].shape[1]
            rbuf.append(empty((OL + 1) * (endt - strt) + 16, dtype=complex))
            nsd = csd + M_subd  # Neighbour SubDomain #
            nown = DomainProc[nsd]  # Neighbour's OWNer process #
            rtag = (globtag + nsd * nsubd + csd) % maxtag  # Senders number *nsubd + Receivers number
            stag = (globtag + csd * nsubd + nsd) % maxtag  # Senders number *nsubd + Receivers number
            rreqs.append(comm.Irecv([rbuf[rq], MPI.COMPLEX], source=nown, tag=rtag))
            rq = rq + 1
            sbuf.append(x[p][-2 * OL - 1:-OL, strt:endt].flatten())
            sreqs.append(comm.Isend([sbuf[sq], MPI.COMPLEX], dest=nown, tag=stag))
            sq = sq + 1
            if DEBA: print(rank, p, SubDomain[p, 0], rq - 1, sq - 1, 'sent to N', nown, rtag, stag)
        if SubDomain[p, 1] > 0 and SubDomain[p, 3] > 0:  # has SW neighbour
            rbuf.append(empty((OL + 1) ** 2 + 16, dtype=complex))
            nsd = csd - M_subd - 1  # Neighbour SubDomain #
            nown = DomainProc[nsd]  # Neighbour's OWNer process #
            rtag = (globtag + nsd * nsubd + csd) % maxtag  # Senders number *nsubd + Receivers number
            stag = (globtag + csd * nsubd + nsd) % maxtag  # Senders number *nsubd + Receivers number
            rreqs.append(comm.Irecv([rbuf[rq], MPI.COMPLEX], source=nown, tag=rtag))
            rq = rq + 1
            sbuf.append(x[p][OL:2 * OL + 1, OL:2 * OL + 1].flatten())
            sreqs.append(comm.Isend([sbuf[sq], MPI.COMPLEX], dest=nown, tag=stag))
            sq = sq + 1
            if DEBA: print(rank, p, SubDomain[p, 0], rq - 1, sq - 1, 'sent to SW', nown, rtag, stag)
        if SubDomain[p, 2] < N and SubDomain[p, 4] < N:  # has NE neighbour
            rbuf.append(empty((OL + 1) ** 2 + 16, dtype=complex))
            nsd = csd + M_subd + 1  # Neighbour SubDomain #
            nown = DomainProc[nsd]  # Neighbour's OWNer process #
            rtag = (globtag + nsd * nsubd + csd) % maxtag  # Senders number *nsubd + Receivers number
            stag = (globtag + csd * nsubd + nsd) % maxtag  # Senders number *nsubd + Receivers number
            rreqs.append(comm.Irecv([rbuf[rq], MPI.COMPLEX], source=nown, tag=rtag))
            rq = rq + 1
            sbuf.append(x[p][-2 * OL - 1:-OL, -2 * OL - 1:-OL].flatten())
            sreqs.append(comm.Isend([sbuf[sq], MPI.COMPLEX], dest=nown, tag=stag))
            sq = sq + 1
            if DEBA: print(rank, p, SubDomain[p, 0], rq - 1, sq - 1, 'sent to NE', nown, rtag, stag)
        if SubDomain[p, 3] > 0 and SubDomain[p, 2] < N:  # has NW neighbour
            rbuf.append(empty((OL + 1) ** 2 + 16, dtype=complex))
            nsd = csd + M_subd - 1  # Neighbour SubDomain #
            nown = DomainProc[nsd]  # Neighbour's OWNer process #
            rtag = (globtag + nsd * nsubd + csd) % maxtag  # Senders number *nsubd + Receivers number
            stag = (globtag + csd * nsubd + nsd) % maxtag  # Senders number *nsubd + Receivers number
            rreqs.append(comm.Irecv([rbuf[rq], MPI.COMPLEX], source=nown, tag=rtag))
            rq = rq + 1
            sbuf.append(x[p][-2 * OL - 1:-OL, OL:2 * OL + 1].flatten())
            sreqs.append(comm.Isend([sbuf[sq], MPI.COMPLEX], dest=nown, tag=stag))
            sq = sq + 1
            if DEBA: print(rank, p, SubDomain[p, 0], rq - 1, sq - 1, 'sent to NW', nown, rtag, stag)
        if SubDomain[p, 4] < N and SubDomain[p, 1] > 0:  # has SE neighbour
            rbuf.append(empty((OL + 1) ** 2 + 16, dtype=complex))
            nsd = csd - M_subd + 1  # Neighbour SubDomain #
            nown = DomainProc[nsd]  # Neighbour's OWNer process #
            rtag = (globtag + nsd * nsubd + csd) % maxtag  # Senders number *nsubd + Receivers number
            stag = (globtag + csd * nsubd + nsd) % maxtag  # Senders number *nsubd + Receivers number
            rreqs.append(comm.Irecv([rbuf[rq], MPI.COMPLEX], source=nown, tag=rtag))
            rq = rq + 1
            sbuf.append(x[p][OL:2 * OL + 1, -2 * OL - 1:-OL].flatten())
            sreqs.append(comm.Isend([sbuf[sq], MPI.COMPLEX], dest=nown, tag=stag))
            sq = sq + 1
            if DEBA: print(rank, p, SubDomain[p, 0], rq - 1, sq - 1, 'sent to SE', nown, rtag, stag)
    # could do some useful work here...?
    #   Prepare for Restricted Additive Schwarz here...
    if Restricted_AS:
        for p in range(n_mysubd):
            if SubDomain[p, 3] > 0:  # not the leftmost subdomain W
                if SubDomain[p, 1] > 0:  # not the downmost subdomain S
                    # strt=OL+1
                    strt = OL
                else:
                    strt = 0
                if SubDomain[p, 2] < N:  # not the upmost subdomain N
                    # endt=x[p].shape[0]-OL-1
                    endt = x[p].shape[0] - OL
                else:
                    endt = x[p].shape[0]
                x[p][strt:endt, :OL] = 0j
            if SubDomain[p, 4] < N:  # not the rightmost subdomain E
                if SubDomain[p, 1] > 0:  # not the downmost subdomain S
                    # strt=OL+1
                    strt = OL
                else:
                    strt = 0
                if SubDomain[p, 2] < N:  # not the upmost subdomain N
                    # endt=x[p].shape[0]-OL-1
                    endt = x[p].shape[0] - OL
                else:
                    endt = x[p].shape[0]
                x[p][strt:endt, -OL:] = 0j
            if SubDomain[p, 1] > 0:  # not the downmost subdomain S
                if SubDomain[p, 3] > 0:  # not the leftmost subdomain W
                    # strt=OL+1
                    strt = OL
                else:
                    strt = 0
                if SubDomain[p, 4] < N:  # not the rightmost subdomain E
                    # endt=x[p].shape[1]-OL-1
                    endt = x[p].shape[1] - OL
                else:
                    endt = x[p].shape[1]
                x[p][:OL, strt:endt] = 0j
            if SubDomain[p, 2] < N:  # not the upmost subdomain N
                if SubDomain[p, 3] > 0:  # not the leftmost subdomain W
                    strt = OL
                else:
                    strt = 0
                if SubDomain[p, 4] < N:  # not the rightmost subdomain E
                    # endt=x[p].shape[1]-OL-1
                    endt = x[p].shape[1] - OL
                else:
                    endt = x[p].shape[1]
                x[p][-OL:, strt:endt] = 0j
            if SubDomain[p, 1] > 0 and SubDomain[p, 3] > 0:  # has SW neighbour
                x[p][:OL, :OL] = 0j  # SW
            if SubDomain[p, 2] < N and SubDomain[p, 4] < N:  # has NE neighbour
                x[p][-OL:, -OL:] = 0j
            if SubDomain[p, 3] > 0 and SubDomain[p, 2] < N:  # has NW neighbour
                x[p][-OL:, :OL] = 0j
            if SubDomain[p, 4] < N and SubDomain[p, 1] > 0:  # has SE neighbour
                x[p][:OL, -OL:] = 0j
    # check that the communication ended and continue
    if DEBA: print('sreqs=', len(sreqs))
    MPI.Request.Waitall(sreqs)  # here finished with all sends.. (remove?)
    if DEBA: print(rank, 'sreqs -- got all', len(sreqs))
    # pdb.set_trace()
    rq = 0
    for p in range(n_mysubd):
        if SubDomain[p, 3] > 0:  # not the leftmost subdomain
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'waiting from W')
            MPI.Request.Wait(rreqs[rq])
            if SubDomain[p, 1] > 0:  # not the downmost subdomain S
                strt = OL
            else:
                strt = 0
            if SubDomain[p, 2] < N:  # not the upmost subdomain N
                endt = x[p].shape[0] - OL
            else:
                endt = x[p].shape[0]
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'got from W')  # ,'rbuf is:',rbuf[rq]
            x[p][strt:endt, :OL + 1] = x[p][strt:endt, :OL + 1] + rbuf[rq][:(endt - strt) * (OL + 1)].reshape(
                (endt - strt, OL + 1))
            # TODO: do we have to reshape these?
            rq = rq + 1
        if SubDomain[p, 4] < N:  # not the rightmost subdomain
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'waiting from E')  # ,'rbuf is:',rbuf[rq]
            MPI.Request.Wait(rreqs[rq])
            if SubDomain[p, 1] > 0:  # not the downmost subdomain S
                strt = OL
            else:
                strt = 0
            if SubDomain[p, 2] < N:  # not the upmost subdomain N
                endt = x[p].shape[0] - OL
            else:
                endt = x[p].shape[0]
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'got from E')  # ,'rbuf is:',rbuf[rq]
            x[p][strt:endt, -OL - 1:] = x[p][strt:endt, -OL - 1:] + rbuf[rq][:(endt - strt) * (OL + 1)].reshape(
                (endt - strt, OL + 1))
            rq = rq + 1
        if SubDomain[p, 1] > 0:  # not the downmost subdomain
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'waiting from S')  # ,'rbuf is:',rbuf[rq]
            MPI.Request.Wait(rreqs[rq])
            if SubDomain[p, 3] > 0:  # not the leftmost subdomain W
                strt = OL
            else:
                strt = 0
            if SubDomain[p, 4] < N:  # not the rightmost subdomain E
                endt = x[p].shape[1] - OL
            else:
                endt = x[p].shape[1]
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'got from S')  # ,'rbuf is:',rbuf[rq]
            x[p][:OL + 1, strt:endt] = x[p][:OL + 1, strt:endt] + rbuf[rq][:(endt - strt) * (OL + 1)].reshape(
                (OL + 1, endt - strt))
            rq = rq + 1
        if SubDomain[p, 2] < N:  # not the upmost subdomain
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'waiting from N')  # ,'rbuf is:',rbuf[rq]
            MPI.Request.Wait(rreqs[rq])
            if SubDomain[p, 3] > 0:  # not the leftmost subdomain W
                strt = OL
            else:
                strt = 0
            if SubDomain[p, 4] < N:  # not the rightmost subdomain E
                endt = x[p].shape[1] - OL
            else:
                endt = x[p].shape[1]
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'got from N')  # ,'rbuf is:',rbuf[rq]
            x[p][-OL - 1:, strt:endt] = x[p][-OL - 1:, strt:endt] + rbuf[rq][:(endt - strt) * (OL + 1)].reshape(
                (OL + 1, endt - strt))
            rq = rq + 1
        if SubDomain[p, 1] > 0 and SubDomain[p, 3] > 0:  # has SW neighbour
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'waiting from SW')  # ,'rbuf is:',rbuf[rq]
            MPI.Request.Wait(rreqs[rq])
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'got from SW')  # ,'rbuf is:',rbuf[rq]
            x[p][:OL + 1, :OL + 1] = x[p][:OL + 1, :OL + 1] + rbuf[rq][:(OL + 1) ** 2].reshape((OL + 1, OL + 1))
            rq = rq + 1
        if SubDomain[p, 2] < N and SubDomain[p, 4] < N:  # has NE neighbour
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'waiting from NE')  # ,'rbuf is:',rbuf[rq]
            MPI.Request.Wait(rreqs[rq])
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'got from NE')  # ,'rbuf is:',rbuf[rq]
            x[p][-OL - 1:, -OL - 1:] = x[p][-OL - 1:, -OL - 1:] + rbuf[rq][:(OL + 1) ** 2].reshape((OL + 1, OL + 1))
            rq = rq + 1
        if SubDomain[p, 3] > 0 and SubDomain[p, 2] < N:  # has NW neighbour
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'waiting from NW')  # ,'rbuf is:',rbuf[rq]
            MPI.Request.Wait(rreqs[rq])
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'got from NW')  # ,'rbuf is:',rbuf[rq]
            x[p][-OL - 1:, :OL + 1] = x[p][-OL - 1:, :OL + 1] + rbuf[rq][:(OL + 1) ** 2].reshape((OL + 1, OL + 1))
            rq = rq + 1
        if SubDomain[p, 4] < N and SubDomain[p, 1] > 0:  # has SE neighbour
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'waiting from SE')  # ,'rbuf is:',rbuf[rq]
            MPI.Request.Wait(rreqs[rq])
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'got from SE')  # ,'rbuf is:',rbuf[rq]
            x[p][:OL + 1, -OL - 1:] = x[p][:OL + 1, -OL - 1:] + rbuf[rq][:(OL + 1) ** 2].reshape((OL + 1, OL + 1))
            rq = rq + 1
    # Take the averages on lines of contact:
    if Force_Averaging:
        for p in range(n_mysubd):
            if SubDomain[p, 3] > 0:  # not the leftmost subdomain
                x[p][:, OL] = x[p][:, OL] / 2.0
            if SubDomain[p, 4] < N:  # not the rightmost subdomain
                x[p][:, -OL - 1] = x[p][:, -OL - 1] / 2.0
            if SubDomain[p, 1] > 0:  # not the downmost subdomain
                x[p][OL, :] = x[p][OL, :] / 2.0
            if SubDomain[p, 2] < N:  # not the upmost subdomain
                x[p][-OL - 1, :] = x[p][-OL - 1, :] / 2.0
    globtag = (globtag + (nsubd + 1) * nsubd) % maxtag
    return x


def Ax_op(A, x, DEBA=False):
    global N
    global DomainProc, SubDomain, nprocs, comm, rank, OL, M_subd, globtag, maxtag
    n_mysubd = SubDomain.shape[0]
    if DEBA: print(rank, ' -- I have DomainProc:', DomainProc)
    if DEBA: print(rank, ' -- my subdomains:', SubDomain[:, 0])
    if DEBA:
        for p in range(n_mysubd):
            print(rank, p, SubDomain[p])
    nsubd = M_subd ** 2
    if DEBA: print(rank, ' ++ globtag=', globtag, ' nsubd=', nsubd)
    y = []
    sreqs = []
    rreqs = []
    sbuf = []
    rbuf = []
    rq = 0
    sq = 0
    for p in range(n_mysubd):
        csd = SubDomain[p, 0]  # Current SubDomain
        y.append(zeros(GLOBALS[p].shape, dtype=complex))
        A[p][0] = aslinearoperator(A[p][0])
        # calculate the value to be communicated to neighbours
        y[p] = y[p] + (A[p][0].matvec(x[p].ravel())).reshape(y[p].shape)
        # start comm nearest neighbour communication
        if SubDomain[p, 3] > 0:  # not the leftmost subdomain W
            if SubDomain[p, 1] > 0:  # not the downmost subdomain S
                strt = 1
            else:
                strt = 0
            if SubDomain[p, 2] < N:  # not the upmost subdomain N
                endt = y[p].shape[0] - 1
            else:
                endt = y[p].shape[0]
            rbuf.append(empty((y[p][strt:endt, 0]).size + 16, dtype=complex))
            nsd = csd - 1  # Neighbour SubDomain #
            nown = DomainProc[nsd]  # Neighbour's OWNer process #
            rtag = (globtag + nsd * nsubd + csd) % maxtag  # Senders number *nsubd + Receivers number
            stag = (globtag + csd * nsubd + nsd) % maxtag  # Senders number *nsubd + Receivers number
            rreqs.append(comm.Irecv([rbuf[rq], MPI.COMPLEX], source=nown, tag=rtag))
            rq = rq + 1
            sbuf.append(y[p][strt:endt, 2 * OL].flatten())
            sreqs.append(comm.Isend([sbuf[sq], MPI.COMPLEX], dest=nown, tag=stag))
            sq = sq + 1
            if DEBA: print(rank, p, SubDomain[p, 0], sq, 'sent to W', nown, stag)
            # print p,'1 sent:',sbuf[sq]
        if SubDomain[p, 4] < N:  # not the rightmost subdomain E
            if SubDomain[p, 1] > 0:  # not the downmost subdomain S
                strt = 1
            else:
                strt = 0
            if SubDomain[p, 2] < N:  # not the upmost subdomain N
                endt = y[p].shape[0] - 1
            else:
                endt = y[p].shape[0]
            rbuf.append(empty((y[p][strt:endt, -1]).size + 16, dtype=complex))
            nsd = csd + 1  # Neighbour SubDomain #
            nown = DomainProc[nsd]  # Neighbour's OWNer process #
            rtag = (globtag + nsd * nsubd + csd) % maxtag  # Senders number *nsubd + Receivers number
            stag = (globtag + csd * nsubd + nsd) % maxtag  # Senders number *nsubd + Receivers number
            rreqs.append(comm.Irecv([rbuf[rq], MPI.COMPLEX], source=nown, tag=rtag))
            rq = rq + 1
            sbuf.append(y[p][strt:endt, -2 * OL - 1].flatten())
            sreqs.append(comm.Isend([sbuf[sq], MPI.COMPLEX], dest=nown, tag=stag))
            sq = sq + 1
            if DEBA: print(rank, p, SubDomain[p, 0], sq, 'sent to E', nown, stag)
            # print p,'2 sent:',sbuf[sq]
        if SubDomain[p, 1] > 0:  # not the downmost subdomain S
            if SubDomain[p, 3] > 0:  # not the leftmost subdomain W
                strt = 1
            else:
                strt = 0
            if SubDomain[p, 4] < N:  # not the rightmost subdomain E
                endt = y[p].shape[1] - 1
            else:
                endt = y[p].shape[1]
            rbuf.append(empty((y[p][0, strt:endt]).size + 16, dtype=complex))
            nsd = csd - M_subd  # Neighbour SubDomain #
            nown = DomainProc[nsd]  # Neighbour's OWNer process #
            rtag = (globtag + nsd * nsubd + csd) % maxtag  # Senders number *nsubd + Receivers number
            stag = (globtag + csd * nsubd + nsd) % maxtag  # Senders number *nsubd + Receivers number
            rreqs.append(comm.Irecv([rbuf[rq], MPI.COMPLEX], source=nown, tag=rtag))
            rq = rq + 1
            sbuf.append(y[p][2 * OL, strt:endt].flatten())
            sreqs.append(comm.Isend([sbuf[sq], MPI.COMPLEX], dest=nown, tag=stag))
            sq = sq + 1
            if DEBA: print(rank, p, SubDomain[p, 0], sq, 'sent to S', nown, stag)
        if SubDomain[p, 2] < N:  # not the upmost subdomain N
            if SubDomain[p, 3] > 0:  # not the leftmost subdomain W
                strt = 1
            else:
                strt = 0
            if SubDomain[p, 4] < N:  # not the rightmost subdomain E
                endt = y[p].shape[1] - 1
            else:
                endt = y[p].shape[1]
            rbuf.append(empty((y[p][-1, strt:endt]).size + 16, dtype=complex))
            nsd = csd + M_subd  # Neighbour SubDomain #
            nown = DomainProc[nsd]  # Neighbour's OWNer process #
            rtag = (globtag + nsd * nsubd + csd) % maxtag  # Senders number *nsubd + Receivers number
            stag = (globtag + csd * nsubd + nsd) % maxtag  # Senders number *nsubd + Receivers number
            rreqs.append(comm.Irecv([rbuf[rq], MPI.COMPLEX], source=nown, tag=rtag))
            rq = rq + 1
            sbuf.append(y[p][-2 * OL - 1, strt:endt].flatten())
            sreqs.append(comm.Isend([sbuf[sq], MPI.COMPLEX], dest=nown, tag=stag))
            sq = sq + 1
            if DEBA: print(rank, p, SubDomain[p, 0], sq, 'sent to N', nown, stag)
            # print p,'4 sent:',sbuf[sq]
        if SubDomain[p, 1] > 0 and SubDomain[p, 3] > 0:  # has SW neighbour
            rbuf.append(empty((y[p][0, 0]).size + 16, dtype=complex))
            nsd = csd - M_subd - 1  # Neighbour SubDomain #
            nown = DomainProc[nsd]  # Neighbour's OWNer process #
            rtag = (globtag + nsd * nsubd + csd) % maxtag  # Senders number *nsubd + Receivers number
            stag = (globtag + csd * nsubd + nsd) % maxtag  # Senders number *nsubd + Receivers number
            rreqs.append(comm.Irecv([rbuf[rq], MPI.COMPLEX], source=nown, tag=rtag))
            rq = rq + 1
            sbuf.append(y[p][2 * OL, 2 * OL].flatten())
            sreqs.append(comm.Isend([sbuf[sq], MPI.COMPLEX], dest=nown, tag=stag))
            sq = sq + 1
            if DEBA: print(rank, p, SubDomain[p, 0], sq, 'sent to SW', nown, stag)
        if SubDomain[p, 2] < N and SubDomain[p, 4] < N:  # has NE neighbour
            rbuf.append(empty((y[p][-1, -1]).size + 16, dtype=complex))
            nsd = csd + M_subd + 1  # Neighbour SubDomain #
            nown = DomainProc[nsd]  # Neighbour's OWNer process #
            rtag = (globtag + nsd * nsubd + csd) % maxtag  # Senders number *nsubd + Receivers number
            stag = (globtag + csd * nsubd + nsd) % maxtag  # Senders number *nsubd + Receivers number
            rreqs.append(comm.Irecv([rbuf[rq], MPI.COMPLEX], source=nown, tag=rtag))
            rq = rq + 1
            sbuf.append(y[p][-2 * OL - 1, -2 * OL - 1].flatten())
            sreqs.append(comm.Isend([sbuf[sq], MPI.COMPLEX], dest=nown, tag=stag))
            sq = sq + 1
            if DEBA: print(rank, p, SubDomain[p, 0], sq, 'sent to NE', nown, stag)
        if SubDomain[p, 3] > 0 and SubDomain[p, 2] < N:  # has NW neighbour
            rbuf.append(empty((y[p][-1, 0]).size + 16, dtype=complex))
            nsd = csd + M_subd - 1  # Neighbour SubDomain #
            nown = DomainProc[nsd]  # Neighbour's OWNer process #
            rtag = (globtag + nsd * nsubd + csd) % maxtag  # Senders number *nsubd + Receivers number
            stag = (globtag + csd * nsubd + nsd) % maxtag  # Senders number *nsubd + Receivers number
            rreqs.append(comm.Irecv([rbuf[rq], MPI.COMPLEX], source=nown, tag=rtag))
            rq = rq + 1
            sbuf.append(y[p][-2 * OL - 1, 2 * OL].flatten())
            sreqs.append(comm.Isend([sbuf[sq], MPI.COMPLEX], dest=nown, tag=stag))
            sq = sq + 1
            if DEBA: print(rank, p, SubDomain[p, 0], sq, 'sent to NW', nown, stag)
        if SubDomain[p, 4] < N and SubDomain[p, 1] > 0:  # has SE neighbour
            rbuf.append(empty((y[p][0, -1]).size + 16, dtype=complex))
            nsd = csd - M_subd + 1  # Neighbour SubDomain #
            nown = DomainProc[nsd]  # Neighbour's OWNer process #
            rtag = (globtag + nsd * nsubd + csd) % maxtag  # Senders number *nsubd + Receivers number
            stag = (globtag + csd * nsubd + nsd) % maxtag  # Senders number *nsubd + Receivers number
            rreqs.append(comm.Irecv([rbuf[rq], MPI.COMPLEX], source=nown, tag=rtag))
            rq = rq + 1
            sbuf.append(y[p][2 * OL, -2 * OL - 1].flatten())
            sreqs.append(comm.Isend([sbuf[sq], MPI.COMPLEX], dest=nown, tag=stag))
            sq = sq + 1
            if DEBA: print(rank, p, SubDomain[p, 0], sq, 'sent to SE', nown, stag)
    for p in range(n_mysubd):  # now calculate the local part...
        A[p][1] = aslinearoperator(A[p][1])
        y[p] = y[p] + (A[p][1].matvec(x[p].ravel())).reshape(y[p].shape)
    # check that the communication ended and continue
    if DEBA: print('sreqs=', len(sreqs))
    MPI.Request.Waitall(sreqs)  # here finished with all sends.. (remove?)
    if DEBA: print(rank, 'sreqs -- got all', len(sreqs))
    rq = 0
    for p in range(n_mysubd):
        if SubDomain[p, 3] > 0:  # not the leftmost subdomain
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'waiting from W')
            MPI.Request.Wait(rreqs[rq])
            if SubDomain[p, 1] > 0:  # not the downmost subdomain S
                strt = 1
            else:
                strt = 0
            if SubDomain[p, 2] < N:  # not the upmost subdomain N
                endt = y[p].shape[0] - 1
            else:
                endt = y[p].shape[0]
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'got from W')  # ,'rbuf is:',rbuf[rq]
            y[p][strt:endt, 0] = rbuf[rq][0:endt - strt]
            rq = rq + 1
        if SubDomain[p, 4] < N:  # not the rightmost subdomain
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'waiting from E')  # ,'rbuf is:',rbuf[rq]
            MPI.Request.Wait(rreqs[rq])
            if SubDomain[p, 1] > 0:  # not the downmost subdomain S
                strt = 1
            else:
                strt = 0
            if SubDomain[p, 2] < N:  # not the upmost subdomain N
                endt = y[p].shape[0] - 1
            else:
                endt = y[p].shape[0]
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'got from E')  # ,'rbuf is:',rbuf[rq]
            y[p][strt:endt, -1] = rbuf[rq][0:endt - strt]
            rq = rq + 1
        if SubDomain[p, 1] > 0:  # not the downmost subdomain
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'waiting from S')  # ,'rbuf is:',rbuf[rq]
            MPI.Request.Wait(rreqs[rq])
            if SubDomain[p, 3] > 0:  # not the leftmost subdomain W
                strt = 1
            else:
                strt = 0
            if SubDomain[p, 4] < N:  # not the rightmost subdomain E
                endt = y[p].shape[1] - 1
            else:
                endt = y[p].shape[1]
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'got from S')  # ,'rbuf is:',rbuf[rq]
            y[p][0, strt:endt] = rbuf[rq][0:endt - strt]
            rq = rq + 1
        if SubDomain[p, 2] < N:  # not the upmost subdomain
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'waiting from N')  # ,'rbuf is:',rbuf[rq]
            MPI.Request.Wait(rreqs[rq])
            if SubDomain[p, 3] > 0:  # not the leftmost subdomain W
                strt = 1
            else:
                strt = 0
            if SubDomain[p, 4] < N:  # not the rightmost subdomain E
                endt = y[p].shape[1] - 1
            else:
                endt = y[p].shape[1]
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'got from N')  # ,'rbuf is:',rbuf[rq]
            y[p][-1, strt:endt] = rbuf[rq][0:endt - strt]
            rq = rq + 1
        if SubDomain[p, 1] > 0 and SubDomain[p, 3] > 0:  # has SW neighbour
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'waiting from SW')  # ,'rbuf is:',rbuf[rq]
            MPI.Request.Wait(rreqs[rq])
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'got from SW')  # ,'rbuf is:',rbuf[rq]
            y[p][0, 0] = rbuf[rq][0]
            rq = rq + 1
        if SubDomain[p, 2] < N and SubDomain[p, 4] < N:  # has NE neighbour
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'waiting from NE')  # ,'rbuf is:',rbuf[rq]
            MPI.Request.Wait(rreqs[rq])
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'got from NE')  # ,'rbuf is:',rbuf[rq]
            y[p][-1, -1] = rbuf[rq][0]
            rq = rq + 1
        if SubDomain[p, 3] > 0 and SubDomain[p, 2] < N:  # has NW neighbour
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'waiting from NW')  # ,'rbuf is:',rbuf[rq]
            MPI.Request.Wait(rreqs[rq])
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'got from NW')  # ,'rbuf is:',rbuf[rq]
            y[p][-1, 0] = rbuf[rq][0]
            rq = rq + 1
        if SubDomain[p, 4] < N and SubDomain[p, 1] > 0:  # has SE neighbour
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'waiting from SE')  # ,'rbuf is:',rbuf[rq]
            MPI.Request.Wait(rreqs[rq])
            if DEBA: print(rank, p, SubDomain[p, 0], rq, 'got from SE')  # ,'rbuf is:',rbuf[rq]
            y[p][0, -1] = rbuf[rq][0]
            rq = rq + 1
    globtag = (globtag + (nsubd + 1) * nsubd) % maxtag
    return y


def Generate_random(DEBA=False):
    global N
    global DomainProc, SubDomain, nprocs, comm, rank, OL, M_subd, globtag, maxtag
    global OshapeD, InactiveNodes
    if rank == 0: print('Generating random initial guess')
    x0 = []
    n_mysubd = SubDomain.shape[0]
    for p in range(n_mysubd):
        x0.append(zeros(GLOBALS[p].shape, dtype=complex))
        x0[p] = x0[p] + (random.random(GLOBALS[p].shape) + random.random(GLOBALS[p].shape) * 1j)
    OL_update(x0, Force_Averaging=True)
    if OshapeD:  # switch the values in inactive nodes to 0.0
        for p in range(n_mysubd):
            x0[p] = x0[p] * InactiveNodes[p]
    return x0


def zsolupcont(H, hcount, iter, nsteps, krylsize, s, VV, sol, check=False):
    # this subroutine continues the updates of the solution with
    # nsteps further components
    #
    # Solve triangular system for y: Hy = s
    #   and update the sulution
    ''' The Matrix H is given in the following order:
     (Fortran)  | 1 2 4 7 |      NumPy:  | 0 1 3 6 |
                |   3 5 8 |              |   2 4 7 |
                |     6 9 |              |     5 8 |
                |      10 |              |       9 |
    '''
    global GLOBALS
    global DomainProc, SubDomain, nprocs, comm, rank, OL, M_subd, globtag, maxtag
    n_mysubd = SubDomain.shape[0]
    too_small = 0
    y = zeros(krylsize, dtype=complex)
    # do i=hcount,hcount-iter+1,-1
    # pdb.set_trace()
    # print hcount,iter,nsteps,krylsize,' HHHH H=',H
    # for i in range(hcount,hcount-iter+1,-1):
    # print 'i in range',range(hcount-1,hcount-iter-1,-1)
    for i in range(hcount - 1, hcount - iter - 1, -1):
        j = i - hcount + iter
        # print '  j=',j
        if j > iter - nsteps - 1:
            y[j] = s[j]
        else:
            y[j] = 0.0j
        hk = i
        # do k=iter-1,j,-1
        # print '  k in ',range(iter-2,j-1,-1)
        for k in range(iter - 2, j - 1, -1):
            # print  '  y[',j,']-H[',hk,']*y[',k+1,']'
            y[j] = y[j] - H[hk] * y[k + 1]
            hk = hk - k - 1
            # print '    hk=',hk
        if abs(y[j]) < abs(H[hk]) * 1e-16:
            too_small = too_small + 1
            # print 'too_small',too_small
            y[j] = 0.0j
        else:
            # print  '  y[',j,']/H[',hk,'] ::: ',y[j],H[hk]
            y[j] = y[j] / H[hk]
    if check:
        # -------Checking that y is solution of the system Hy=s --+
        #       ! The check is relevant only with the whole syst!
        c = zeros(iter, dtype=complex)
        if nsteps == iter:
            i = 0
            for k in range(iter):
                for j in range(k + 1):
                    # print  '  c[',j,']+H[',i,']*y[',k,']'
                    c[j] = c[j] + H[i] * y[k]
                    i = i + 1
            tmp = 0.0
            for j in range(iter):
                # print c[j],s[j]
                tmp = tmp + abs(c[j] - s[j])
            print(rank, ': SOL ERROR:', tmp)
        else:
            print('can check only ifnsteps==iter ', nsteps, iter)
    # --------------------------------------------------------+
    # x = x_0 + y_1*v^(1) + y_2*v^(2) + ... + y_i*v^(i)
    for p in range(n_mysubd):
        for kk in range(iter):
            sol[p] = sol[p] + y[kk] * VV[p][kk]
        sol[p] = sol[p].reshape(GLOBALS[p].shape)
    return sol


def zpgmres(A, b, M=None, x0=None, tol=1e-6, krylsize=None, callback=None):
    '''Generalized Minimum Residual Method, Flexible, Right-Left prec.
       translated from DOUG code (f77 for Navier-Stokes eigenvalue calc...)
    '''
    global N, GLOBALS
    global DomainProc, SubDomain, nprocs, comm, rank, OL, M_subd, globtag, maxtag
    global OshapeD, InactiveNodes

    def wdot(xx, Dk, yy, dim1=None, DEB=False):
        n_mysubd = SubDomain.shape[0]
        if Dk == None:
            # return dot(conjugate(xx),yy)
            if dim1 == None:
                if DEB:
                    print('xx.shapes:')
                    for p in range(n_mysubd):
                        print(rank, p, xx[p].shape)
                nrm = zeros(1, dtype=complex)
                for p in range(n_mysubd):
                    if xx[p].ndim == 1:
                        xxx = xx[p].reshape(GLOBALS[p].shape)
                    else:
                        xxx = xx[p]
                    # take the unique's slice
                    xs = xxx[SubDomain[p, 5]:SubDomain[p, 6], SubDomain[p, 7]:SubDomain[p, 8]]
                    # and ravel it:
                    nrm = nrm + dot(conjugate(xs.ravel()), \
                                    ravel(yy[p][SubDomain[p, 5]:SubDomain[p, 6], SubDomain[p, 7]:SubDomain[p, 8]]))
                    if DEB:
                        print(rank, p, ' nrm=', nrm)
            else:
                nrm = zeros(dim1, dtype=complex)
                for p in range(n_mysubd):
                    if xx[p].ndim == 1:  # reshape according to last 2 indeces
                        print('xx[p] has only 1 dim???')
                        exit(0)
                    elif xx[p].ndim == 2:  # reshape according to last 2 indeces
                        xxx = xx[p][0:dim1].reshape((dim1, GLOBALS[p].shape[0], GLOBALS[p].shape[1]))
                    else:  # has 3 dims already
                        xxx = xx[p]
                    # take the unique's slice
                    xs = xxx[:dim1, SubDomain[p, 5]:SubDomain[p, 6], SubDomain[p, 7]:SubDomain[p, 8]]
                    # ravel by 2 and 3rd dim:
                    xs = xs.reshape((dim1, xs.shape[1] * xs.shape[2]))
                    # and ravel it together in 2rd and 3rd dim:
                    nrm = nrm + dot(conjugate(xs), \
                                    ravel(yy[p][SubDomain[p, 5]:SubDomain[p, 6], SubDomain[p, 7]:SubDomain[p, 8]]))
        else:
            print('wdot Dk case not written yet...')
            exit(0)
        if DEB:
            print(rank, p, 'nrm before Allreduce:', nrm)
        nrm2 = comm.allreduce(nrm)
        if DEB:
            print(rank, p, 'nrm2 after Allreduce:', nrm2, nrm3)
        return nrm2

    zsolver = 'FGMRES'
    if krylsize is None:
        krylsize = N

    # krylsize=8

    n_mysubd = SubDomain.shape[0]
    x = list(range(n_mysubd))
    r = list(range(n_mysubd))
    z = list(range(n_mysubd))
    v = list(range(n_mysubd))
    pp = list(range(n_mysubd))
    VV = list(range(n_mysubd))
    for p in range(n_mysubd):
        # x[p]=zeros(SubDomain[p,9],dtype=complex)
        # r[p] = b[p].copy()
        z[p] = zeros(SubDomain[p, 9], dtype=complex)
        v[p] = zeros(SubDomain[p, 9], dtype=complex)
        pp[p] = zeros(SubDomain[p, 9], dtype=complex)
        VV[p] = zeros((krylsize + 1, SubDomain[p, 9]), dtype=complex)
    if zsolver == 'FGMRES':
        MV = list(range(n_mysubd))
        for p in range(n_mysubd):
            MV[p] = zeros((krylsize + 1, SubDomain[p, 9]), dtype=complex)
    itgsdots = zeros(krylsize + 1, dtype=complex)
    nupdated = 0
    globits = 0
    Dk = None
    # TODO -- nonzero initial guess -- calculate residual r! -- siin
    if x0 == None:
        for p in range(n_mysubd):
            x[p] = zeros(SubDomain[p, 9], dtype=complex)
            r[p] = b[p].copy()
        norm_b = norm(b)
    else:  # nonzero initial guess
        for p in range(n_mysubd):
            x[p] = x0[p]
            r[p] = zeros(SubDomain[p, 9], dtype=complex)
        pp = Ax_op(A, x)
        for p in range(n_mysubd):
            r[p] = b[p] - pp[p]
        norm_b = norm(r)
        if rank == 0: print('zpgmres: Nonzero initial guess')

    if norm_b != 0.0:
        tol = tol * norm_b
    for outer_it in range(krylsize):
        if zsolver == 'FGMRES':
            for p in range(n_mysubd):
                z[p] = r[p].copy()
        else:
            z = M(r)
        # pdb.set_trace()
        dotp = norm(z)

        if callback is not None and globits > 0:
            callback(dotp)
        for p in range(n_mysubd):
            z[p] = z[p] / dotp
            VV[p][0] = z[p].ravel()  # store z in VV

        H = zeros((krylsize + 1) * krylsize // 2 + 1, dtype=complex)
        giv1 = zeros(krylsize, dtype=float)
        giv2 = zeros(krylsize, dtype=complex)
        s = zeros(krylsize + 1, dtype=complex)
        s[0] = dotp
        kk = 0
        hcount = 0
        for inner_it in range(krylsize):
            if kk > 0:
                for p in range(n_mysubd):
                    z[p] = VV[p][kk].reshape(GLOBALS[p].shape)
            if zsolver == 'FGMRES':
                v = M(z)  # Apply the preconditioner MV[kk]=M(VV[kk])
                for p in range(n_mysubd):  # store v in MV
                    MV[p][kk] = v[p].ravel()

                # pdb.set_trace()
                pp = Ax_op(A, v)
            else:  # Left preconditioning
                v = Ax_op(A, z)
                pp = M(v)
            # Classical Gram-Schmidt:
            for orth_trips in range(2):
                itgsdots[0:kk + 1] = wdot(VV, Dk, pp, dim1=kk + 1)
                for p in range(n_mysubd):
                    dtmp = dot(VV[p][0:kk + 1].T, itgsdots[0:kk + 1])
                    pp[p] = pp[p] - dtmp.reshape(GLOBALS[p].shape)
                H[hcount:hcount + kk + 1] = H[hcount:hcount + kk + 1] + itgsdots[0:kk + 1]
            # print 'added values to H[',hcount,':',hcount+kk,']'
            hcount = hcount + kk + 1
            # pdb.set_trace()
            # maybe check that pp orthogonal to VV?
            H_subd = norm(pp)
            for p in range(n_mysubd):
                VV[p][kk + 1] = pp[p].ravel() / H_subd
            # Apply J_{1},...,J_{i-1} on (H_{1,i},...,H_{i+1,i})
            #      / giv1 -giv2^H \
            # J = <                >
            #      \ giv2   giv1  /
            #####for kkk in range(kk-1):
            i = hcount - kk - 1
            # print kk,'  hcount=',hcount
            # print kk,'hcount-kk-1:::',i,'...',i+kk
            # pdb.set_trace()
            for kkk in range(kk):
                # print 'kk,hcount,kkk,i:::',kk,hcount,kkk,i,i+kkk
                ztmp = H[i + kkk]
                H[i + kkk] = giv1[kkk] * H[i + kkk] + giv2[kkk].conjugate() * H[i + kkk + 1]
                H[i + kkk + 1] = giv1[kkk] * H[i + kkk + 1] - giv2[kkk] * ztmp
            # Construct J_{i}:
            # print 'updating H[',hcount-1,']'
            dotp = sqrt(abs(H[hcount - 1]) ** 2 + abs(H_subd) ** 2)
            if abs(H[hcount - 1]) != 0.0:
                giv2[kk] = H_subd * abs(H[hcount - 1]) / (H[hcount - 1] * dotp)
                giv1[kk] = abs(H[hcount - 1]) / dotp
            elif abs(H_subd) != 0.0:
                giv1[kk] = 0.0
                giv2[kk] = H_subd / abs(H_subd)
            else:
                giv1[kk] = 1.0
                giv2[kk] = 0.0j
            # Apply J_{i} to H_{*,i} and s:
            H[hcount - 1] = giv1[kk] * H[hcount - 1] + giv2[kk].conjugate() * H_subd
            s[kk + 1] = -giv2[kk] * s[kk]
            s[kk] = giv1[kk] * s[kk]

            res = abs(s[kk + 1])
            if callback is not None:
                callback(res)
            # ctol=res*res/norm_b
            # if ctol <= tol**2 or kk==krylsize-1:
            if res < tol or kk == krylsize - 1:
                if zsolver == 'FGMRES':
                    zsolupcont(H, hcount, kk + 1, kk + 1 - nupdated, krylsize, s, MV, x, check=False)
                else:
                    zsolupcont(H, hcount, kk + 1, kk + 1 - nupdated, krylsize, s, VV, x)
                return (x, 0)
            kk = kk + 1
            # if globtag > 32000: globtag=0
        globits = globits + 1


def weighted_gmres(A, b, M=None, restart=None, tol=None, x0=None,
                   maxiter=None, hard_failure=None, require_monotonicity=True,
                   no_progress_factor=None, stall_iterations=None,
                   callback=None, Dk=None, CGS=False):
    global N
    global DomainProc, SubDomain, nprocs, comm, rank, OL, M_subd, globtag, maxtag

    def wdot(xx, Dk, yy, dim1=None, DEB=False):
        if Dk == None:
            # return dot(conjugate(xx),yy)
            if dim1 == None:
                if DEB:
                    print('xx.shapes:')
                    for p in range(n_mysubd):
                        print(rank, p, xx[p].shape)
                nrm = zeros(1, dtype=complex)
                for p in range(n_mysubd):
                    if xx[p].ndim == 1:
                        xxx = xx[p].reshape(GLOBALS[p].shape)
                    else:
                        xxx = xx[p]
                    # take the unique's slice
                    xs = xxx[SubDomain[p, 5]:SubDomain[p, 6], SubDomain[p, 7]:SubDomain[p, 8]]
                    # and ravel it:
                    # print p,'dot ',conjugate(xs.ravel()).shape, \
                    #    (ravel(yy[p][SubDomain[p,5]:SubDomain[p,6],SubDomain[p,7]:SubDomain[p,8]])).shape
                    nrm = nrm + dot(conjugate(xs.ravel()),
                                    ravel(yy[p][SubDomain[p, 5]:SubDomain[p, 6], SubDomain[p, 7]:SubDomain[p, 8]]))
                    if DEB:
                        print(rank, p, ' nrm=', nrm)
            else:
                nrm = zeros(dim1, dtype=complex)
                for p in range(n_mysubd):
                    # print 'xx ,xx[p].shape',xx[p].shape
                    # print 'xx ,xx[p].ndim',xx[p].ndim
                    # print 'xx x[p]=',x[p]
                    if xx[p].ndim == 1:  # reshape according to last 2 indeces
                        print('xx[p] has only 1 dim???')
                        exit(0)
                    elif xx[p].ndim == 2:  # reshape according to last 2 indeces
                        # print p,'xx xx ',dim1,GLOBALS[p].shape[0],GLOBALS[p].shape[1]
                        xxx = xx[p][0:dim1].reshape((dim1, GLOBALS[p].shape[0], GLOBALS[p].shape[1]))
                    else:  # has 3 dims already
                        xxx = xx[p]
                    # take the unique's slice
                    xs = xxx[:dim1, SubDomain[p, 5]:SubDomain[p, 6], SubDomain[p, 7]:SubDomain[p, 8]]
                    # ravel by 2 and 3rd dim:
                    xs = xs.reshape((dim1, xs.shape[1] * xs.shape[2]))
                    # and ravel it together in 2rd and 3rd dim:
                    nrm = nrm + dot(conjugate(xs),
                                    ravel(yy[p][SubDomain[p, 5]:SubDomain[p, 6], SubDomain[p, 7]:SubDomain[p, 8]]))
            # else:
            #    print 'wdot with 2D arrays not existing...'
            #    exit(0)
        else:
            # return dot(xx.conj(),Dk.matvec(yy))
            print('wdot Dk case not written yet...')
            exit(0)
        # if dim1==None:
        #    nrm2=zeros(1,dtype=complex)
        # else:
        #    nrm2=zeros(dim1,dtype=complex)
        if DEB:
            print(rank, p, 'nrm before Allreduce:', nrm)
        # comm.Allreduce([nrm,MPI.COMPLEX],[nrm2,MPI.COMPLEX])
        nrm2 = comm.allreduce(nrm)
        if DEB:
            print(rank, p, 'nrm2 after Allreduce:', nrm2, nrm3)
        return nrm2

    def value_print(v, txt=''):
        icount = 0
        for i in range(M_subd ** 2):
            if rank == DomainProc[i]:
                print(i, rank, txt, v[icount])
                icount = icount + 1
            comm.Barrier()

    # A=aslinearoperator(A)
    # if M is not None:
    #    M=aslinearoperator(M)

    if restart is None:
        restart = maxiter

    if tol is None:
        tol = 1e-6

    if maxiter is None:
        maxiter = 2 * N

    if hard_failure is None:
        hard_failure = True

    if stall_iterations is None:
        stall_iterations = 10
    if no_progress_factor is None:
        no_progress_factor = 1.025

    n_mysubd = SubDomain.shape[0]
    Ae = list(range(n_mysubd))
    e = list(range(n_mysubd))
    if x0 is None:
        x = list(range(n_mysubd))
        r = list(range(n_mysubd))
        for p in range(n_mysubd):
            x[p] = zeros(GLOBALS[p].shape, dtype=complex)
            r[p] = b[p].copy()
        if M is not None:
            # r=M.matvec(r)
            r = M(r)
        recalc_r = False
    else:
        for p in range(n_mysubd):
            x[p] = x0[p]
        del x0
        recalc_r = True

    for p in range(n_mysubd):
        # Ae[p]=zeros((restart,GLOBALS[p].shape[0],GLOBALS[p].shape[1]),dtype=complex)
        # e[p]=zeros((restart,GLOBALS[p].shape[0],GLOBALS[p].shape[1]),dtype=complex)
        Ae[p] = zeros((restart, SubDomain[p, 9]), dtype=complex)
        e[p] = zeros((restart, SubDomain[p, 9]), dtype=complex)

    kk = 0

    norm_b = norm(b, Dk)
    last_resid_norm = None
    norm_r = 0.0
    stall_count = 0
    residual_norms = []
    Dk = None
    for iteration in range(maxiter):
        # restart if required
        if kk == restart:
            kk = 0
            orth_count = restart
        else:
            orth_count = kk

        # recalculate residual every 10 steps
        if recalc_r:
            #    r = b - A.matvec(x)
            # tmpx=empty((N,N),dtype=complex)
            # for p in range(n_mysubd):
            #    tmpx[SubDomain[p,1]:SubDomain[p,2],SubDomain[p,3]:SubDomain[p,4]]=x[p][:,:]
            # print 'tmpx=',tmpx
            y = Ax_op(A, x)
            for p in range(n_mysubd):
                r[p] = b[p] - y[p]
            ### apply preconditioner
            # r = m_call(r)
            if M is not None:
                r = M(r)
        norm_r = norm(r, Dk)
        # debugging parallel:
        # tmpr=empty((N,N),dtype=complex)
        # for p in range(n_mysubd):
        #    tmpr[SubDomain[p,1]:SubDomain[p,2],SubDomain[p,3]:SubDomain[p,4]]=r[p][:,:]
        # print 'tmpr=',tmpr
        # pdb.set_trace()
        # deb...
        residual_norms.append(norm_r)
        if callback is not None and iteration > 0:
            callback(norm_r)
        # print 'nnorm_r,nnorm_b:',norm_r,norm_b
        if norm_r < tol * norm_b:
            return x, 0

        if last_resid_norm is not None:
            if norm_r > 1.25 * last_resid_norm:
                state = "non-monotonic residuals"
                if require_monotonicity:
                    if hard_failure:
                        # raise GMRESError(state)
                        if rank == 0: print("raise GMRESError(state) non-monotonic res hard", state)
                    else:
                        # return GMRESResult(solution=x,
                        #        residual_norms=residual_norms,
                        #        iteration_count=iteration, success=False,
                        #        state=state)
                        if rank == 0: print("raise GMRESError(state) non-monotonic res soft", state)
                        return x, 0
                else:
                    if rank == 0: print("*** WARNING: non-monotonic residuals in GMRES")

            if stall_iterations and \
                    norm_r > last_resid_norm / no_progress_factor:
                stall_count += 1
                if stall_count >= stall_iterations:
                    state = "stalled"
                    if hard_failure:
                        # raise GMRESError(state)
                        if rank == 0: print("raise GMRESError(state)", state)
                    else:
                        # return GMRESResult(solution=x,
                        #        residual_norms=residual_norms,
                        #        iteration_count=iteration, success=False,
                        #        state=state)
                        if rank == 0: print('wgmres returning due to stalled state#########')
                        return x, 0
            else:
                stall_count = 0

        last_resid_norm = norm_r

        # initial new direction guess
        # pdb.set_trace()
        w = Ax_op(A, r)

        ### apply preconditioner
        if M is not None:
            w = M(w)

        # double-orthogonalize the new direction against preceding ones
        # rp = r
        rp = list(range(n_mysubd))
        for p in range(n_mysubd):
            rp[p] = r[p].copy()

        for orth_trips in range(2):
            # Classical Gram-Schmidt (or modified one)
            ##dd = wdot(Ae[0:orth_count+1],Dk,w)
            dd = wdot(Ae, Dk, w, dim1=orth_count + 1)
            # print rank,'   -- dddd dd=',dd
            for p in range(n_mysubd):
                # print p,'www w[p].shape, Ae[p][0:orth+1].shape:',w[p].shape,Ae[p][0:orth_count+1].shape
                dtmp = dot(Ae[p][0:orth_count + 1].T, dd)
                w[p] = w[p] - dtmp.reshape(GLOBALS[p].shape)
                dtmp = dot(e[p][0:orth_count + 1].T, dd)
                rp[p] = rp[p] - dtmp.reshape(GLOBALS[p].shape)
            # normalize
            d = 1. / norm(w, Dk)
            # print rank,'   -- dddd d=',d
            for p in range(n_mysubd):
                w[p] = d * w[p]
                rp[p] = d * rp[p]
        for p in range(n_mysubd):
            Ae[p][kk] = w[p].ravel()
            e[p][kk] = rp[p].ravel()
        d = wdot(w, Dk, r)
        recalc_r = (iteration + 1) % 10 == 0
        if not recalc_r:
            for p in range(n_mysubd):
                tmp = Ae[p][kk].reshape(r[p].shape)
                r[p] = r[p] - d * tmp
        for p in range(n_mysubd):
            x[p] = x[p] + d * e[p][kk].reshape(x[p].shape)
        kk += 1
    state = "max iterations"
    if hard_failure:
        if rank == 0: print("raise GMRESError(state)", state)
        return x, 0
    else:
        return x, 0


def gmres(GMRES_VER, A, b, M, x0=None, tol=1e-6, restrt=600, callback=counter, Dk=None):
    global it, N
    global DomainProc, SubDomain, nprocs, comm, rank, globtag, maxtag
    it = 0
    if rank == 0: print('   Using Krylov method: ', GMRES_VER)
    if GMRES_VER == 'fgmres':
        xa, info = zpgmres(A, b, M=M, x0=x0, tol=tol, krylsize=restrt, callback=callback)
    elif GMRES_VER == 'gmres':
        xa, info = pyamg_gmres(A, b, M=M, x0=x0, tol=tol, callback=callback)
    elif GMRES_VER == 'gmres_mgs':
        xa, info = gmres_mgs(A, b, M=M, x0=x0, tol=tol, callback=callback)
    elif GMRES_VER == 'gmres_cgs':
        xa, info = gmres_cgs(A, b, M=M, x0=x0, tol=tol, callback=callback)
    elif GMRES_VER == 'wDgmres':
        xa, info = weighted_gmres(A, b, M=M, x0=x0, tol=tol, maxiter=restrt, callback=callback, Dk=Dk)
    elif GMRES_VER == 'wgmres':
        xa, info = weighted_gmres(A, b, M=M, x0=x0, tol=tol, maxiter=restrt, callback=callback)
    else:
        xa, info = scipy_gmres(A, b, M=M, tol=tol, restrt=restrt, callback=callback)
    # check_nd_plot_global_vec(xa,'Solution:')
    ###check_nd_plot3d_global_vec(xa,'Solution:')
    n_mysubd = SubDomain.shape[0]
    axxa = Ax_op(A, xa)
    tmp = list(range(n_mysubd))
    for p in range(n_mysubd):
        tmp[p] = axxa[p] - b[p]
    nrm = norm(tmp, Dk=None)
    if rank == 0: print('  residual norm:', nrm, ' ####it:', it)
    if x0 == None:
        normb = norm(b)
    else:
        axxa = Ax_op(A, x0)
        tmp = list(range(n_mysubd))
        for p in range(n_mysubd):
            tmp[p] = axxa[p] - b[p]
        normb = norm(tmp, Dk=None)
    if nrm > tol * normb:
        if rank == 0: print('############ did it converge to the solution????  <--------')
        if x0 == None:
            if rank == 0: print('#### ||A*x0-b||=', normb)
        else:
            if rank == 0: print('#### ||b||=', normb)
        if rank == 0: print('#### norm(A*x-b)=', nrm, 'tol=', tol)
        if rank == 0: print('#### tol*||r0||=', tol * normb)
    return it


def HSolver(k_in, W_subd_in, M_subd_in, ep1_in, OL_in, AS_prec):
    global OL, Coarse_Drh_0, it, GLOBALS, P, N, AVERAGER, Use_Poisson
    global R, RT, A, Dk, A_c, n, m, scale_int, k, epsilon
    global eps_prec1, A_eps, Restricted_AS, Averaging, CHOOSER, GLOBALS_0
    global eps_prec2, b, GMRES_VER, M_subd, Robin
    # 3-level method:
    global GLOBALS_3, A_3, P_3, N_3, AVERAGER_3, n_3, epsilon_3, k, eps_prec1_3
    global A_eps_3, OL_3, CHOOSER_3, GLOBALS_0_3, M_subd_3, W_subd
    global Mhoriz_3, Mvert_3, Mhoriz2_3, Mvert2_3
    global iterations_3, iits, tol_3
    global DomainProc, SubDomain, nprocs, comm, rank, globtag, maxtag
    global t1
    global VarCoeff, Morig, Marmousi, Marmousi_c, OL_vary, MarMult_in, MarMult_out
    global MarMatch, Marshift
    global UseMarmousi

    set_globals()

    k = k_in
    W_subd = W_subd_in
    M_subd = M_subd_in
    eps_prec1 = ep1_in
    OL = OL_in

    DomainProc = zeros(M_subd ** 2, dtype=int)
    n_my = 0
    for p in range(M_subd ** 2):
        if AS_prec == 2 or AS_prec == 4 or AS_prec == 6:  # 2-level method?
            DomainProc[p] = (p + 1) % nprocs  # TODO: add other distribution options here...
        else:  # 1-level method
            DomainProc[p] = p % nprocs  # TODO: add other distribution options here...
        if DomainProc[p] == rank:
            n_my = n_my + 1
    SubDomain = zeros((n_my, 21), dtype=int)
    n_my = 0
    for p in range(M_subd ** 2):
        if DomainProc[p] == rank:
            SubDomain[n_my, 0] = p
            DomainProc[p] = -n_my - 1  # so that proc 0 will be -1 etc.
            n_my = n_my + 1
    if rank == 0:
        DP = DomainProc.copy()
        for p in range(len(DP)):
            if DP[p] < 0:
                DP[p] = 0
        DPR = DP.reshape((M_subd, M_subd))
        DPT = DPR[-1::-1, :]
        print('Subdomain distribution among processes:')
        print(DPT)
    if OL < 0:
        OL = -OL
        N = (W_subd - 1) * M_subd + 1  # number of freedoms in each direction
        W_subd = (N - 1) // M_subd + 1
        GLOBALS = create_varsize_subdomain_indeces(M_subd, N, OL)
    else:
        N = (W_subd - 1) * M_subd + 1  # initial number of freedoms in each direction
        ### To get equal-sized subdomains, we expand the outermost one by OL:
        W_subd = (N - 1) // M_subd + 1
        N = N + 2 * OL
        if rank == 0:
            print('Expanding outer subd-s by', OL, 'layers, N=', N)
        GLOBALS = create_eqsize_subdomain_indeces(M_subd, N, OL)
    if VarCoeff:
        Marmousi = ones((N - 1, N - 1))
        if not UseMarmousi:
            if rank == 0:
                print('Diff multiplier in/out:', MarMult_in, MarMult_out)
                print('Generating 1/3 island of coeff')
            if MarMatch:
                if rank == 0:
                    print('Jumps matching the subdomain boundaries')
                lft = int(float(M_subd) / 3.0)
                rgt = int(float(2 * M_subd) / 3.0)
                Morig0 = zeros((M_subd, M_subd), dtype=float)
                Morig0[lft:rgt, lft:rgt] = 1.0
                Morig1 = ones((M_subd, M_subd), dtype=float)
                Morig1[lft:rgt, lft:rgt] = 0.0
            else:
                if rank == 0:
                    print('Jumps not taken to match subdomain boundaries')
                Morig0 = zeros((3, 3), dtype=float)
                Morig0[1, 1] = 1.0
                Morig1 = ones((3, 3), dtype=float)
                Morig1[1, 1] = 0.0
            Morig = Morig0.copy()
            Morig[:, :] = Morig0[:, :] * MarMult_in + Morig1[:, :] * MarMult_out
            if rank == 0:
                print('Morig==', Morig)
            Morigx = Morig.shape[0]
            Morigy = Morig.shape[1]

            #   M_subd kaudu!
            if Marshift != 0:
                if rank == 0:
                    print('<------ Shifting the 1/3 island NW by Marshift=', Marshift, 'cells')
            for i in range(N - 1):
                ii = min(i + Marshift, N - 2)
                for j in range(N - 1):
                    jj = min(j + Marshift, N - 2)
                    Marmousi[i, j] = Morig[int(float(ii) / (N - 1) * Morigx), int(float(jj) / (N - 1) * Morigy)]
    #
    Tol = 1e-6  ##################################################
    it = 0
    n = N * N
    P = list(range(M_subd ** 2))

    if Use_Poisson:
        A = Poisson(N)
        b = ones(n, dtype=float)
    else:
        if VarCoeff:
            A = helm_fe_var(N, k, C=Marmousi, rho=epsilon)
        else:
            A = helm_fe(N, k, epsilon)
        b = rhs(N, k)  # special rhs by Ivan
        if rank == 0: print('  ...using special RHS!')
        # check_nd_print_global_vec(b,'rhs')
        # exit(0)

    MAS = as_prec
    MC = counter
    MCP = prec_cp
    MD = precd

    if rank == 0: print('\n M_subd=', M_subd, ' # subd:', M_subd ** 2, ' of dim ', (W_subd - 1), '^2+')
    if Use_Poisson:
        if rank == 0: print('Poisson problem')
    else:
        if rank == 0: print(' epsilon:', epsilon, 'k:', k, 'eps_p:', eps_prec1, eps_prec2)

    if rank == 0: print('  UseTriangles=', UseTriangles)

    guess=1
    ###guess = 2
    n_mysubd = SubDomain.shape[0]
    if guess == 1:  # Using initial guess of ones
        x0 = []
        for p in range(n_mysubd):
            # x0.append(ones(GLOBALS[p].shape,dtype=complex))
            x0.append(ones(GLOBALS[p].size, dtype=complex))
        if OshapeD:  # switch the values in inactive nodes to 0.0
            for p in range(n_mysubd):
                x0[p] = x0[p] * InactiveNodes[p].ravel()
        if rank == 0: print('GMRES: Using initial guess of ones')
    elif guess == 2:  # Using random initial guess
        x0 = Generate_random()
        # check_nd_print_global_vec(x0,'x0')
        for p in range(n_mysubd):
            x0[p] = x0[p].flatten()
        if rank == 0: print('GMRES: Using initial guess of randoms')

    else:
        x0 = None
        if rank == 0: print('GMRES: Using initial guess of zeros')

    # t1=time()
    it = 0
    if AS_prec == 0:
        if rank == 0: print('  Un-preconditioned GMRES!')
        it2 = gmres(GMRES_VER, A, b, M=None, x0=x0, tol=Tol, restrt=1000, callback=counter, Dk=Dk)
    elif AS_prec == 1:
        if rank == 0: print('  1-Level Additive Schwarz preconditioner!')
        it2 = gmres(GMRES_VER, A, b, M=MAS, x0=x0, tol=Tol, restrt=600, callback=counter, Dk=Dk)
    return it2


def set_globals():
    global OL, Coarse_Drh_0, it, GLOBALS, P, N, AVERAGER, Use_Poisson
    global R, RT, A, Dk, A_c, n, m, scale_int, k, epsilon, GMRES_VER
    global eps_prec1, A_eps, Averaging, Restricted_AS, CHOOSER, GLOBALS_0
    global eps_prec2, b, M_subd, UseTriangles, it_3, iits
    global DomainProc, SubDomain, nprocs, comm, rank, globtag, maxtag
    global current_norm, ExtOL0

    OL = 0
    it = 0
    it_3 = 0
    iits = 0
    # tol_3=0.5e-1
    current_norm = 1.0
    GLOBALS = []
    GLOBALS_0 = []
    P = list(range(M_subd ** 2))
    M_subd = 1
    N = 1
    n = 1
    m = 1
    k = 1
    AVERAGER = zeros(1, dtype=float)
    Use_Poisson = False
    # Use_Poisson=True
    scale_int = False  # Set to False!!!
    A = scipy.sparse.coo_matrix(([0.], ([0], [0])), shape=(1, 1))
    A_c = A.copy()
    A_eps = A.copy()
    R = []
    RT = []
    CHOOSER = []
    b = zeros(1, dtype=complex)
    GMRES_VER = 'fgmres'
    # GMRES_VER='wgmres'
    # GMRES_VER='gmres'
    # GMRES_VER='scipy gmres'

    # Parameters to check out the effect of: #####################

    Averaging = 1  # 0,1,3 (1-Arithmetic, 2-Geometric, 3-testing)
    UseTriangles = False
    # UseTriangles=True


##### Globals: -- the values do not matter...  #################
OL = 0
Coarse_Drh_0 = False
it = 0
GLOBALS = []
M_coarse = 1  #
M_coarse_param = 0
M_subd = 1  #
P = list(range(M_subd ** 2))
N = 1
n = 1
m = 1
k = 1
W_coarse = 0
W_subd = 0  #
AVERAGER = zeros(1, dtype=float)
M_subd = 1  #
Use_Poisson = False
scale_int = True
epsilon = 0.0
eps_prec1 = 0.0  #
Explicit_Acoarse = True
GLOBALS_0 = []
eps_prec2 = 0.0
b = 1  #
A = scipy.sparse.coo_matrix(([0.], ([0], [0])), shape=(1, 1))  #
Dk = scipy.sparse.coo_matrix(([0.], ([0], [0])), shape=(1, 1))  #
A_c = A.copy()
A_eps = A.copy()
R = []  #
RT = []
Averaging = 1
Restricted_AS = True
CHOOSER = []  #
verbose = 10
GMRES_VER = 'wgmres'
Mhoriz = zeros(1, dtype=int)  #
Mvert = zeros(1, dtype=int)
Mhoriz2 = zeros(1, dtype=int)  #
Mvert2 = zeros(1, dtype=int)
UseTriangles = False
tol_3 = 0.01  #
Marmousi = zeros(1, dtype=float)
Marmousi_c = zeros(1, dtype=float)  #
Morig = zeros(1, dtype=float)
OL_vary = True
MarMatch = True  #
Marshift = 0
MarMult_in = 1.0
MarMult_out = 1.0  #
VarCoeff = False  #
UseMarmousi = False  #
# OshapeD=True # O-shape Domain                                 #
OshapeD = False  # O-shape Domain                                 #
InactiveNodes = []  #
CGtol = 1e-5
CGMaxIT = 256
################################################################
# for 3rd level:
################################################################
################################################################
# for parallelisation
#
# Matrix assemble is called only on subdomains[MyD[0:len(MyD)]]
#   (extended with overlap return will include A for // Ax op as well)
################################################################
# MyD=zeros(1,dtype=int) # subdomain numbers on current process???
SubDomain = zeros((1, 10), dtype=int)  # Subdomain properties (on my proc)
# [p,0] -- subdomain number (in global ordering)
# [p,1]:[p,2] -- (global) rownumbers of subdomain p
# [p,3]:[p,4] -- (global) colnumbers
# [p,5]:[p,6] -- (local) unique rownumbers
# [p,7]:[p,8] -- (local) unique colnumbers
# [p,9]       -- # subdomain nodes
#
comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()
DomainProc = -ones(nprocs, dtype=int)
globtag = 0
maxtag = 99999999
################################################################

verbose = 10

kk = 10  # set it here!

if len(sys.argv) != 4 and len(sys.argv) != 5 :
    if rank == 0:
        print("====> please supply 4 [5] arguments: subdomain_width number_of_subdomains [UseCG]")
    exit(0)
else:
    M_s = int(sys.argv[1])
    W_s = int(sys.argv[2])
    UseCG = int(sys.argv[3])
    if  len(sys.argv) == 5:
        CGMaxIT = int(sys.argv[4])

if UseCG==0:
    print('=== Using EXACT SubSolves!')
elif UseCG==1:
    print('=== Using',CGMaxIT,'iterations of GPGPU PyCL-CG with Single RHS SubSolves!')
elif UseCG==2:
    print('=== Using',CGMaxIT,'iterations of GPGPU PyCL-CG with Multiple RHS SubSolves!')
elif UseCG==11:
    print('=== Using',CGMaxIT,'iterations of GPGPU C_CL-CG with Single RHS SubSolves!')
elif UseCG==12:
    print('=== Using',CGMaxIT,'iterations of GPGPU C_CL-CG with Multiple RHS SubSolves!')
elif UseCG==3:
    print('=== Using',CGMaxIT,'iterations of NumPy-CG SubSolves!')
else:
    print('=== -- unknown SubSolver!')
    exit(0)


AS_prec = 1

if AS_prec == 1 or AS_prec == 5:
    if rank == 0: print("One-level AS preconditioner")
else:
    if rank == 0: print('undefine prec-type', AS_prec)
    exit(0)

kkk = 20
alpha = 0.5
beta = 1.0
NN = (W_s - 1) * M_s + 1
ol = int((W_s - 2) / 2)
# ol = 2

# ol=-ol # subdomains with different sizes on the edges (previous code)
if rank == 0:
    print('N=', NN, 'k=', kkk, 'alpha=', alpha, 'M_s=', M_s, 'W_s=', W_s, 'OL=', ol)
Robin = 1  # Impedence BC
epsilon = kkk ** (beta)
ep1 = epsilon
ep2 = epsilon
if rank == 0: print('----> setting epsilon=k^beta: ', epsilon)
cgs = [0,1,2,3,11,12]
times = []
times_pi = []
for cg in cgs:
    try:
        UseCG = cg
        t1 = time()
        its = HSolver(k_in=kkk, W_subd_in=W_s, M_subd_in=M_s, ep1_in=ep1, OL_in=ol, AS_prec=AS_prec)
        t2 = time()
        if rank == 0: print('Total time:',t2-t1, '(',(t2-t1)/60,'minutes )')
        if rank == 0: print('Aver. time per iter:',time_per_it/(its-1))
        times.append(t2 - t1)
        times_pi.append(time_per_it/(its-1))
    except Exception as ex:
        times.append(0)
        times_pi.append(0)

labels = ['Exact subsolves', 'PyOpenCL-CG single RHS subsolves ', 'PyOpenCL-CG multiple RHS', 'NumPy-CG subsolves', 'C-CG single RHS Subsolves', 'C-CG multiple RHS']
plt.figure(figsize=(25.60, 14.40), dpi=100)

# Plot the data
plt.bar(labels, times_pi)
plt.bar(labels, times, bottom=times_pi)
plt.xlabel('CG implementations')
plt.ylabel('Time taken to converge (s)')
plt.title(f'Performance comparison of CG implementations for {M_s} subdomain width and {W_s*W_s} total number of subdomains and max iteration {CGMaxIT}')
# plt.grid(True, color = "grey", linewidth = "1.4", linestyle = "--")
# plt.show()
plt.savefig(f'./graphs/cg_{M_s}_{W_s}.png')
