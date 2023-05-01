#!/usr/bin/env python3
import scipy.sparse, scipy.sparse.linalg, math
from numpy import array,zeros,ones,random,dot,sqrt,arange
#from minres_qlp import MinresQLP
import scipy.sparse as sp

import pdb

def helmFE_var(N,omega,C,rho,Nhoriz,Nvert):
    #%  helmFE.m
    # 
    # Author:  Ivan Graham  January 4th 2016
    # 
    #   Discretises the BVP,
    #
    #   -\Delta u - (1+i\rho)(k^2) u = f, on \, Omega=(0,1)^2
    #            \partial_{n} u -ik u = 0 , on \partial\Omega
    #
    # where k = omega/c, and omega is the (constant) frequency while c is the (variable) wave spped
    # using piecewise linear finite elements
    # the matrix C contains the values of the variable wave speed c on a
    # uniform grid of squares, each square contains two triangular elements
    # When c is constant and rho = epsilon/k^2 the output of the code coincides
    # with the output of the original code helmFE_ivan.m
    # 

    #%  Input:
    #   
    #   N= number of grid points in each direction
    #  omega = a positive real (the frequency)
    #  C = the wave speed (a variable function given as an M-1 \times M-1
    # matrix of positive numbers - representing the speed on each square of
    # size h x h  (labelled lexicographicaly). Each square contains two elements. 
    # note that the symbol c is used later as a variable in the code, to the
    # wave speed is stored as C here

    #
    #%  output:
    #
    #   S = finite element matrix
    #   
    #%  Summary:
    #
    # Discretises Helmholtz eqn in a square (using Finite Element method) with
    # Robin conditions on all sides.

    h=1./(N-1.)  
    h2=h**2
 
    # The matrix is assembled at the end using the matlab sparse function
    
    # The vector a contains the row indices of the non-zero elements 
    # The vector b contains the column indices
    # The vector c contains the entries themselves. 
    
    #nn=N**2
    nn=Nhoriz*Nvert
    nnz=N**2 + 4*(N-1)*N + 2*(N-1)**2

    a=zeros(nnz,dtype=int)   # x's
    b=zeros(nnz,dtype=int)   # y's
    c=zeros(nnz,dtype=complex)   # nz's
    pos=0
    # form S
    for j in range(Nhoriz): # index of x coordinate of node
        for m in range(Nvert): # index of y coordinate  of node
            # Note that the node with indices (j,m) lies in the bottom left
            # hand corner of the (j,m)th square with vave speed c(j,m)        
            ##  Matrix S = K - M - i*B
            # where K= Stiffness matrix, 
            # M = Domain Mass matrix with weight (omega/c)^2
            # and B = Boundary mass matrix with weight (omega/c) 
            
            ## Diagonal entries of matrix 
            
            ## diagonal entries at corners
            if j==0 and m==0:    # bottom LH corner 
                k = omega/C[0,0]
                a[pos]= 0
                b[pos]= 0
                c[pos]= 1. - (1.+rho*1j)*k**2*h2/6. - 1j*k*2*h/3.    
                pos=pos+1
            if j==Nhoriz-1 and m==0:   # bottom RH corner 
                k = omega/C[0,Nhoriz-2]
                a[pos]= Nhoriz-1
                b[pos]= Nhoriz-1
                c[pos]= 1. - (1.+rho*1j)*k**2*h2/12. - 1j*k*2.*h/3. 
                pos=pos+1
            if j==0 and m==Nvert-1:  # top LH corner 
                k = omega/C[Nvert-2,0]
                a[pos]=(Nvert-1)*Nhoriz
                b[pos]=(Nvert-1)*Nhoriz
                c[pos]= 1. - (1.+rho*1j)*k**2*h2/12. - 1j*k*2.*h/3. 
                pos=pos+1
            if j==Nhoriz-1 and m==Nvert-1:  # top RH corner 
                k = omega/C[Nvert-2,Nhoriz-2]
                a[pos]= Nvert*Nhoriz - 1
                b[pos]= Nvert*Nhoriz - 1
                c[pos]= 1. - (1.+rho*1j)*k**2*(h2/6.) - 1j*k*2.*h/3.  
                pos=pos+1
            
            ## Diagonal entries on edges and interior 
            
            if m==0 and j>0 and j<Nhoriz-1:   # bottom edge 
                kl = omega/C[0,j-1]
                kr = omega/C[0,j]      # contributions from squares to left and right
                a[pos]=m*Nhoriz+j
                b[pos]=m*Nhoriz+j
                c[pos]= 2. - (1.+rho*1j)*(kl**2 + 2.*kr**2)*h2/12. - 1j*(kl+kr)*h/3.
                pos=pos+1
            if m==Nvert-1 and j>0 and j<Nhoriz-1:    # top edge 
                kl = omega/C[Nvert-2,j-1]
                kr = omega/C[Nvert-2,j]    # contributions from squares to left and right 
                a[pos]=m*Nhoriz+j
                b[pos]=m*Nhoriz+j
                c[pos]= 2.-(1.+rho*1j)*(2.*kl**2 + kr**2)*h2/12. - 1j*(kl+kr)*h/3.
                pos=pos+1
            if j==0 and m>0 and m<Nvert-1:   # Left hand edge 
                kt = omega/C[m,0]
                kb = omega/C[m-1,0]  # contributions from squares above and below
                a[pos]=m*Nhoriz+j
                b[pos]=m*Nhoriz+j
                c[pos]= 2. - (1.+rho*1j)*(2.*kt**2 + kb**2)*h2/12. - 1j*(kt+kb)*h/3.
                pos=pos+1
            if j==Nhoriz-1 and m>0 and m<Nvert-1:  # Right hand edge 
                kt = omega/C[m,Nhoriz-2]
                kb = omega/C[m-1,Nhoriz-2]   # contributions from squares above and below
                a[pos]=m*Nhoriz+j
                b[pos]=m*Nhoriz+j
                c[pos]= 2. - (1.+rho*1j)*(kt**2 + 2.*kb**2)*h2/12. - 1j*(kt+kb)*h/3.
                pos=pos+1
            if m>0 and m<Nvert-1 and j>0 and j<Nhoriz-1:   # interior 
                knw = omega/C[m,j-1]
                ksw = omega/C[m-1,j-1]
                kne = omega/C[m,j]
                kse = omega/C[m-1,j]    # contributions from 4 surrounding squares
                a[pos]=m*Nhoriz+j
                b[pos]=m*Nhoriz+j
                c[pos]= 4. - (1.+rho*1j)*(knw**2 + 2.*ksw**2 + 2.*kne**2 + kse**2)*h2/12.
                pos=pos+1
            ## Off diagonal entries 
            #
            ## off diagonal corner nodes 
            if j==0 and m==0:   # bottom LH corner 
                k = omega/C[0,0]   # value of k on square 
                rowindex = 0
                a[pos]= rowindex
                b[pos]= rowindex+1  # node to right 
                c[pos]= -0.5 - (1.+rho*1j)*k**2*h2/24. - 1j*k*h/6.
                pos=pos+1
                a[pos]=rowindex
                b[pos]= rowindex + Nhoriz  # node  above 
                c[pos]= -0.5 - (1.+rho*1j)*k**2*h2/24. - 1j*k*h/6.
                pos=pos+1
                a[pos]=rowindex
                b[pos]=rowindex + Nhoriz+1  # node above to right   
                c[pos]=  - (1.+rho*1j)*k**2*h2/12.
                pos = pos+1
            if j==Nhoriz-1 and m==Nvert-1:   # Top RH corner 
                k = omega/C[Nvert-2,Nhoriz-2]   # value of k on square 
                rowindex = Nhoriz*Nvert-1
                a[pos]= rowindex
                b[pos]= rowindex-1  # node to left 
                c[pos]= -0.5 - (1.+rho*1j)*(k**2)*h2/24. - 1j*k*h/6.
                pos=pos+1
                a[pos]=rowindex
                b[pos]= rowindex - Nhoriz  # node  below
                c[pos]= -0.5 - (1.+rho*1j)*k**2*h2/24. - 1j*k*h/6.
                pos=pos+1
                a[pos]=rowindex
                b[pos]=rowindex - Nhoriz-1  # node below to left 
                c[pos]=  - (1.+rho*1j)*k**2*h2/12.
                pos = pos+1;          
            if j==Nhoriz-1 and m==0:   # Bottom RH corner 
                k = omega/C[0,Nhoriz-2]   # value of k on square 
                rowindex = Nhoriz-1
                a[pos]= rowindex
                b[pos]= rowindex-1  # node to left 
                c[pos]= -0.5 - (1.+rho*1j)*k**2*h2/24. - 1j*k*h/6.
                pos=pos+1
                a[pos]=rowindex
                b[pos]= rowindex + Nhoriz # node  above 
                c[pos]= -0.5 - (1.+rho*1j)*k**2*h2/24. - 1j*k*h/6.
                pos=pos+1
            if j==0 and m==Nvert-1:   # Top LH corner
                k = omega/C[Nvert-2,0]   # value of k on square 
                #rowindex = (M-1)*M + 1;
                rowindex = (Nvert-1)*Nhoriz
                a[pos]= rowindex
                b[pos]= rowindex+1  # node to right 
                c[pos]= -0.5 - (1.+rho*1j)*k**2*h2/24. - 1j*k*h/6.
                pos=pos+1
                a[pos]=rowindex
                b[pos]= rowindex - Nhoriz  # node  below 
                c[pos]= -0.5 -(1.+rho*1j)*k**2*h2/24. - 1j*k*h/6.
                pos=pos+1
            
            ## off diagonal - bottom boundary  nodes, not corners : node has 4 neighbours 
            
            if j>0 and j<Nhoriz-1 and m==0:
                kl = omega/C[0,j-1]   # value of k on left-hand square 
                kr = omega/C[0,j]     # value of k on right-hand square 
                rowindex = m*Nhoriz+j
                a[pos]= rowindex
                b[pos]= rowindex-1  # node to left 
                c[pos]= -0.5 - (1.+rho*1j)*kl**2*h2/24. - 1j*kl*h/6.
                pos=pos+1
                a[pos]=rowindex
                b[pos]= rowindex + 1  # node  to right
                c[pos]= -0.5 - (1.+rho*1j)*kr**2*h2/24. - 1j*kr*h/6.
                pos=pos+1
                a[pos]=rowindex
                b[pos]=rowindex+Nhoriz   # node directly above  
                c[pos]= -1. -  (1.+rho*1j)*(kl**2 + kr**2)*h2/24.
                pos = pos+1
                a[pos]=rowindex
                #b[pos]=rowindex+M+1;  % node above, diagonal to right   
                b[pos]=rowindex+Nhoriz+1  # node above, diagonal to right   
                c[pos]= - (1.+rho*1j)*kr**2*h2/12.
                pos = pos+1
             
            ## off diagonal - top boundary  nodes, not corners : node has 4 neighbours 
            
            if j>0 and j<Nhoriz-1 and m==Nvert-1:
                kl = omega/C[Nvert-2,j-1]   # value of k on left-hand square 
                kr = omega/C[Nvert-2,j]     # value of k on right-hand square 
                rowindex = m*Nhoriz+j
                a[pos]=rowindex
                b[pos]=rowindex-1  # node to left 
                c[pos]= -0.5 - (1.+rho*1j)*kl**2*h2/24. - 1j*kl*h/6.
                pos=pos+1
                a[pos]=rowindex
                b[pos]=rowindex+1  # node  to right
                c[pos]= -0.5 - (1.+rho*1j)*kr**2*h2/24. - 1j*kr*h/6.
                pos=pos+1
                a[pos]=rowindex
                b[pos]=rowindex-Nhoriz  # node directly below
                c[pos]= -1. - (1.+rho*1j)*(kl**2 + kr**2)*h2/24.
                pos = pos+1
                a[pos]=rowindex
                b[pos]=rowindex-Nhoriz-1  # node below, diagonal to left   
                c[pos]= - (1.+rho*1j)*kl**2*h2/12.
                pos = pos+1
             
            ## off diagonal - right boundary  nodes, not corners : node has 4 neighbours 
            
            if m>0 and m<Nvert-1 and j==Nhoriz-1:
                kb = omega/C[m-1,Nhoriz-2]   # value of k on bottom square 
                kt = omega/C[m,Nhoriz-2]     # value of k on top square 
                #rowindex = m*M;
                rowindex = m*Nhoriz + Nhoriz-1
                a[pos]=rowindex
                b[pos]= rowindex-Nhoriz  # node below 
                c[pos]= -0.5 - (1.+rho*1j)*kb**2*h2/24. - 1j*kb*h/6.
                pos=pos+1
                a[pos]= rowindex
                b[pos]= rowindex + Nhoriz  # node  above 
                c[pos]= -0.5 - (1.+rho*1j)*kt**2*h2/24. - 1j*kt*h/6.
                pos=pos+1
                a[pos]= rowindex
                b[pos]= rowindex-1  # node directly to the left 
                c[pos]= -1. - (1.+rho*1j)*(kb**2 + kt**2)*h2/24.
                pos = pos+1
                a[pos]= rowindex
                b[pos]= rowindex-Nhoriz-1  # node below, diagonal to left   
                c[pos]= - (1.+rho*1j)*kb**2*h2/12.
                pos = pos+1
             
            ## off diagonal - left boundary  nodes, not corners : node has 4 neighbours 
            
            if m>0 and m<Nvert-1 and j==0:
                kb = omega/C[m-1,0]   # value of k on bottom square 
                kt = omega/C[m,0]     # value of k on top square 
                #rowindex = (m-1)*M+1;
                rowindex = m*Nhoriz
                a[pos]= rowindex
                b[pos]= rowindex-Nhoriz  # node below 
                c[pos]= -0.5 - (1.+rho*1j)*kb**2*h2/24. - 1j*kb*h/6.
                pos=pos+1
                a[pos]= rowindex
                b[pos]= rowindex + Nhoriz   # node  above 
                c[pos]= -0.5 - (1.+rho*1j)*kt**2*h2/24. - 1j*kt*h/6.
                pos=pos+1
                a[pos]= rowindex
                b[pos]= rowindex + 1  # node directly to the right 
                c[pos]= -1. - (1.+rho*1j)*(kb**2 + kt**2)*h2/24.
                pos = pos+1
                a[pos]=  rowindex
                b[pos]=  rowindex + Nhoriz+1 # node above, diagonal to right 
                c[pos]= - (1.+rho*1j)*kt**2*h2/12.
                pos = pos+1
           
            ## Off diagonal - interior nodes: node has 6 neighbours
            
            if j>0 and j<Nhoriz-1 and m>0 and m<Nvert-1:     
                knw = omega/C[m,j-1]
                ksw = omega/C[m-1,j-1]
                kne = omega/C[m,j]
                kse = omega/C[m-1,j]    # contributions from 4 surrounding squares
                rowindex = m*Nhoriz+j
                a[pos]= rowindex 
                b[pos]= rowindex + 1   # same horizontal line, one node to the right 
                c[pos]= -1. - (1.+rho*1j)*(kne**2 + kse**2)*h2/24.
                pos=pos+1
                a[pos]= rowindex
                b[pos]= rowindex - 1    # same horizontal line, one node to the left 
                c[pos]= -1. - (1.+rho*1j)*(knw**2 + ksw**2)*h2/24.
                pos=pos+1
                a[pos]= rowindex   
                b[pos]= rowindex +Nhoriz        # same vertical line,  one node above 
                c[pos]= -1. - (1.+rho*1j)*(knw**2 + kne**2)*h2/24.
                pos=pos+1
                a[pos]=rowindex
                b[pos]= rowindex - Nhoriz       # same vertical line,  one node below 
                c[pos]= -1. - (1.+rho*1j)*(ksw**2 + kse**2)*h2/24.  
                pos=pos+1
                a[pos]=rowindex
                b[pos]= rowindex-Nhoriz-1                    # diagonal below to the left 
                c[pos]= - (1.+rho*1j)*ksw**2*h2/12. 
                pos=pos+1
                a[pos]= rowindex
                b[pos]= rowindex +Nhoriz+1                   # diagonal above to the right 
                c[pos]= -(1.+rho*1j)*kne**2*h2/12. 
                pos=pos+1
    a=a[0:pos]
    b=b[0:pos]
    c=c[0:pos]
    #print 'calling scipy'
    ret = scipy.sparse.csr_matrix((c,(a,b)),shape=(nn,nn))
    #print 'finished calling scipy'
    return ret

def rhs(N,k): # special RHS from Ivan
    from numpy import arange,exp
    aa=[1./math.sqrt(2.),1./math.sqrt(2.)]
    aaa=array(aa).transpose()
    h = 1./(N-1.) 
    b=zeros((N,N),dtype=complex)
    x=arange(0.0,1.00001,1.0/(N-1))
    #y = (x(2:M) + x(1:M-1))/2;   % mid points 
    y=(x[1:]+x[:-1])/2.0
    #% multipliers i*k*(a.n-1) computed on each side of boundary
    multbot =  1.j*k*(-aa[1] - 1.) 
    multtop = 1.j*k*(aa[1] - 1.)
    multleft = 1.j*k*(-aa[0] - 1.)
    multright = 1.j*k*(aa[0] - 1.)
    for j in range(1,N-1): # global rownumbers
         points = array([[y[j-1],0.],[x[j],0.],[y[j],0.]])  # interior of bottom boundary
         b[0,j] = (h/3.)*multbot*sum(exp(1.*1.j*k*(dot(points,aaa))))
         points = array([[y[j-1],1.],[x[j],1.],[y[j],1.]]) # interior of top boundary
         b[-1,j] = (h/3.)*multtop*sum(exp(1.*1.j*k*(dot(points,aaa))))
         points = array([[0.,y[j-1]],[0.,x[j]],[0,y[j]]])  # interior of left boundary
         b[j,0] = (h/3.)*multleft*sum(exp(1.*1.j*k*(dot(points,aaa))))
         points = array([[y[j-1],1.],[x[j],1.],[y[j],1.]]) # interior of right boundary
         b[j,-1] = (h/3.)*multright*sum(exp(1.*1.j*k*(dot(points,aaa))))
    points = array([[0.,y[0]],[0.,0.],[y[0],0.]]) # bottom left corner
    b[0,0] = (h/6.)*multleft*(2.*exp(1.*1.j*k*(dot(points[0,:],aaa))) + exp(1.*1.j*k*dot(points[1,:],aaa)))
    b[0,0] = b[0,0] + (h/6.)*multbot*(2.*exp(1.*1.j*k*dot(points[2,:],aaa)) + exp(1.*1.j*k*dot(points[1,:],aaa)));
    points = array([[y[N-2],0.],[1.,0.],[1.,y[0]]])        # bottom right corner
    b[0,-1] = (h/6.)*multbot*(2.*exp(1.*1.j*k*dot(points[0,:],aaa)) + exp(1.*1.j*k*dot(points[1,:],aaa)))
    b[0,-1] = b[0,-1] + (h/6.)*multright*(2.*exp(1.*1.j*k*dot(points[2,:],aaa)) + exp(1.*1.j*k*dot(points[1,:],aaa)))
    points = array([[0.,y[N-2]],[0.,1.],[y[0],1.]]) # top left corner
    b[-1,0] = (h/6.)*multleft*(2.*exp(1.*1.j*k*dot(points[0,:],aaa)) + exp(1.*1.j*k*dot(points[1,:],aaa)))
    b[-1,0] = b[-1,0] + (h/6.)*multtop*(2.*exp(1.*1.j*k*dot(points[2,:],aaa)) + exp(1.*1.j*k*dot(points[1,:],aaa)))
    points = array([[y[N-2],1.],[1.,1.],[1.,y[N-2]]]) # top right corner
    b[-1,-1] = (h/6.)*multtop*(2.*exp(1.*1.j*k*dot(points[0,:],aaa)) + exp(1.*1.j*k*dot(points[1,:],aaa)))
    b[-1,-1] = b[-1,-1] + (h/6.)*multright*(2.*exp(1.*1.j*k*dot(points[2,:],aaa)) + exp(1.*1.j*k*dot(points[1,:],aaa)))
    return b

def rhsL(N,k): # special RHS from Ivan
    from numpy import arange,exp
    aa=[1./math.sqrt(2.),1./math.sqrt(2.)]
    aaa=array(aa).transpose()
    h = 1./(N-1.) 
    b=zeros((N,N),dtype=complex)
    b[1:N-1,0]=k*k
    return b

def rhsA(N,k): # special RHS from Ivan
    from numpy import arange,exp
    aa=[1./math.sqrt(2.),1./math.sqrt(2.)]
    aaa=array(aa).transpose()
    h = 1./(N-1.) 
    b=zeros((N,N),dtype=complex)
    b[:,0]=k*k
    b[:,-1]=k*k
    b[0,:]=k*k
    b[-1,:]=k*k
    return b

def GaussSeidel(Acoo,b,maxit=1000):
    rowinds=Acoo.row.copy()
    colinds=Acoo.col.copy()
    vals=Acoo.data.copy()
    diag=Acoo.diagonal().copy()
    nnz=Acoo.nnz
    # Preprocessing the values
    #b=b/diag
    # we assume the matrix elements are sorted row-wise having all
    # elements in row i consecutively
    #print 'Acoo=',Acoo
    offdiag=0
    i_prev=-1
    for h in range(nnz):
        i=rowinds[h]
        if i<i_prev:
            print('WARNING, rowinds not in sorted order!')
        j=colinds[h]
        # take out diagonal elements; scale everything with the diagonal
        if j!=i: # non-diagonal entry
            rowinds[offdiag]=i
            colinds[offdiag]=j
            #vals[offdiag]=vals[h]/diag[i]
            vals[offdiag]=vals[h]
            offdiag=offdiag+1
        i_prev = i
    # Perform Gauss-seidel iterations here...:
    #x = zeros(len(b),dtype=complex)
    #x = zeros(len(b))
    x=b.copy()
    #pdb.set_trace()
    for t in range(maxit):
        i_prev=-1
        for h in range(offdiag):
            i=rowinds[h]
            j=colinds[h]
            #print h,i,colinds[h],vals[h],diag[i]
            if i==i_prev:
                x[i]=x[i]-vals[h]*x[j]
            elif i>i_prev: # new row in matrix (i changed...)
                if i_prev>-1: # still to finish with the previous row:
                    x[i_prev]=x[i_prev]/diag[i_prev]
                x[i]=b[i]-vals[h]*x[j]
            else:
                print('something wrong...',i,i_prev)
            i_prev=i
        x[i]=x[i]/diag[i]
        print(t,':',max(abs(Acoo.dot(x)-b)))
    return x

def SymmGaussSeidel(Acoo,b):
    rowinds=Acoo.row.copy()
    colinds=Acoo.col.copy()
    vals=Acoo.data.copy()
    diag=Acoo.diagonal().copy()
    nnz=Acoo.nnz
    # Preprocessing the values
    #b=b/diag
    # we assume the matrix elements are sorted row-wise having all
    # elements in row i consecutively
    print('Acoo=',Acoo)
    offdiag=0
    i_prev=-1
    N=Acoo.shape[0]
    for h in range(nnz):
        i=rowinds[h]
        if i<i_prev:
            print('WARNING, rowinds not in sorted order!')
        j=colinds[h]
        # take out diagonal elements; scale everything with the diagonal
        if j!=i: # non-diagonal entry
            rowinds[offdiag]=i
            colinds[offdiag]=j
            #vals[offdiag]=vals[h]/diag[i]
            vals[offdiag]=vals[h]
            offdiag=offdiag+1
        i_prev = i
    # Perform Gauss-seidel iterations here...:
    #x = zeros(len(b),dtype=complex)
    #x = zeros(len(b))
    x=b.copy()
    #pdb.set_trace()
    for t in range(3):
        i_prev=-1
        for h in range(offdiag):
            i=rowinds[h]
            j=colinds[h]
            #print h,i,colinds[h],vals[h],diag[i]
            if i==i_prev:
                x[i]=x[i]-vals[h]*x[j]
            elif i>i_prev: # new row in matrix (i changed...)
                if i_prev>-1: # still to finish with the previous row:
                    x[i_prev]=x[i_prev]/diag[i_prev]
                x[i]=b[i]-vals[h]*x[j]
            else:
                print('something wrong...',i,i_prev)
            i_prev=i
        x[i]=x[i]/diag[i]
        i_prev=N
        for h in range(offdiag-1,-1,-1):
            i=rowinds[h]
            j=colinds[h]
            #print h,i,j,vals[h],diag[i]
            if i==i_prev:
                x[i]=x[i]-vals[h]*x[j]
            elif i<i_prev: # new row in matrix (i changed...)
                if i_prev<N: # still to finish with the previous row:
                    x[i_prev]=x[i_prev]/diag[i_prev]
                x[i]=b[i]-vals[h]*x[j]
            else:
                print('something wrong...',i,i_prev,N)
            i_prev=i
        x[i]=x[i]/diag[i]
        print(t,':',max(abs(Acoo.dot(x)-b)))
    return x

def CG(A, b, x=None, tol=1e-5, maxit=1000, verbose=False):
    if x is None:
        x = zeros(b.size, dtype=complex)
    ax = A.dot(x)

    r = b-ax

    d = r

    deltaNew = dot(r, r)
    # print(f'Delta new: {deltaNew}')

    for i in range(maxit):
        q = A.dot(d)

        dq = dot(d, q)

        alpha = deltaNew/dq
        # print(f'Alpha : {alpha}')

        x = x + alpha * d
        # print(f'X: {x}')

        r = r - alpha * q
        # print(f'R: {r}')

        deltaOld = deltaNew

        deltaNew = dot(r, r)
        # print(f'Delta new {i} : {deltaNew}')
        # print(f'Delta old{i} : {deltaOld}')

        beta = deltaNew/deltaOld
        # print(f'Beta: {beta}')

        d = r + beta * d
        # print(f'D: {d}')
    return x

def PCG(A,b,M=None,x=None,tol=1e-6,maxit=1000,verbose=False):
    """
    A - square matrix, should be positive semi-definite with A.dot() operation defined
    b - numpy vector with conformable size to A
    x - initial guess on input (optional)
    tol - residual tolerance in L2-norm 
    """
    if x is None:
        x=zeros(b.size,dtype=complex)
    ax=A.dot(x)
    r = b-A.dot(x)
    for i in range(maxit):
        if M is None:
            z = r 
        elif type(M) is scipy.sparse.csr.csr_matrix:
            if M.nnz > M.shape[0]:
                z = scipy.sparse.linalg.spsolve(M,r)
            else:
                z = M.dot(r)
        elif type(M) is float:
           z = CG(A,r,tol=M)
        else:
            z = M(r)
        rho = dot(r, z)
        #pdb.set_trace()
        if i==0:
            p = z
        else:
            beta = rho/rho_2
            p = z + beta * p
        q = A.dot(p)
        alpha = rho / dot(p, q)
        x = x + alpha * p
        r = r - alpha * q
        res2norm = sqrt(abs(dot(r, r)))
        if verbose:
            print(i,res2norm) # ,dot(r,r),abs(dot(r,r))
        if res2norm < tol:
            break
        rho_2 = rho
    return x,i

def gnuplot3d(f,label): # for plotting 1D functions
    import Gnuplot, Gnuplot.funcutils
    from numpy import arange
    g = Gnuplot.Gnuplot(debug=0)
    g.clear()
    N=int(sqrt(f.size))
    ff=f.reshape((N,N))
    x = arange(len(ff[:,0]))
    y = arange(len(ff[0,:]))
    g('set parametric')
    #g('set dgrid3d')
    g('set style data lines')
    #g('set style data pm3d')
    #g('set hidden')
    g('set contour base')
    g.title(label)
    g.xlabel('x')
    g.ylabel('y')
    g.splot(Gnuplot.GridData(ff,x,y, binary=0))
    input("click enter <--' ")

def matplotlib_plot3d(f,label):
    import matplotlib.pyplot as plt
    from numpy import arange,meshgrid,array
    N=int(sqrt(f.size))
    #print 'f=',f
    #print 'N=',N
    #print 'f=',f.reshape((N,N))
    x=arange(0.0,1.00001,1.0/(N-1))
    y=arange(0.0,1.00001,1.0/(N-1))
    plt.figure()
    xx,yy = meshgrid(x,y)
    plt.pcolor(xx,yy,f.reshape((N,N)))
    plt.colorbar()
    plt.title(label)
    plt.show()

def PrecA(A,x):
    return scipy.sparse.linalg.spsolve(A,x)

if __name__ == "__main__":
    #N=64
    #N=96
    N=128
    #N=256
    #N=512
    n=N*N
    print('# unknowns:',n)
    tol=1e-3
    Marmousi = ones((N-1,N-1)) 
    #Marmousi = random.random((N-1,N-1)) 
    # CG breaks down in case of omega=0.0
    omega=12.0 # parameter - can be changed!
    rho=0.15 # - can be changed
    H=helmFE_var(N=N,omega=omega,C=Marmousi,rho=rho,Nhoriz=N,Nvert=N)
    #print 'H=',H
    #Hcoo=scipy.sparse.coo_matrix(H)
    #b=random.random(n)+random.random(n)*1j
    #b = sum(abs(H.toarray()))
    #b=rhsL(N,omega)
    b=rhsA(N,omega)
    #print 'b=',b
    #gnuplot3d(abs(b),'abs(rhs)')
    b=b.flatten()
    x0=zeros(n,dtype=complex)
    #x0=random.random(n)+random.random(n)*1j
    print('Conjugate Gradient iterations upto tol:',tol)
    x = CG(H,b,x=x0,tol=tol,maxit=2*n,verbose=True)
    bb=H.dot(x)
    res=bb-b
    print('CG unpreconditioned resnorm:',sqrt(abs(dot(res,res))))
    print('max residual:',max(abs(res)))
    #Hcoo=scipy.sparse.coo_matrix(H)
    #rowinds=Hcoo.row.copy()
    #colinds=Hcoo.col.copy()
    #vals=Hcoo.data.copy()
    ## keep the tridiagonal matrix:
    #k=0
    #for i in range(rowinds.size):
    #    if abs(rowinds[i]-colinds[i])<10:
    #        rowinds[k]=rowinds[i]
    #        colinds[k]=colinds[i]
    #        vals[k]=vals[i]
    #        k=k+1
    #Htrid=scipy.sparse.csr_matrix((vals[:k],(rowinds[:k],colinds[:k])),shape=(n,n))
    #x,it = PCG(H,b,M=Htrid,x=x0,tol=tol,maxit=2*n,verbose=False)
    #bb=H.dot(x)
    #res=bb-b
    #print('TriDiagonal-precoditioned:',it,'iterations, resnorm:',sqrt(abs(dot(res,res))))
    #print('max residual:',max(abs(res)))
    #gnuplot3d(abs(x),'abs(solution)')
    matplotlib_plot3d(abs(x),'abs(solution)')
    #matplotlib_plot3d(x.real,'solution.real')
    #matplotlib_plot3d(x.imag,'solution.imag')
