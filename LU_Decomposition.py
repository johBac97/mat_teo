# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 19:44:26 2020

@author: jback
"""
import numpy as np
import scipy.linalg as lin


def _P(row1 , row2 , n):
    """
    Returns the permutation matrix that switches row1 and 
    row2 with each other. The matrix is n x n

    Parameters
    ----------
    row1 : int
        The index of the first row (zero for first row).
    row2 : int
        The index of the second row (zero for first row).
    n : int
        The size of the returned quadratic P matrix.

    Returns
    -------
    P : numpy.ndarray n x n
        The permuation matrix.

    """
    P = np.zeros((n , n))
    #P = _I(n,n)
    P = P + _E(row1,row2,n,n) + _E(row2,row1,n,n)
    
    
    for i in range(n):
        if i == row1 or i == row2:
            continue
        else:
            P[i,i] = 1
    
    return P
    
def _E(row , col , n , m):
    """
    

    Parameters
    ----------
    row : int
        Row.
    col : int
        Coloumn.
    n : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.

    Returns
    -------
    c : TYPE
        DESCRIPTION.

    """
    if m is None:
        m = n
    c = np.zeros((n , m))
    c[row,  col] = 1
    return c

def _I(N, M = None):
    if M is None:
        M = N
    I = np.zeros((N,M))
    for i in range(N):
        I[i,i] = 1
    return I
    
def _L(row , col, factor, N,M = None):
    return _I(N,M) + factor * _E(row,col,N,M)

def LU_Decomposition(A):
    
    U = A.copy()
    L = []
    P = []
    m = A.shape[1]
    n = A.shape[0]
    
    i = 0 # Coloumn index
    r = 0 # Row index
    
    # First gaussian elimination to obtain P
    while i < m and r < n:
        if U[r , i] == 0: 
            # Check if any element in same coloumn is not 0
            # If so switch with _P. Otherwise,
            # continue to next coloumn
            for rr in range(r + 1, n):
                if not U[rr , i] == 0:
                    P.append(_P(rr , r , n))
                    U = P[-1] @ U   
                    break
            else:
                i = i + 1
                continue
        else:
            # Pivot element!
            # Clear rest of coloumn with _L
            
            for ii in range(r + 1 , n):
                if U[ii, i] == 0:
                    continue
                factor = - U[ii,i] / U[r , i]
                L.append(_L(ii , r, factor, n,n))
                U = L[-1] @ U
                
            # Done with coloumn move to next
            i = i + 1
            r = r + 1
    
    Ptot = _I(n,n)
    for p in P:
        Ptot = p @ Ptot
    
    
    
    # If no permutaions necessary, then return L
    if len(Ptot) == 1:
        Ltot = _I(n)
    
    
        for l in L:
            Ltot = l @ Ltot
        
        Ltot = lin.inv(Ltot)
        L = Ltot
        return [L , P , U]
        pass
    P = Ptot
    
    # Second gaussian eleminiation to obtain L
    i = 0 # Coloumn index
    r = 0 # Row index
    U = P @ A.copy()
    L = []
    while i < m and r < n:
        if U[r , i] == 0:
            # Already permutated so only advance one coloumn
            i = i + 1
        else:
            # Pivot element!
            # Clear rest of coloumn with _L
            for ii in range(r + 1 , n):
                if U[ii, i] == 0:
                    continue
                factor = - U[ii,i] / U[r , i]
                L.append(_L(ii , r, factor, n,n))
                U = L[-1] @ U 
                
            # Done with coloumn move to next
            i = i + 1
            r = r + 1
    
    Ltot = _I(n)
    
    
    for l in L:
        Ltot = l @ Ltot
        
    Ltot = lin.inv(Ltot)
    
    L = Ltot
    return [L , P , U]

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 2 and sys.argv[1] == "example":
        
        A = np.array([[1,2,3],[4,5,6],[7,8,9]])
        B = np.array([[0,0,1,2,3],[0,2,4,6,8],[0,4,8,12,16],[0,1,2,3,5]])
        
        [La , Pa , Ua] = LU_Decomposition(A)
        [Lb , Pb , Ub] = LU_Decomposition(B)
        
        
        print("Two examples of usage of LU_Decomposition!\n")
        
        print("The Matrix A,")
        print(A,",")
        print("is deconstructed using the function into L,\n",La)
        print("P, \n",Pa,"\nand U\n",Ua,".")
        
        print
        
    
    