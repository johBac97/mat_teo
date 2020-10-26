# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 13:15:42 2020

@author: jback
"""



import unittest
import numpy as np
from LU_Decomposition import LU_Decomposition 
import random as r


class test_LU_Decomposition(unittest.TestCase):
    
    
    def setUp(self):
        
        A1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
        A2 = np.array([[5,4,1,2],[8,9,2,4],[1,1,2,4]])
        A3 = np.array([[0,0,1,2,3],[0,2,4,6,8],[0,4,8,12,16],[0,1,2,3,5]])
        # Add more static test matrices
        # with some special characteristics
        
        
        self.A = [A1 , A2 , A3]
        
    
    def test_Static(self):
        
        for i,A in enumerate(self.A):
            [L , P, U] = LU_Decomposition(A)
            
            self.assertTrue(np.allclose(P @ A , L @ U), msg="Failed for matrix number %i " % i)
    
    
    def test_random(self):
        
        n_tests = 10
        
        # test 5 x 5 arrays
        for n in range(n_tests):
            A = np.random.randint(low = - 100 , high=100 , size=(5,5))
            [L,P,U] = LU_Decomposition(A)
            
            self.assertTrue(np.allclose( P @ A , L @ U))
            
            # U should be upper triangular
            for i in range(4):
                self.assertTrue(np.allclose(U[i+1:,i] , np.zeros(4 - i)))
        
        # Test different sizes
        for n in range(n_tests):
            n = r.randint(3,20)
            m = r.randint(3,20)
            
            A = np.random.randint(low=-100 , high=100, size=(n,m))
            #print(A)
            [L,P,U] = LU_Decomposition(A)
    
            for i in range(n-1):
                
                # If not square matrix, make sure i does not surpass dimensions
                if i == m:
                    break
                self.assertTrue(np.allclose(U[i+1:,i] , np.zeros(n-1 - i)))
            
            
        
if __name__ == "__main__":
    unittest.main()       