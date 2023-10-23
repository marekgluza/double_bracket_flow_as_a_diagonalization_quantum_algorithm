import numpy as np
from simple_exact_diagonalization_routines.local_matrix_class import *
from itertools import combinations

class operators:
    def __init__(self, L):
        self.L = L
        self.X = X_class(self.L)
        self.Y = Y_class(self.L)
        self.Z = Z_class(self.L)
        self.id = np.eye(2**self.L)
        
class ising_model(operators):
    def __init__(self, B, D, max_coupling_range):
        self.B = B
        self.D = D
        self.L = len(B)
        self.max_coupling_range = max_coupling_range
        self.interactions = True
        super().__init__(self.L)
    
    def J_ij(self, i=int, j=int):
        if np.abs(i-j) > self.max_coupling_range or i==j:
            return 0
        else:
            return 1/np.abs(i-j)
    
    
    def hamiltonian_ising(self):
        H = sum([self.B[i] * self.Z.at(i+1) + 2 * self.D[i] * self.X.at(i+1) for i in range (self.L)])
        if self.interactions == False:
            return H 
        comb = list(combinations(range(self.L),2))
        for (i,j) in comb:
            H += self.J_ij(i,j) * self.Z.at(i+1) * self.Z.at(j+1)
        return H