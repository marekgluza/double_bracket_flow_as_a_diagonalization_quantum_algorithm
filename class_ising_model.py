import numpy as np
from simple_exact_diagonalization_routines.local_matrix_class import *
import itertools

class Pauli_algebra:
    def __init__(self, L):
        self.L = L
        self.X = X_class(self.L)
        self.Y = Y_class(self.L)
        self.Z = Z_class(self.L)
        self.id = np.eye(2**self.L)
        self.pauli_basis_dict = self.Pauli_basis()
    
    @staticmethod
    def compute_expectation(H, P):
        return np.trace(P @ H)
    
    @staticmethod
    def select_onsite_Z_components(decomp_dict):
        return {key: value for key, value in decomp_dict.items() if key.count(3) == 1 and len(key) - key.count(0) == 1}
    
    def op_dict(self, i, pos):
        if i == 0:
            return self.id
        else:
            op_dict = {1:self.X, 2:self.Y, 3:self.Z}
            return op_dict[i].at(pos+1)
        
    def Pauli_basis(self):
        basis_index_list = list(itertools.product([0,1,2,3], repeat=self.L))
        basis_op_dict = {}
        for index in basis_index_list:
            ops = [self.op_dict(index[i], i) for i in range (self.L)]
            product = np.linalg.multi_dot(ops)
            basis_op_dict[index] = product
        return basis_op_dict
        
    def Pauli_decomposition(self, H, non_zero=True):
        decomp = dict()
        for pauli_op in self.pauli_basis_dict.items():
            expect = self.compute_expectation(H, pauli_op[1])/2**self.L
            if non_zero and expect == 0:
                continue
            else:
                decomp[pauli_op[0]] = expect
        return decomp
        
class ising_model(Pauli_algebra):
    def __init__(self, B, D, max_coupling_range):
        self.B = B
        self.D = D
        self.L = len(B)
        self.H = None
        self.max_coupling_range = max_coupling_range
        self.interactions = True
        super().__init__(self.L)
    
    def J_ij_1D(self, i=int, j=int):
        if np.abs(i-j) > self.max_coupling_range or i==j:
            return 0
        else:
            return 1/np.abs(i-j)
    
    
    def hamiltonian_ising(self):
        H = sum([self.B[i] * self.Z.at(i+1) + 2 * self.D[i] * self.X.at(i+1) for i in range (self.L)])
        if self.interactions == False:
            self.H = H
            return H 
        comb = list(itertools.combinations(range(self.L),2))
        for (i,j) in comb:
            H += self.J_ij_1D(i,j) * self.Z.at(i+1) * self.Z.at(j+1)
        self.H = H
        return H