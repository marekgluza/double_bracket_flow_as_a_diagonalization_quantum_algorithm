import numpy as np
from simple_exact_diagonalization_routines.local_matrix_class import *
import itertools
import math

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
    """
    H = J_{i,j}Z_iZ_j + BiZi + 2DXi
    Options for J:
    1. 1D
    2. circular
    3. custom
    """
    def __init__(self, B, D, 
                 J_type = '1D',
                 max_coupling_range = 1,
                 J_custom = None):
        self.B = B
        self.D = D
        self.L = len(B)
        self.J_type = J_type
        self.J_custom = J_custom
        self.max_coupling_range = max_coupling_range
        self.interactions = True
        super().__init__(self.L)
        self.H = self.hamiltonian_ising()
    
    def J_1D_ij(self, i=int, j=int):
        """
        J_ij = 1/|i-j|, non-circular
        Example: 
        1-2-3-4, coupling range 2: (1,2)=1, (1,3)=1/2, (1,4)=0
        """
        if np.abs(i-j) > self.max_coupling_range or i==j:
            if self.J_type == 'circular' and (i,j)==(0,self.L-1):
                return 1
            else:
                return 0
        else:
            return 1/np.abs(i-j)
    
    
    def hamiltonian_ising(self):
        H = sum([self.B[i] * self.Z.at(i+1) + 2 * self.D[i] * self.X.at(i+1) for i in range (self.L)])
        if self.interactions == False:
            return H 
        comb = list(itertools.combinations(range(self.L),2))
        if self.J_type == 'custom' and self.J_custom != None:
            for (i,j) in comb:
                if (i+1,j+1) in list(self.J_custom.keys()):
                    H += self.J_custom[(i+1,j+1)] * self.Z.at(i+1) * self.Z.at(j+1)
        else:
            for (i,j) in comb:
                H += self.J_1D_ij(i,j) * self.Z.at(i+1) @ self.Z.at(j+1)
        return H
    
    def X_odd(self):
        X_odd_list = [self.X.at(2*i + 1) for i in range(math.ceil(self.L/2))]
        X_odd = np.linalg.multi_dot(X_odd_list) 
        return X_odd