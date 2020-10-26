from c2qa.qumoderegister import QumodeRegister
import numpy as np
from scipy.linalg import expm

class CVOperators:
    def __init__(self, qmr: QumodeRegister):        
        I = np.eye(qmr.cutoff)
        
        # Annihilation operator
        self.a = np.zeros((qmr.cutoff, qmr.cutoff))
        for i in range(qmr.cutoff - 1):
            self.a[i, i + 1]= np.sqrt(i + 1)

        # Creation operator
        self.a_dag = self.a.conj().T

        # Number operator
        self.N = np.matmul(self.a_dag, self.a)

        # 2-qumodes operators
        self.a1 = np.kron(self.a, I)
        self.a2 = np.kron(I, self.a)
        self.a1_dag = self.a1.conj().T
        self.a2_dag = self.a2.conj().T

    def D(self, alpha):
        # Displacement operator matrix for nxn
        arg = alpha * self.a_dag-np.conjugate(alpha) * self.a
        return expm(arg)

    def S(self, z):
        #Single mode squeezing
        a2 = np.matmul(self.a, self.a)
        a2_dag = np.matmul(self.a_dag, self.a_dag)
        arg = (np.conjugate(z) * a2) - (z * a2_dag)
        return expm(arg)

    def R(self, phi):
        arg = 1j * phi * np.matmul(self.a_dag, self.a)
        return expm(arg)

    def K(self, kappa):
        j = np.complex(0,1)
        arg = j * kappa * np.matmul(self.N, self.N)
        return expm(arg)

    def S2(self, z):
        #two mode squeeze
        a12 = np.matmul(self.a1, self.a2)
        a12_dag = np.matmul(self.a1_dag, self.a2_dag)
        arg = (np.conjugate(z) * a12) - (z * a12_dag)
        return expm(arg)

    def BS(self, phi):
        a12dag = np.matmul(self.a1, self.a2_dag)
        a1dag2 = np.matmul(self.a1_dag, self.a2)
        arg = (phi * a12dag) - (np.conjugate(phi) * a1dag2)
        return expm(arg)
