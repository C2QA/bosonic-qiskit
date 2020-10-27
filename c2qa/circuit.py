from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions import UnitaryGate
from c2qa.operators import CVOperators
from c2qa.qumoderegister import QumodeRegister
import numpy
#from thewalrus.fock_gradients import displacement


class CVCircuit(QuantumCircuit):
    def __init__(self, qmr: QumodeRegister, qr: QuantumRegister, cr: ClassicalRegister):
        super().__init__(qmr.qreg, qr, cr)

        self.qmr = qmr
        self.qr = qr
        self.cr = cr

        self.ops = CVOperators(self.qmr)


    def initialize(self, fock_states):
        """ Initialize qumodes to the given fock_states. """
        for qumode, n in enumerate(fock_states):
            if n >= self.qmr.cutoff:
                raise ValueError("The parameter n should be lower than the cutoff")
            
            vector = numpy.zeros((self.qmr.cutoff,))
            vector[n] = 1

            super().initialize(vector, self.qmr[qumode])

    def cv_bs(self, phi, qumode_a, qumode_b):
        operator = self.ops.BS(phi)
        
        self.unitary(obj = operator, qubits = qumode_a + qumode_b, label = 'BS')

    def cv_d(self, alpha, qumode):       
        operator = self.ops.D(alpha)
        # operator = displacement(r = numpy.abs(alpha), phi = numpy.angle(alpha), cutoff = self.cutoff)

        self.unitary(obj = operator, qubits = qumode, label = 'D')

    def cv_cnd_d(self, alpha, beta, ctrl, qumode_a, qumode_b):
        self.append(UnitaryGate(self.ops.D(alpha)).control(num_ctrl_qubits = 1, label = 'Da', ctrl_state = 0), [ctrl] + qumode_a)
        self.append(UnitaryGate(self.ops.D(beta)).control(num_ctrl_qubits = 1, label = 'Db', ctrl_state = 1), [ctrl] + qumode_b)

    def cv_r(self, phi, qumode):
        operator = self.ops.R(phi)
        
        self.unitary(obj = operator, qubits = qumode, label = 'R')

    def cv_s(self, z, qumode):
        operator = self.ops.S(z)
       
        self.unitary(obj = operator, qubits = qumode, label = 'S')

    def cv_cnd_s(self, z_a, z_b, ctrl, qumode_a, qumode_b):
        self.append(UnitaryGate(self.ops.S(z_a)).control(num_ctrl_qubits = 1, label = 'Sa', ctrl_state = 0), [ctrl] + qumode_a)
        self.append(UnitaryGate(self.ops.S(z_b)).control(num_ctrl_qubits = 1, label = 'Sb', ctrl_state = 1), [ctrl] + qumode_b)

    def cv_s2(self, z, qumode_a, qumode_b):
        operator = self.ops.S2(z)
        
        self.unitary(obj = operator, qubits = qumode_a + qumode_b, label = 'S2')

