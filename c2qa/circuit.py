from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions import UnitaryGate
from c2qa.operators import CVOperators
from c2qa.qumoderegister import QumodeRegister
import numpy
import warnings


class CVCircuit(QuantumCircuit):
    def __init__(self, *regs, name: str = None):
        self.qmr = None
        registers = []

        for reg in regs:
            if isinstance(reg, QumodeRegister):
                if self.qmr is not None:
                    warnings.warn("More than one QumodeRegister provided. Using the last one for cutoff.", UserWarning)
                self.qmr = reg
                registers.append(self.qmr.qreg)
            else:
                registers.append(reg)
        
        if self.qmr is None:
            raise ValueError("At least one QumodeRegister must be provided.")

        super().__init__(*registers, name=name)

        self.ops = CVOperators(self.qmr.cutoff)

    def cv_initialize(self, fock_state, qumodes):
        """ Initialize the qumode to a Fock state. """

        # Qumodes are already represented as arrays of qubits,
        # but if this is an array of arrays, then we are initializing multiple qumodes.
        modes = qumodes
        if not isinstance(qumodes[0], list):
            modes = [qumodes]

        if fock_state > self.qmr.cutoff:
            raise ValueError("The given Fock state is greater than the cutoff.")

        for qumode in modes:
            value = numpy.zeros((self.qmr.cutoff,))
            value[fock_state] = 1

            super().initialize(value, [qumode])

    def cv_conditional(self, name, op_0, op_1):
        """ Make two operators conditional (i.e., controlled by qubit in either the 0 or 1 state) """
        sub_qr = QuantumRegister(1)
        sub_qmr = QumodeRegister(1, self.qmr.num_qubits_per_mode)
        sub_circ = QuantumCircuit(sub_qr, sub_qmr.qreg, name=name)
        sub_circ.append(UnitaryGate(op_0).control(num_ctrl_qubits=1, ctrl_state=0), [sub_qr[0]] + sub_qmr[0])
        sub_circ.append(UnitaryGate(op_1).control(num_ctrl_qubits=1, ctrl_state=1), [sub_qr[0]] + sub_qmr[0])

        return sub_circ.to_instruction()

    def cv_bs(self, phi, qumode_a, qumode_b):
        operator = self.ops.bs(phi)

        self.unitary(obj=operator, qubits=qumode_a + qumode_b, label='BS')

    def cv_d(self, alpha, qumode):
        operator = self.ops.d(alpha)

        self.unitary(obj=operator, qubits=qumode, label='D')

    def cv_cnd_d(self, alpha, beta, ctrl, qumode):
        self.append(self.cv_conditional('Dc', self.ops.d(alpha), self.ops.d(beta)), [ctrl] + qumode)

    def cv_r(self, phi, qumode):
        operator = self.ops.r(phi)

        self.unitary(obj=operator, qubits=qumode, label='R')

    def cv_s(self, z, qumode):
        operator = self.ops.s(z)

        self.unitary(obj=operator, qubits=qumode, label='S')

    def cv_cnd_s(self, z_a, z_b, ctrl, qumode_a):
        self.append(self.cv_conditional('Sc', self.ops.s(z_a), self.ops.s(z_b)), [ctrl] + qumode_a)

    def cv_s2(self, z, qumode_a, qumode_b):
        operator = self.ops.s2(z)

        self.unitary(obj=operator, qubits=qumode_a + qumode_b, label='S2')
