import warnings

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from c2qa.operators import CVGate, CVOperators, ParameterizedOperator
from c2qa.qumoderegister import QumodeRegister


class CVCircuit(QuantumCircuit):
    def __init__(self, *regs, name: str = None, probe_measure: bool = False):
        """
        Initialize the registers (at least one must be QumodeRegister), set
        the circuit name, and the number of steps to animate (default is to not animate).
        """
        self.qmregs = []
        self._qubit_regs = []  # This needs to be unique from qregs[] in the superclass

        registers = []

        num_qumodes = 0
        num_qubits = 0

        for reg in regs:
            if isinstance(reg, QumodeRegister):
                if len(self.qmregs) > 0:
                    warnings.warn(
                        "More than one QumodeRegister provided. Using the last one for cutoff.",
                        UserWarning,
                    )
                num_qumodes += reg.num_qumodes
                self.qmregs.append(reg)
                registers.append(reg.qreg)
                num_qubits += reg.size
            elif isinstance(reg, QuantumRegister):
                self._qubit_regs.append(reg)
                registers.append(reg)
                num_qubits += reg.size
            else:
                registers.append(reg)

        if len(self.qmregs) == 0:
            raise ValueError("At least one QumodeRegister must be provided.")

        # Support measurement using probe qubits
        self.probe_measure = probe_measure
        if probe_measure:
            self.probe = QuantumRegister(size=num_qubits, name="probe")
            registers.append(self.probe)

        super().__init__(*registers, name=name)

        self.ops = CVOperators(self.cutoff, num_qumodes)

    @property
    def cutoff(self):
        return self.qmregs[-1].cutoff

    @property
    def num_qubits_per_qumode(self):
        return self.qmregs[-1].num_qubits_per_qumode

    def cv_initialize(self, fock_state, qumodes):
        """Initialize the qumode to a Fock state."""

        # Qumodes are already represented as arrays of qubits,
        # but if this is an array of arrays, then we are initializing multiple qumodes.
        modes = qumodes
        if not isinstance(qumodes[0], list):
            modes = [qumodes]

        if fock_state > self.qmregs[-1].cutoff:
            raise ValueError("The given Fock state is greater than the cutoff.")

        for qumode in modes:
            value = np.zeros((self.qmregs[-1].cutoff,))
            value[fock_state] = 1

            super().initialize(value, qumode)

    @staticmethod
    def cv_conditional(name, op_0, op_1, num_qubits_per_qumode, num_qumodes=1):
        """Make two operators conditional (i.e., controlled by qubit in either the 0 or 1 state)"""
        sub_qr = QuantumRegister(1)
        sub_qmr = QumodeRegister(num_qumodes, num_qubits_per_qumode)
        sub_circ = QuantumCircuit(sub_qr, sub_qmr.qreg, name=name)

        # TODO Use size of op_0 and op_1 to calculate the number of qumodes instead of using parameter
        qargs = [sub_qr[0]]
        for i in range(num_qumodes):
            qargs += sub_qmr[i]

        sub_circ.append(CVGate(op_0).control(num_ctrl_qubits=1, ctrl_state=0), qargs)
        sub_circ.append(CVGate(op_1).control(num_ctrl_qubits=1, ctrl_state=1), qargs)

        # Create a single instruction for the conditional gate, flag it for later processing
        inst = sub_circ.to_instruction()
        inst.cv_conditional = True
        inst.num_qubits_per_qumode = num_qubits_per_qumode
        inst.num_qumodes = num_qumodes

        return inst

    def cv_aklt(self, qumode_a, qumode_b, qubit_ancilla):
        operator = ParameterizedOperator(self.ops.aklt)
        self.append(CVGate(data=operator, label="AKLT"), qargs=qumode_a + qumode_b + [qubit_ancilla])

    def cv_snap2(self, qumode_a):
        operator = ParameterizedOperator(self.ops.snap2)
        self.append(CVGate(data=operator, label="SNAP2"), qargs=qumode_a)

    def cv_controlledparity(self, qumode_a, qubit_ancilla):
        operator = ParameterizedOperator(self.ops.controlledparity)
        self.append(CVGate(data=operator, label="controlledparity"), qargs=qumode_a + [qubit_ancilla])

    def cv_qubitDependentCavityRotation(self, qumode_a, qubit_ancilla):
        operator = ParameterizedOperator(self.ops.qubitDependentCavityRotation)
        self.append(CVGate(data=operator, label="qubitDependentCavityRotation"), qargs=qumode_a + [qubit_ancilla])

    def cv_bs2m1q(self, qumode_a, qumode_b, qubit_ancilla):
        operator = ParameterizedOperator(self.ops.bs2m1q)
        self.append(CVGate(data=operator, label="bin_bs2m1q"), qargs=qumode_a + qumode_b + [qubit_ancilla])

    def cv_bs(self, phi, qumode_a, qumode_b):
        operator = ParameterizedOperator(self.ops.bs, phi)
        self.append(CVGate(data=operator, label="BS"), qargs=qumode_a + qumode_b)

    def cv_cpbs(self, phi, qumode_a, qumode_b, qubit_ancilla):
        operator = ParameterizedOperator(self.ops.cpbs, phi)
        self.append(CVGate(data=operator, label="CPBS"), qargs=qumode_a + qumode_b + [qubit_ancilla])

    def cv_cnd_bs(self, phi, chi, ctrl, qumode_a, qumode_b):
        op_0 = ParameterizedOperator(self.ops.bs, phi)
        op_1 = ParameterizedOperator(self.ops.bs, chi)
        self.append(
            CVCircuit.cv_conditional(
                "BSc", op_0, op_1, self.num_qubits_per_qumode, num_qumodes=2
            ),
            [ctrl] + qumode_a + qumode_b,
        )

    def cv_d(self, alpha, qumode):
        operator = ParameterizedOperator(self.ops.d, alpha)
        self.append(CVGate(data=operator, label="D"), qargs=qumode)

    def cv_dBHC(self, alpha, qumode):
        operator = ParameterizedOperator(self.ops.dBCH, alpha)
        self.append(CVGate(data=operator, label="D-BCH"), qargs=qumode)

    def cv_cnd_d(self, alpha, beta, ctrl, qumode):
        op_0 = ParameterizedOperator(self.ops.d, alpha)
        op_1 = ParameterizedOperator(self.ops.d, beta)
        self.append(
            CVCircuit.cv_conditional("Dc", op_0, op_1, self.num_qubits_per_qumode),
            [ctrl] + qumode,
        )

    def cv_r(self, phi, qumode):
        operator = ParameterizedOperator(self.ops.r, phi)
        self.append(CVGate(data=operator, label="R"), qargs=qumode)

    def cv_s(self, z, qumode):
        operator = ParameterizedOperator(self.ops.s, z)
        self.append(CVGate(data=operator, label="S"), qargs=qumode)

    def cv_cnd_s(self, z_a, z_b, ctrl, qumode_a):
        op_0 = ParameterizedOperator(self.ops.s, z_a)
        op_1 = ParameterizedOperator(self.ops.s, z_b)
        self.append(
            CVCircuit.cv_conditional("Sc", op_0, op_1, self.num_qubits_per_qumode),
            [ctrl] + qumode_a,
        )

    def cv_s2(self, z, qumode_a, qumode_b):
        operator = ParameterizedOperator(self.ops.s2, z)
        self.append(CVGate(data=operator, label="S2"), qargs=qumode_a + qumode_b)

    def measure_z(self, qubit, cbit):
        if not self.probe_measure:
            warnings.warn(
                "Probe qubits not in use, set probe_measure to True for measure support.",
                UserWarning,
            )

        return super.measure(qubit, cbit)

    def measure_y(self, qubit, cbit):
        if not self.probe_measure:
            warnings.warn(
                "Probe qubits not in use, set probe_measure to True for measure support.",
                UserWarning,
            )

        self.sdg(qubit)
        self.h(qubit)
        return self.measure(qubit, cbit)

    def measure_x(self, qubit, cbit):
        if not self.probe_measure:
            warnings.warn(
                "Probe qubits not in use, set probe_measure to True for measure support.",
                UserWarning,
            )

        self.h(qubit)
        return self.measure(qubit, cbit)
