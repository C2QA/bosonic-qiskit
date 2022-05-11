import warnings

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from c2qa.operators import CVGate, CVOperators, ParameterizedOperator
from c2qa.qumoderegister import QumodeRegister
import qiskit.providers.aer.library.save_instructions as save


class CVCircuit(QuantumCircuit):
    """Extension of QisKit QuantumCircuit to add continuous variable (bosonic) gate support to simulations."""

    def __init__(self, *regs, name: str = None, probe_measure: bool = False):
        """Initialize the registers (at least one must be QumodeRegister), set
        the circuit name, and the number of steps to animate (default is to not animate).

        Args:
            name (str, optional): circuit name. Defaults to None.
            probe_measure (bool, optional): automatically support measurement with probe qubits. Defaults to False.

        Raises:
            ValueError: If no QumodeReigster are provided.
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
        """Integer cutoff size."""
        return self.qmregs[-1].cutoff

    @property
    def num_qubits_per_qumode(self):
        """Integer number of qubits to represent a qumode."""
        return self.qmregs[-1].num_qubits_per_qumode

    def cv_initialize(self, fock_state, qumodes):
        """Initialize the qumode to a Fock state.

        Args:
            fock_state (int): Fock state to initialize
            qumodes (list): list of qubits representing qumode

        Raises:
            ValueError: If the Fock state is greater than the cutoff.
        """
        # Qumodes are already represented as arrays of qubits,
        # but if this is an array of arrays, then we are initializing multiple qumodes.
        modes = qumodes
        if not isinstance(qumodes[0], list):
            modes = [qumodes]

        if fock_state > self.qmregs[-1].cutoff:
            raise ValueError("The given Fock state is greater than the cutoff.")

        for qumode in modes:
            value = np.zeros((self.qmregs[-1].cutoff,), dtype=np.complex_)
            value[fock_state] = 1 +0j

            super().initialize(value, qumode)

    @staticmethod
    def cv_conditional(name, op_0, op_1, num_qubits_per_qumode, num_qumodes=1):
        """Make two operators conditional (i.e., controlled by qubit in either the 0 or 1 state)

        Args:
            name (str): name of conditional gate
            op_0 (ndarray): operator matrix for 0 controlled gate
            op_1 (ndarray): operator matrix for 1 controlled gate
            num_qubits_per_qumode (int): number of qubits representing a single qumode
            num_qumodes (int, optional): number of qubodes used in this gate. Defaults to 1.

        Returns:
            Instruction: QisKit Instruction appended to the circuit
        """
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

    def save_circuit(self, conditional, pershot, label="statevector"):
        """Save the simulator statevector using a qiskit class"""
        return save.save_statevector(
            label=label, conditional=conditional, pershot=pershot
        )

    def cv_d(self, alpha, qumode):
        """Displacement gate.

        Args:
            alpha (real): displacement
            qumode (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        operator = ParameterizedOperator(self.ops.d, alpha)
        return self.append(CVGate(data=operator, label="D"), qargs=qumode)

    def cv_cd(self, alpha, beta, qumode, qubit_ancilla):
        """Conditional displacement gate.

        Args:
            alpha (real): displacement for 0 control
            beta (real): displacemet for 1 control
            ctrl (Qubit): QisKit control Qubit
            qumode (list): list of qubits representing qumode
            inverse (bool): True to calculate the inverse of the operator matrices

        Returns:
            Instruction: QisKit instruction
        """
        operator = ParameterizedOperator(self.ops.cd, alpha, beta)
        return self.append(CVGate(data=operator, label="CD"), qargs=qumode + [qubit_ancilla])

    def cv_ecd(self, alpha, qumode, qubit_ancilla):
        """Echoed controlled displacement gate.

        Args:
            alpha (real): displacement
            qumode (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        operator = ParameterizedOperator(self.ops.ecd, alpha)
        return self.append(CVGate(data=operator, label="ECD"), qargs=qumode + [qubit_ancilla])

    def cv_rh1(self, alpha, qumode_a, qumode_b, qubit_ancilla):
        operator = ParameterizedOperator(self.ops.rh1, alpha)
        self.append(CVGate(data=operator, label="rh1"), qargs=qumode_a + qumode_b + [qubit_ancilla])

    def cv_rh2(self, alpha, qumode_a, qumode_b, qubit_ancilla):
        operator = ParameterizedOperator(self.ops.rh2, alpha)
        self.append(CVGate(data=operator, label="rh2"), qargs=qumode_a + qumode_b + [qubit_ancilla])

    def cv_cnd_d(self, alpha, beta, ctrl, qumode, inverse: bool = False):
        """Conditional displacement gate.

        Args:
            alpha (real): displacement for 0 control
            beta (real): displacemet for 1 control
            ctrl (Qubit): QisKit control Qubit
            qumode (list): list of qubits representing qumode
            inverse (bool): True to calculate the inverse of the operator matrices

        Returns:
            Instruction: QisKit instruction
        """
        op_0 = ParameterizedOperator(self.ops.d, alpha, inverse=inverse)
        op_1 = ParameterizedOperator(self.ops.d, beta, inverse=inverse)
        return self.append(
            CVCircuit.cv_conditional("Dc", op_0, op_1, self.num_qubits_per_qumode),
            [ctrl] + qumode,
        )

    def cv_s(self, z, qumode):
        """Squeezing gate.

        Args:
            z (real): squeeze
            qumode (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        operator = ParameterizedOperator(self.ops.s, z)
        return self.append(CVGate(data=operator, label="S"), qargs=qumode)

    def cv_cnd_s(self, z_a, z_b, ctrl, qumode_a):
        """Conditional squeezing gate

        Args:
            z_a (real): squeeze for 0 control
            z_b (real): squeeze for 1 control
            ctrl (Qubit): QisKit control Qubit
            qumode_a (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        op_0 = ParameterizedOperator(self.ops.s, z_a)
        op_1 = ParameterizedOperator(self.ops.s, z_b)
        return self.append(
            CVCircuit.cv_conditional("Sc", op_0, op_1, self.num_qubits_per_qumode),
            [ctrl] + qumode_a,
        )

    def cv_s2(self, z, qumode_a, qumode_b):
        """Two-mode squeezing gate

        Args:
            z (real): squeeze
            qumode_a (list): list of qubits representing first qumode
            qumode_b (list): list of qubits representing second qumode

        Returns:
            Instruction: QisKit instruction
        """
        operator = ParameterizedOperator(self.ops.s2, z)
        return self.append(CVGate(data=operator, label="S2"), qargs=qumode_a + qumode_b)

    def cv_bs(self, phi, qumode_a, qumode_b):
        """Beam splitter gate.

        Args:
            phi (real): real phase
            qumode_a (list): list of qubits representing first qumode
            qumode_b (list): list of qubits representing second qumode

        Returns:
            Instruction: QisKit instruction
        """
        operator = ParameterizedOperator(self.ops.bs, phi)
        return self.append(CVGate(data=operator, label="BS"), qargs=qumode_a + qumode_b)

    def cv_cnd_bs(self, phi, chi, ctrl, qumode_a, qumode_b):
        """Conditional beam splitter gate.

        Args:
            phi (real): real phase for 0 qubit state
            chi (real): phase for 1 qubit state
            ctrl (Qubit): QisKit control Qubit
            qumode_a (list): list of qubits representing first qumode
            qumode_b (list): list of qubits representing second qumode

        Returns:
            Instruction: QisKit instruction
        """
        op_0 = ParameterizedOperator(self.ops.bs, phi)
        op_1 = ParameterizedOperator(self.ops.bs, chi)
        return self.append(
            CVCircuit.cv_conditional(
                "BSc", op_0, op_1, self.num_qubits_per_qumode, num_qumodes=2
            ),
            [ctrl] + qumode_a + qumode_b,
        )

    def cv_cpbs(self, phi, qumode_a, qumode_b, qubit_ancilla):
        """Controlled phase two-mode beam splitter

        Args:
            phi (real): phase
            qubit_ancilla (Qubit): QisKit control Qubit
            qumode_a (list): list of qubits representing first qumode
            qumode_b (list): list of qubits representing second qumode

        Returns:
            Instruction: QisKit instruction
        """
        operator = ParameterizedOperator(self.ops.cpbs, phi)
        self.append(CVGate(data=operator, label="CPBS"), qargs=qumode_a + qumode_b + [qubit_ancilla])

    def cv_cpbs_z2vqe(self, phi, qumode_a, qumode_b, qubit_ancilla):
        """Controlled phase two-mode beam splitter

        Args:
            phi (real): phase
            qubit_ancilla (Qubit): QisKit control Qubit
            qumode_a (list): list of qubits representing first qumode
            qumode_b (list): list of qubits representing second qumode

        Returns:
            Instruction: QisKit instruction
        """
        operator = ParameterizedOperator(self.ops.cpbs_z2vqe, phi)
        self.append(CVGate(data=operator, label="CPBS"), qargs=qumode_a + qumode_b + [qubit_ancilla])


    def cv_r(self, phi, qumode):
        """Phase space rotation gate.

        Args:
            phi (real): rotation
            qumode (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        operator = ParameterizedOperator(self.ops.r, phi)
        return self.append(CVGate(data=operator, label="R"), qargs=qumode)

    def cv_qdcr(self, theta, qumode_a, qubit_ancilla):
        """Qubit dependent cavity rotation gate.

        Args:
            theta (real): phase
            qumode_a (list): list of qubits representing qumode
            qubit_ancilla (qubit): QisKit control qubit

        Returns:
            Instruction: QisKit instruction
        """
        operator = ParameterizedOperator(self.ops.qubitDependentCavityRotation, theta)
        self.append(CVGate(data=operator, label="QDCR"), qargs=qumode_a + [qubit_ancilla])

    def cv_cp(self, theta, qumode_a, qubit_ancilla):
        """Controlled parity gate.

        Args:
            qumode_a (list): list of qubits representing qumode
            qubit_ancilla (qubit): QisKit control qubit

        Returns:
            Instruction: QisKit instruction
        """
        operator = ParameterizedOperator(self.ops.controlledparity, theta)
        self.append(CVGate(data=operator, label="CP"), qargs=qumode_a + [qubit_ancilla])

    def cv_snap(self, theta, n, qumode_a):
        """SNAP (Selective Number-dependent Arbitrary Phase) gate.

        Args:
            theta (real): phase
            n (integer): Fock state in which the mode should acquire the phase
            qumode_a (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        operator = ParameterizedOperator(self.ops.snap, theta, n)
        self.append(CVGate(data=operator, label="SNAP"), qargs=qumode_a)

    def cv_eswap(self, theta, qumode_a, qumode_b):
        """Exponential SWAP gate.

        Args:
            theta (real): phase
            qumode_a (list): list of qubits representing qumode
            qumode_b (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        operator = ParameterizedOperator(self.ops.eswap, theta)
        self.append(CVGate(data=operator, label="eSWAP"), qargs=qumode_a + qumode_b)

    def cv_pncqr(self, theta, n, qumode_a, qubit_ancilla, qubit_rotation):
        """Photon Number Controlled Qubit Rotation gate.

        Args:
            theta (real): phase
            n (integer): Fock state in which the mode should acquire the phase
            qumode_a (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        operator = ParameterizedOperator(self.ops.photonNumberControlledQubitRotation, theta, n, qubit_rotation)
        self.append(CVGate(data=operator, label="PNCQR"), qargs=qumode_a + [qubit_ancilla])

    def measure_z(self, qubit, cbit):
        """Measure qubit in z using probe qubits

        Args:
            qubit (Qubit): QisKit qubit to measure
            cbit (ClassicalBit): QisKit classical bit to measure into

        Returns:
            Instruction: QisKit measure instruction
        """
        if not self.probe_measure:
            warnings.warn(
                "Probe qubits not in use, set probe_measure to True for measure support.",
                UserWarning,
            )

        return super.measure(qubit, cbit)

    def measure_y(self, qubit, cbit):
        """Measure qubit in y using probe qubits

        Args:
            qubit (Qubit): QisKit qubit to measure
            cbit (ClassicalBit): QisKit classical bit to measure into

        Returns:
            Instruction: QisKit measure instruction
        """
        if not self.probe_measure:
            warnings.warn(
                "Probe qubits not in use, set probe_measure to True for measure support.",
                UserWarning,
            )

        self.sdg(qubit)
        self.h(qubit)
        return self.measure(qubit, cbit)

    def measure_x(self, qubit, cbit):
        """Measure qubit in x using probe qubits

        Args:
            qubit (Qubit): QisKit qubit to measure
            cbit (ClassicalBit): QisKit classical bit to measure into

        Returns:
            Instruction: QisKit measure instruction
        """
        if not self.probe_measure:
            warnings.warn(
                "Probe qubits not in use, set probe_measure to True for measure support.",
                UserWarning,
            )

        self.h(qubit)
        return self.measure(qubit, cbit)

    def cv_measure(self, qregister_list, cregister_list):
        """Measure QumodeRegisters, QuantumRegisters, and ClassicalRegisters in specified order

                Args:
                    qregister_list (List): List of individual QumodeRegister Qubits, QuantumRegister Qubits and ClassicalRegister Qubits
                    cbit (ClassicalBit): List of classical bits to measure into

                Returns:
                    Instruction: QisKit measure instruction
                """
        if not self.probe_measure:
            warnings.warn(
                "Probe qubits not in use, set probe_measure to True for measure support.",
                UserWarning,
            )

        # Flattens the list (if necessary)
        flat_list = []
        for el in qregister_list:
            if isinstance(el, list):
                flat_list += el
            else:
                flat_list += [el]
        # Check to see if too many classical registers were passed in. If not, only use those needed (starting with least significant bit).
        # This piece is useful so that the user doesn't need to think about how many bits are needed to read out a list of qumodes, qubits, etc.
        if len(flat_list) < len(cregister_list):
            self.measure(flat_list, cregister_list[0:len(flat_list)])
        else:
            self.measure(flat_list, cregister_list)
