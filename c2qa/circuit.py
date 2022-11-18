import copy
import warnings

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.parametertable import ParameterTable
import qiskit.providers.aer.library.save_instructions as save

from c2qa.operators import CVOperators, ParameterizedUnitaryGate
from c2qa.qumoderegister import QumodeRegister


class CVCircuit(QuantumCircuit):
    """Extension of QisKit QuantumCircuit to add continuous variable (bosonic) gate support to simulations."""

    def __init__(self, *regs, name: str = None, probe_measure: bool = False):
        """Initialize the registers (at least one must be QumodeRegister) and set the circuit name.

        Args:
            name (str, optional): circuit name. Defaults to None.
            probe_measure (bool, optional): automatically support measurement with probe qubits. Defaults to False.

        Raises:
            ValueError: If no QumodeRegister is provided.
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
        self.cv_snapshot_id = 0

    def merge(self, circuit: QuantumCircuit):
        """
        Merge in properties of QisKit QuantumCircuit into this instance.

        Useful if QisKit returned a new instance of QuantumCircuit after
        passing in this instance. Calling merge() can merge the two, keeping
        this instance.

        See https://qiskit.org/documentation/_modules/qiskit/circuit/quantumcircuit.html#QuantumCircuit.copy
        """

        self.qregs = circuit.qregs.copy()
        self.cregs = circuit.cregs.copy()
        self._qubits = circuit._qubits.copy()
        self._ancillas = circuit._ancillas.copy()
        self._clbits = circuit._clbits.copy()
        self._qubit_indices = circuit._qubit_indices.copy()
        self._clbit_indices = circuit._clbit_indices.copy()

        instr_instances = {id(instr): instr for instr, _, __ in circuit._data}

        instr_copies = {id_: instr.copy() for id_, instr in instr_instances.items()}

        self._parameter_table = ParameterTable(
            {
                param: [
                    (instr_copies[id(instr)], param_index)
                    for instr, param_index in circuit._parameter_table[param]
                ]
                for param in circuit._parameter_table
            }
        )

        self._data = [
            (instr_copies[id(inst)], qargs.copy(), cargs.copy())
            for inst, qargs, cargs in circuit._data
        ]

        self._calibrations = copy.deepcopy(circuit._calibrations)
        self._metadata = copy.deepcopy(circuit._metadata)

    @property
    def cutoff(self):
        """Integer cutoff size."""
        return self.qmregs[-1].cutoff

    @property
    def num_qubits_per_qumode(self):
        """Integer number of qubits to represent a qumode."""
        return self.qmregs[-1].num_qubits_per_qumode

    @property
    def qumode_qubits(self):
        """All the qubits representing the qumode registers on the circuit"""
        qubits = []
        for reg in self.qmregs:
            qubits += reg[::]
        return qubits

    @property
    def qumode_qubit_indices(self):
        """A qubit index list of the qubits representing the qumode registers on the circuit"""
        qmodes = self.qumode_qubits
        indices = []

        for index, qubit in enumerate(self.qubits):
            if qubit in qmodes:
                indices.append(index)

        return indices

    @property
    def cv_gate_labels(self):
        """
        All the CV gate names on the current circuit. These will either be
        instances of ParameterizedUnitaryGate or be instances of super
        Intstruction and flagged with 'cv_conditional' if a conditional gate.
        """
        cv_gates = set()
        for instruction, qargs, cargs in self.data:
            if isinstance(instruction, ParameterizedUnitaryGate):
                cv_gates.add(instruction.label)
            elif hasattr(instruction, "cv_conditional") and instruction.cv_conditional:
                cv_gates.add(instruction.label)
        return list(cv_gates)

    def cv_snapshot(self):
        """Wrap the Qiskit QuantumCircuit Snapshot function, giving it a known label for later Wigner function plot generation"""
        self.snapshot(f"cv_snapshot_{self.cv_snapshot_id}")
        self.cv_snapshot_id += 1

    def add_qubit_register(self, *regs):
        """Add a qubit register to the circuit.

        Args:
            regs (list): List of Registers
        """
        for reg in regs:
            if isinstance(reg, QuantumRegister):
                self._qubit_regs.append(reg)
            else:
                raise ValueError(
                    "Only QuantumRegisters are allowed to be added for now"
                )
        super().add_register(*regs)

    def cv_initialize(self, params, qumodes):
        """Initialize qumode (or qumodes) to a particular state specified by params

        Args:
            params (list or int): If an int, all specified qumodes will be initialized to the Fock state with n=params.
                                  If a list, all specified qumodes will be initialized to a superposition of Fock states,
                                  with params[n] the complex amplitude of Fock state |n>. The length of params must be less
                                  than or equal to the cutoff.
            qumodes (list): list of qubits representing a single qumode, or list of multiple qumodes

        Raises:
            ValueError: If the Fock state is greater than the cutoff.
        """
        # Qumodes are already represented as arrays of qubits,
        # but if this is an array of arrays, then we are initializing multiple qumodes.
        modes = qumodes
        if not isinstance(qumodes[0], list):
            modes = [qumodes]

        if isinstance(params, int):
            if params >= self.qmregs[-1].cutoff:
                raise ValueError("The given Fock state is greater than the cutoff.")

            for qumode in modes:
                value = np.zeros((self.qmregs[-1].cutoff,), dtype=np.complex_)
                value[params] = 1 + 0j

                super().initialize(value, qumode)
        else:
            if len(params) > self.qmregs[-1].cutoff:
                raise ValueError("len(params) exceeds the cutoff.")

            for qumode in modes:
                params = np.array(params) / np.linalg.norm(np.array(params))
                amplitudes = np.zeros((self.qmregs[-1].cutoff,), dtype=np.complex_)
                for ind in range(len(params)):
                    amplitudes[ind] = complex(params[ind])

                super().initialize(amplitudes, qumode)

    @staticmethod
    def cv_conditional(
        name,
        op,
        params_0,
        params_1,
        num_qubits_per_qumode,
        num_qumodes=1,
        duration=100,
        unit="ns",
    ):
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

        gate_0 = ParameterizedUnitaryGate(
            op,
            params_0,
            num_qubits=num_qubits_per_qumode * num_qumodes,
            duration=duration,
            unit=unit,
        )
        gate_1 = ParameterizedUnitaryGate(
            op,
            params_1,
            num_qubits=num_qubits_per_qumode * num_qumodes,
            duration=duration,
            unit=unit,
        )

        sub_circ.append(gate_0.control(num_ctrl_qubits=1, ctrl_state=0), qargs)
        sub_circ.append(gate_1.control(num_ctrl_qubits=1, ctrl_state=1), qargs)

        # Create a single instruction for the conditional gate, flag it for later processing
        inst = sub_circ.to_instruction(label=name)
        inst.cv_conditional = True
        inst.num_qubits_per_qumode = num_qubits_per_qumode
        inst.num_qumodes = num_qumodes

        return inst

    def save_circuit(self, conditional, pershot, label="statevector"):
        """Save the simulator statevector using a qiskit class"""
        return save.save_statevector(
            label=label, conditional=conditional, pershot=pershot
        )

    def cv_r(self, theta, qumode, duration=100, unit="ns"):
        """Phase space rotation gate.

        Args:
            theta (real): rotation
            qumode (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.r, [theta], num_qubits=len(qumode), label="R", duration=duration, unit=unit
            ),
            qargs=qumode,
        )

    def cv_d(self, alpha, qumode, duration=100, unit="ns"):
        """Displacement gate.

        Args:
            alpha (real or complex): displacement
            qumode (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.d, [alpha], num_qubits=len(qumode), label="D", duration=duration, unit=unit
            ),
            qargs=qumode,
        )

    def cv_sq(self, theta, qumode, duration=100, unit="ns"):
        """Squeezing gate.

        Args:
            theta (real or complex): squeeze
            qumode (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.s, [theta], num_qubits=len(qumode), label="S", duration=duration, unit=unit
            ),
            qargs=qumode,
        )

    def cv_sq2(self, theta, qumode_a, qumode_b, duration=100, unit="ns"):
        """Two-mode squeezing gate

        Args:
            theta (real or complex): squeeze
            qumode_a (list): list of qubits representing first qumode
            qumode_b (list): list of qubits representing second qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.s2, [theta], num_qubits=len(qumode_a) + len(qumode_b), label="S2", duration=duration, unit=unit
            ),
            qargs=qumode_a + qumode_b,
        )

    def cv_bs(self, theta, qumode_a, qumode_b, duration=100, unit="ns"):
        """Two-mode beam splitter gate.

        Args:
            theta (real or complex): beamsplitter phase
            qumode_a (list): list of qubits representing first qumode
            qumode_b (list): list of qubits representing second qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.bs,
                [theta],
                num_qubits=len(qumode_a) + len(qumode_b),
                label="BS",
                duration=duration,
                unit=unit
            ),
            qargs=qumode_a + qumode_b,
        )

    def cv_c_r(self, theta, qumode, qubit, duration=100, unit="ns"):
        """Qubit dependent phase-space rotation gate.

        Args:
            theta (real): phase
            qumode_a (list): list of qubits representing qumode
            qubit_ancilla (qubit): QisKit control qubit

        Returns:
            Instruction: QisKit instruction
        """
        self.append(
            ParameterizedUnitaryGate(
                self.ops.cr,
                [theta],
                num_qubits=len(qumode) + 1,
                label="cR",
                duration=duration,
                unit=unit
            ),
            qargs=qumode + [qubit],
        )

    def cv_c_rx(self, theta, qumode, qubit, duration=100, unit="ns"):
        """Qubit dependent phase-space rotation around sigma^x gate.

        Args:
            theta (real): phase
            qumode_a (list): list of qubits representing qumode
            qubit_ancilla (qubit): QisKit control qubit

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.crx, [theta], num_qubits=len(qumode) + 1, label="cRX", duration=duration, unit=unit
            ),
            qargs=qumode + [qubit],
        )

    def cv_c_ry(self, theta, qumode, qubit, duration=100, unit="ns"):
        """Qubit dependent phase-space rotation around sigma^y gate.

        Args:
            theta (real): phase
            qumode_a (list): list of qubits representing qumode
            qubit_ancilla (qubit): QisKit control qubit

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.cry, [theta], num_qubits=len(qumode) + 1, label="cRY", duration=duration, unit=unit
            ),
            qargs=qumode + [qubit],
        )

    def cv_c_d(self, theta, qumode, qubit, beta=None, duration=100, unit="ns"):
        """Conditional displacement gate.

        Args:
            theta (real): displacement
            qumode (list): list of qubits representing qumode
            qubit (Qubit): control qubit
            beta (real): By default is None, and qumode will be displaced by alpha and -alpha for qubit
            state 0 and 1, respectively. If specified, qumode will be displaced by alpha and beta for qubit state 0 and 1.

        Returns:
            Instruction: QisKit instruction
        """
        if beta is None:
            beta = -theta

        return self.append(
            ParameterizedUnitaryGate(
                self.ops.cd, [theta, beta], num_qubits=len(qumode) + 1, label="cD", duration=duration, unit=unit
            ),
            qargs=qumode + [qubit],
        )

    def cv_ecd(self, theta, qumode, qubit, duration=100, unit="ns"):
        """Echoed controlled displacement gate.

        Args:
            theta (real): displacement
            qumode (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.ecd, [theta], num_qubits=len(qumode) + 1, label="ECD", duration=duration, unit=unit
            ),
            qargs=qumode + [qubit],
        )

    def cv_c_bs(self, theta, qumode_a, qumode_b, qubit, duration=100, unit="ns"):
        """Controlled phase two-mode beam splitter

        Args:
            theta (real or complex): phase
            qubit_ancilla (Qubit): QisKit control Qubit
            qumode_a (list): list of qubits representing first qumode
            qumode_b (list): list of qubits representing second qumode

        Returns:
            Instruction: QisKit instruction
        """
        self.append(
            ParameterizedUnitaryGate(
                self.ops.cbs,
                [theta],
                num_qubits=len(qumode_a) + len(qumode_b) + 1,
                label="cBS",
                duration=duration,
                unit=unit
            ),
            qargs=qumode_a + qumode_b + [qubit],
        )

    def cv_snap(self, theta, n, qumode, qubit=None, duration=100, unit="ns"):
        """SNAP (Selective Number-dependent Arbitrary Phase) gate.
        TODO: Add second snap implementation that includes sigma_z qubit

        Args:
            theta (real): phase
            n (integer): Fock state in which the mode should acquire the phase
            qumode (list): list of qubits representing qumode
            qubit (Qubit): control qubit. If no qubit is passed, the gate will implement for sigma^z = +1.

        Returns:
            Instruction: QisKit instruction
        """
        if qubit is None:
            self.append(
                ParameterizedUnitaryGate(
                    self.ops.snap, [theta, n], num_qubits=len(qumode), label="SNAP", duration=duration, unit=unit
                ),
                qargs=qumode,
            )
        else:
            self.append(
                ParameterizedUnitaryGate(
                    self.ops.csnap, [theta, n], num_qubits=len(qumode), label="SNAP", duration=duration, unit=unit
                ),
                qargs=qumode + [qubit],
            )

    def cv_eswap(self, theta, qumode_a, qumode_b, duration=100, unit="ns"):
        """Exponential SWAP gate.

        Args:
            theta (real): phase
            qumode_a (list): list of qubits representing qumode
            qumode_b (list): list of qubits representing qumode

        Returns:
            Instruction: QisKit instruction
        """
        self.append(
            ParameterizedUnitaryGate(
                self.ops.eswap,
                [theta],
                num_qubits=len(qumode_a) + len(qumode_b),
                label="eSWAP",
                duration=duration,
                unit=unit
            ),
            qargs=qumode_a + qumode_b,
        )

    def cv_c_sq(self, theta, qumode, qubit, duration=100, unit="ns"):
        """Conditional squeezing gate.

        Args:
            theta (real or complex): squeezing ampltiude
            qumode (list): list of qubits representing qumode
            qubit (Qubit): control Qubit

        Returns:
            Instruction: QisKit instruction
        """
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.csq, [theta], num_qubits=len(qumode) + 1, label="cS", duration=duration, unit=unit
            ),
            qargs=qumode + [qubit],
        )

    def cv_testqubitorderf(self, phi, qubit_1, qubit_2, duration=100, unit="ns"):
        return self.append(
            ParameterizedUnitaryGate(
                self.ops.testqubitorderf, [phi], label="testqubitorderf", num_qubits=2, duration=duration, unit=unit
            ),
            qargs=[qubit_1] + [qubit_2],
        )

    def measure_z(self, qubit, cbit, duration=100, unit="ns"):
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

    def cv_measure(self, qubit_qumode_list, cbit_list):
        """Measure Qumodes and Qubits in qubit_qumode_list and map onto
        classical bits specified in cbit_list.

        Args:
            qubit_qumode_list(List): List of individual Qubits and Qumodes
                (i.e., indexed elements of QubitRegisters and QumodeRegisters)
            cbit_list (List): List of classical bits to map measurements onto.
                Note: Measurement of qumodes requires log(c) classical bits,
                where c is the cutoff.
                If len(cbit_list) is greater than the required number of
                classical bits, excess will be ignored. If len(cbit_list) is
                insufficient, an error will be thrown.

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
        for el in qubit_qumode_list:
            if isinstance(el, list):
                flat_list += el
            else:
                flat_list += [el]

        # Check to see if too many classical registers were passed in.
        # If not, only use those needed (starting with least significant bit).
        # This piece is useful so that the user doesn't need to think about
        # how many bits are needed to read out a list of qumodes, qubits, etc.
        if len(flat_list) < len(cbit_list):
            self.measure(flat_list, cbit_list[0:len(flat_list)])
        else:
            self.measure(flat_list, cbit_list)
