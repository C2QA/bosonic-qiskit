"""
Combined phase and amplitude damping noise pass.
"""
import warnings
from typing import Optional, Union, Sequence, List

import numpy as np

import c2qa

from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.providers.aer.noise.passes.local_noise_pass import LocalNoisePass
from qiskit.providers.aer.noise.errors.standard_errors import phase_amplitude_damping_error


class PhaseAmplitudeDampingNoisePass(LocalNoisePass):
    """Add combined phase and amplitude damping noise after instructions."""

    def __init__(
            self,
            param_amp: float,
            param_phase: float,
            circuit: c2qa.CVCircuit,
            op_types: Optional[Union[type, Sequence[type]]] = None,
            excited_state_population: float = 0,
            canonical_kraus: bool = True,
    ):
        """Initialize PhaseAmplitudeDampingNoisePass.

        Args:
            param_amp: Amplitude damping error parameter for each qubit.
            param_phase: Phase damping error parameter for each qubit.
            circuit: c2qa.CVCircuit for iterating over available qubits and applying error to only qubits, not qumodes
            op_types: Optional, the operation types to add relaxation to. If None
                relaxation will be added to all operations.
            excited_state_population (double): the population of :math:`|1\rangle`
                                               state at equilibrium (default: 0).
            canonical_kraus (bool): Convert input Kraus matrices into the
                                    canonical Kraus representation (default: True).
        """
        params_amp = []
        qumode_indices = circuit.qumode_qubit_indices
        for i in range(circuit.num_qubits):
            if i in qumode_indices:
                params_amp.append(0.0)
            else:
                params_amp.append(param_amp)

        params_phase = []
        qumode_indices = circuit.qumode_qubit_indices
        for i in range(circuit.num_qubits):
            if i in qumode_indices:
                params_phase.append(0.0)
            else:
                params_phase.append(param_phase)

        self._params_amp = np.asarray(params_amp)
        self._params_phase = np.asarray(params_phase)
        self._excited_state_population = excited_state_population
        self._canonical_kraus = canonical_kraus

        super().__init__(self._phase_amplitude_damping_error, op_types=op_types, method="append")

    def _phase_amplitude_damping_error(
            self,
            op: Instruction,
            qubits: Sequence[int]
    ):
        """Return combined phase and amplitude damping error on each operand qubit"""
        params_amp = self._param_amp[qubits]
        params_phase = self._param_phase[qubits]

        # pylint: disable=invalid-name
        if op.num_qubits == 1:
            param_amp, param_phase = params_amp[0], params_phase[0]
            if param_amp == np.inf and param_phase == np.inf:
                return None
            return phase_amplitude_damping_error(param_amp, param_phase, self._excited_state_population, self._canonical_kraus)

        # TODO -- apply noise to control/ancilla qubits, not qumode qubits!
        # General multi-qubit case
        noise = QuantumCircuit(op.num_qubits)
        for qubit, (param_amp, param_phase) in enumerate(zip(params_amp, params_phase)):
            if param_amp == np.inf and param_phase == np.inf:
                # No phase or amplitude damping on this qubit
                continue
            error = phase_amplitude_damping_error(param_amp, param_phase, self._excited_state_population, self._canonical_kraus)
            noise.append(error.to_instruction(), [qubit])

        return noise