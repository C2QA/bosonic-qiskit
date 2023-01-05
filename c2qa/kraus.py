import math
from typing import Sequence
import warnings


import c2qa
from qiskit.circuit import Instruction
from qiskit.providers.aer.noise.passes.local_noise_pass import LocalNoisePass
from qiskit.providers.aer.noise import kraus_error
from qiskit.providers.aer.noise.noiseerror import NoiseError
from qiskit.utils.units import apply_prefix
import scipy


def calculate_kraus(photon_loss_rate: float, time: float, circuit: c2qa.CVCircuit, op: Instruction = None, qubit_indices: list = None):
    """
    Calculate Kraus operator given number of photons and photon loss rate over specified time.

    Following equation 44 from Bosonic Oprations and Measurements, Girvin

    Args:
        photon_loss_rate (float): kappa, the rate of photon loss per second
        time (float): current duration of time in seconds
        circuit (CVCircuit): cq2a.CVCircuit with ops for N and a
        op (Instruction): current instruction to apply noise model
        qubit_indices (list): list of integer qubit indices on which to apply Kraus operators

    Returns:
        List of Kraus operators for all qubits up to circuit.cutoff
    """
    n = circuit.ops.N
    a = circuit.ops.a

    # FIXME -- This will resize N & a based on the number of qubits in an operation, filling zero.
    # The circuit's N & a will be the size of 2^num_qubits_per_qumode.
    # But will the position of the new values be correct? 
    # Numpy will fill zero to the right & down (higher indices). That may not be the qubit indices for the qumodes selected.
    #
    # Redefine N and a for the operation size
    # Qiskit `delay` gates are always for one qubit, see https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.delay.html
    if op is not None and (op.name == "delay" or op.num_qubits > math.sqrt(circuit.cutoff)):
        new_dim = 2**op.num_qubits
        n = n.copy()
        n.resize((new_dim, new_dim))
        a = a.copy()
        a.resize((new_dim, new_dim))

    return __kraus_operators(photon_loss_rate, time, circuit, a, n)


def calculate_kraus_tensor(photon_loss_rate: float, time: float, circuit: c2qa.CVCircuit, op: Instruction = None, qubit_indices: list = None):
    """
    Calculate Kraus operator given number of photons and photon loss rate over specified time.

    Apply Krause operator to provided qubit_indices only, tensor product with identity for remaining qubits.

    Following equation 44 from Bosonic Oprations and Measurements, Girvin

    Args:
        photon_loss_rate (float): kappa, the rate of photon loss per second
        time (float): current duration of time in seconds
        circuit (CVCircuit): cq2a.CVCircuit with ops for N and a
        op (Instruction): current instruction to apply noise model
        qubit_indices (list): list of integer qubit indices on which to apply Kraus operators

    Returns:
        List of Kraus operators for all qubits up to circuit.cutoff
    """

    # Identity for individual qubit
    qubit_eye = scipy.sparse.eye(2)

    # Kraus operators for selected qumode
    operators = __kraus_operators(photon_loss_rate, time, circuit, circuit.ops.a, circuit.ops.N)

    for operator in operators:
        value = []
        op_tensor = False
        for qubit_index in range(0, circuit.num_qubits):
            if qubit_index in qubit_indices:
                # Tensor Kraus operators (once)
                if not op_tensor:
                    value = scipy.sparse.kron(value, operator)
                    op_tensor = True
            else:
                # Tensor identity
                value = scipy.sparse.kron(value, qubit_eye)
        operators.append(value)

    return operators


def __kraus_operators(photon_loss_rate: float, time: float, circuit: c2qa.CVCircuit, a, n):
    operators = []
    for photons in range(circuit.cutoff + 1):
        kraus = math.sqrt(
            math.pow((1 - math.exp(-1 * photon_loss_rate * time)), photons)
            / math.factorial(photons)
        )
        kraus = kraus * scipy.sparse.linalg.expm(
            -1 * (photon_loss_rate / 2) * time * n
        )
        kraus = kraus.dot(a**photons)
        operators.append(kraus.toarray())

    return operators


class PhotonLossNoisePass(LocalNoisePass):
    """Add photon loss noise model to a circuit during transpiler transformation pass."""

    def __init__(
        self, photon_loss_rate: float, circuit: c2qa.CVCircuit, instructions = None, qumode = None, time_unit: str = "s", dt: float = None
    ):
        """
        Initialize the Photon Loss noise pass
        
        Args:
            photon_loss_rate (float): kappa, the rate of photon loss per second
            circuit (CVCircuit): cq2a.CVCircuit with ops for N and a, and cutoff
            instructions (str or list[str]): the instructions error applies to
            qumode (Sequence[Qubit]): qumode qubits error noise pass applies to
            time_unit (string): string photon loss rate unit of time (default "s" for seconds)
            dt (float): optional conversion factor for photon_loss_rate and operation duration to seconds
        """

        self._photon_loss_rate = photon_loss_rate
        self._circuit = circuit
        self._time_unit = time_unit
        self._dt = dt

        if instructions is None:
            self._instructions = None
        elif isinstance(instructions, list):
            self._instructions = instructions
        else:
            self._instructions = [instructions]

        if qumode is None:
            self._qumode = None
        elif isinstance(qumode, list):
            self._qumode = qumode
        else:
            self._qumode = [qumode]

        if self._qumode is None:
            self._qumode_indices = None
        else:
            self._qumode_indices = circuit.get_qubit_indices(self._qumode)

        # Convert photon loss rate to photons per second
        if self._time_unit == "dt":
            self._photon_loss_rate_sec = self._photon_loss_rate * self._dt
        else:
            conversion = 1.0 / apply_prefix(1.0, self._time_unit)
            self._photon_loss_rate_sec = self._photon_loss_rate * conversion

        super().__init__(self._photon_loss_error)

    def _photon_loss_error(self, op: Instruction, qubits: Sequence[int]):
        """Return photon loss error on each operand qubit"""
        error = None
        
        if (self._instructions is None or op.name in self._instructions) and (self._qumode_indices is None or set(qubits).issubset(self._qumode_indices)):
            if not op.duration:
                if op.duration is None:
                    warnings.warn(
                        "PhotonLossNoisePass ignores instructions without duration,"
                        " you may need to schedule circuit in advance.",
                        UserWarning,
                    )
                return None

            # Convert op duration to seconds
            if op.unit == "dt":
                if self._dt is None:
                    raise NoiseError(
                        "PhotonLossNoisePass cannot apply noise to a 'dt' unit duration"
                        " without a dt time set."
                    )
                duration = op.duration * self._dt
            else:
                duration = apply_prefix(op.duration, op.unit)

            if self._qumode_indices:
                kraus_operators = calculate_kraus_tensor(
                    self._photon_loss_rate_sec, duration, self._circuit, op, self._qumode_indices
                )
            else:
                kraus_operators = calculate_kraus(
                    self._photon_loss_rate_sec, duration, self._circuit, op
                )

            error = kraus_error(kraus_operators)

        return error
