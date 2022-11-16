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


def calculate_kraus(photon_loss_rate: float, time: float, circuit: c2qa.CVCircuit, op: Instruction = None):
    """
    Calculate Kraus operator given number of photons and photon loss rate over specified time.

    Following equation 44 from Bosonic Oprations and Measurements, Girvin

    Args:
        photon_loss_rate (float): kappa, the rate of photon loss per second
        time (float): current duration of time in seconds
        circuit (CVCircuit): cq2a.CVCircuit with ops for N and a

    Returns:
        List of Kraus operators for all qubits up to circuit.cutoff
    """
    operators = []

    n = circuit.ops.N
    a = circuit.ops.a
    if op is not None and op.num_qubits > math.sqrt(circuit.cutoff):
        new_dim = 2**op.num_qubits
        n = n.copy()
        n.resize((new_dim, new_dim))
        a = a.copy()
        a.resize((new_dim, new_dim))

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
        self, photon_loss_rate: float, circuit: c2qa.CVCircuit, time_unit: str = "s", dt: float = None
    ):
        """
        Initialize the Photon Loss noise pass
        
        Args:
            photon_loss_rate (float): kappa, the rate of photon loss per second
            circuit (CVCircuit): cq2a.CVCircuit with ops for N and a
            time_unit (string): string photon loss rate unit of time (default "s" for seconds)
            dt (float): optional conversion factor for photon_loss_rate and operation duration to seconds
        """

        self._photon_loss_rate = photon_loss_rate
        self._circuit = circuit
        self._time_unit = time_unit
        self._dt = dt

        # Convert photon loss rate to photons per second
        if self._time_unit == "dt":
            self._photon_loss_rate_sec = self._photon_loss_rate * self._dt
        else:
            conversion = 1.0 / apply_prefix(1.0, self._time_unit)
            self._photon_loss_rate_sec = self._photon_loss_rate * conversion

        super().__init__(self._photon_loss_error)

    def _photon_loss_error(self, op: Instruction, qubits: Sequence[int]):
        """Return photon loss error on each operand qubit"""
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

        kraus_operators = calculate_kraus(
            self._photon_loss_rate_sec, duration, self._circuit, op
        )

        error = kraus_error(kraus_operators)

        return error
