from collections.abc import Sequence

import numpy as np
from qiskit.circuit import Clbit, Qubit
from qiskit_aer.noise import LocalNoisePass
from typing_extensions import TypeIs

Qumode = Sequence[Qubit]
"""Qumode as an ordered collection of qubits"""

NoisePassLike = LocalNoisePass | Sequence[LocalNoisePass]
"""Type representing one or more noise passes"""


def is_int_type(x) -> TypeIs[int | np.integer]:
    """Determines if the passed object is a built-in int or numpy integer"""

    return isinstance(x, (int, np.integer))


__all__ = ["Qumode", "NoisePassLike", "Qubit", "Clbit", "is_int_type"]
