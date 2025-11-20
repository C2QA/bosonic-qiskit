from collections.abc import Sequence

import numpy as np
from qiskit.circuit import Clbit, Qubit
from typing_extensions import TypeIs

Qumode = Sequence[Qubit]


def is_int_type(x) -> TypeIs[int | np.integer]:
    return isinstance(x, (int, np.integer))
