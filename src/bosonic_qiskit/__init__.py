"""Top-level package for bosonic-qiskit."""

# flake8: noqa

from .circuit import CVCircuit
from .qumoderegister import QumodeRegister
from . import (
    animate,
    discretize,
    kraus,
    operators,
    parameterized_unitary_gate,
    util,
    wigner,
)

import warnings

# --- WARNING MESSAGE START ---
warnings.warn(
    "The 'c2qa' package has been deprecated and will be renamed "
    "to 'bosonic-qiskit' in a future release of version 15.0. "
    "Imports will need to be modified to `bosonic_qiskit`. Please update your imports accordingly. ",
    DeprecationWarning,
    stacklevel=2,  # This points the warning to the line that imports the package
)
# --- WARNING MESSAGE END ---

__all__ = [
    "CVCircuit",
    "QumodeRegister",
    "animate",
    "discretize",
    "kraus",
    "operators",
    "parameterized_unitary_gate",
    "util",
    "wigner",
]
