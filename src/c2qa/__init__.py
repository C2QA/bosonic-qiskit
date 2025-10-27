"""Top-level package for bosonic-qiskit."""

# flake8: noqa

from c2qa.circuit import CVCircuit
from c2qa.qumoderegister import QumodeRegister

import c2qa.animate
import c2qa.discretize
import c2qa.kraus
import c2qa.operators
import c2qa.parameterized_unitary_gate
import c2qa.util
import c2qa.wigner

import warnings

# --- WARNING MESSAGE START ---
warnings.warn(
    "The 'c2qa' package has been deprecated and will be renamed "
    "to 'bosonic-qiskit' in a future release of version 15.0 on November 16, 2025. "
    "Imports will need to be modified to `bosonic_qiskit`. Please update your imports accordingly. ",
    DeprecationWarning,
    stacklevel=2 # This points the warning to the line that imports the package
)
# --- WARNING MESSAGE END ---
