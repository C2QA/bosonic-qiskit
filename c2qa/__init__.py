"""Top-level package for c2qa-qiskit."""
# flake8: noqa

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Import order between QuTiP and QisKit is important, QuTiP fails if QisKit is imported first.
import qutip

# QisKit 0.23.4 uses deprecated np.bool in Numpy 1.20.0, this works around the warning messages
# FIXME Remove once https://github.com/Qiskit/qiskit-terra/pull/5758 is incorporated into a release!
logging.captureWarnings(True)

from c2qa.circuit import CVCircuit
from c2qa.qumoderegister import QumodeRegister

import c2qa.util
