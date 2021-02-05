import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector, partial_trace
from qutip import Qobj, wigner

from c2qa import CVCircuit


def cv_partial_trace(circuit: CVCircuit, state_vector: Statevector):
    """ Return reduced density matrix by tracing out the qubits from the given Fock state vector. """

    # Find indices of qubits representing qumodes
    qmargs = []
    for reg in circuit.qmregs:
        qmargs.extend(reg.qreg)

    # Trace over the qubits not representing qumodes
    index = 0
    indices = []
    for qubit in circuit.qubits:
        if qubit not in qmargs:
            indices.append(index)
        index += 1

    return partial_trace(state_vector, indices)


def plot_wigner_fock_state(
    circuit: CVCircuit, state_vector: Statevector, file: str = None
):
    """Produce a Matplotlib figure for the Wigner function on the given state vector.

    This code follows the example from QuTiP to plot Fock state at
    http://qutip.org/docs/latest/guide/guide-visualization.html#wigner-function.

    NOTE: On Windows QuTiP requires MS Visual C++ Redistributable v14+
          See http://qutip.org/docs/latest/installation.html for platform-specific
          installation instructions.
    """
    xvec = np.linspace(-5, 5, 200)
    density_matrix = cv_partial_trace(circuit, state_vector)
    w_fock = wigner(Qobj(density_matrix.data), xvec, xvec)
    fig, ax = plt.subplots(constrained_layout=True)
    cont = ax.contourf(xvec, xvec, w_fock, 100)
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    fig.colorbar(cont, ax=ax)

    if file:
        plt.savefig(file)
    else:
        plt.show()
