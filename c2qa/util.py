import c2qa
import matplotlib
import numpy
import qiskit
import qutip

def cv_partial_trace(circuit:c2qa.CVCircuit, state_vector:qiskit.quantum_info.Statevector):
    """ Return reduced density matrix by tracing out the qubits from the given Fock state vector. """

    # Find indices of qubits representing qumodes
    qargs = []
    for reg in circuit.qmregs:
        qargs.extend(reg.qreg)

    index = 0
    indices = []
    for qubit in circuit.qubits:
        if qubit in qargs:
            indices.append(index)
        index += 1

    return qiskit.quantum_info.partial_trace(state_vector, indices)

def plot_wigner_fock_state(circuit:c2qa.CVCircuit, state_vector:qiskit.quantum_info.Statevector, file:str = None):
    """ Produce a Matplotlib figure for the Wigner function on the given state vector. 
        
        This code follows the example from QuTiP to plot Fock state at http://qutip.org/docs/latest/guide/guide-visualization.html#wigner-function 
        NOTE: On Windows QuTiP requires MS Visual C++ Redistributable v14+
    """
    xvec = numpy.linspace(-5,5,200)
    density_matrix = cv_partial_trace(circuit, state_vector)
    w_fock = qutip.wigner(qutip.Qobj(density_matrix.data), xvec, xvec)
    fig, ax = matplotlib.pyplot.subplots(constrained_layout=True)
    cont = ax.contourf(xvec, xvec, w_fock, 100)
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    cb = fig.colorbar(cont, ax=ax)

    if file:
        matplotlib.pyplot.savefig(file)
    else:
        matplotlib.pyplot.show()
