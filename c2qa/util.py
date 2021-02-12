import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.result import Result

# Wigner plotting is currently non-functinal
# from qutip import Qobj, wigner

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
    cont = ax.contourf(x=xvec, y=xvec, z=w_fock, levels=100)
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    fig.colorbar(cont, ax=ax)

    if file:
        plt.savefig(file)
    else:
        plt.show()


def animate_wigner_fock_state(circuit: CVCircuit, result: Result, file: str = None):
    # Calculate the Wigner functions for each frame
    xvec = np.linspace(-5, 5, 200)
    w_fock = []
    for frame in range(circuit.animation_steps):
        state_vector = result.data(circuit)["snapshots"]["statevector"][
            circuit.get_snapshot_name(frame)
        ][0]
        density_matrix = cv_partial_trace(circuit, state_vector)
        w_fock.append(wigner(Qobj(density_matrix.data), xvec, xvec))

    # Create empty plot to animate
    fig, ax = plt.subplots(constrained_layout=True)

    # Animate
    anim = matplotlib.animation.FuncAnimation(
        fig=fig,
        func=_animate,
        frames=circuit.animation_steps,
        fargs=(fig, ax, xvec, w_fock),
        interval=200,
        repeat=True,
    )

    # Save to file using ffmpeg or display
    if file:
        writervideo = matplotlib.animation.FFMpegWriter(fps=60)
        anim.save(file, writer=writervideo)
    else:
        plt.show()


def _animate(frame, *fargs):
    fig = fargs[0]
    ax = fargs[1]
    xvec = fargs[2]
    w_fock = fargs[3]

    ax.clear()
    cont = ax.contourf(xvec, xvec, w_fock[frame], levels=100)
    ax.set_xlabel("x")
    ax.set_ylabel("p")
    # fig.colorbar(cont, ax=ax)  # FIXME Colorbar shifts position in animation?
