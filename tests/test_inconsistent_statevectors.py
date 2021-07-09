import numpy
import qiskit
from qiskit import Aer
import scipy.linalg

# Define parameters
num_qubits_per_qumode = 2
cutoff = 2 ** num_qubits_per_qumode
alpha = numpy.sqrt(numpy.pi)


def displacement_operator(arg):
    """Create displacement operator matrix"""
    a = numpy.diag(numpy.sqrt(range(1, cutoff)), k=1)
    a_dag = a.conj().T
    return scipy.linalg.expm((arg * a_dag) - (numpy.conjugate(arg) * a))


def displacemnt_gate(circuit, arg, qumode):
    circuit.unitary(displacement_operator(arg), qumode)


def conditional_displacement_gate(circuit, arg_0, arg_1, qbit, qumode):
    """Append a conditional displacement to the circuit
    Displace by arg_0 if qbit is 0, by arg_1 if qbit is 1."""

    op_0 = displacement_operator(arg_0)
    op_1 = displacement_operator(arg_1)

    circuit.append(
        qiskit.extensions.UnitaryGate(op_0).control(num_ctrl_qubits=1, ctrl_state=0),
        [qbit] + qumode,
    )
    circuit.append(
        qiskit.extensions.UnitaryGate(op_1).control(num_ctrl_qubits=1, ctrl_state=1),
        [qbit] + qumode,
    )


def qumode_initialize(circuit, fock_state, qumode):
    """Initialize the qumode to a Fock state."""

    value = numpy.zeros((cutoff,))
    value[fock_state] = 1

    circuit.initialize(value, qumode)


def run_displacement_calibration(enable_measure):
    """
    Run the simulation that has different state vector results on Windows vs Linux.

      - Create the qumode register as a QuantumRegiser where n qubits represent a qumode.
      - Create a QuantumRegister(1) to represent the control qubit.
      - Initialize the qubits.
      - Simulate the circuit.
    """

    # Instantiate QisKit registers & circuit
    qmr = qiskit.QuantumRegister(num_qubits_per_qumode)  # qumode register
    qr = qiskit.QuantumRegister(1)
    cr = qiskit.ClassicalRegister(1)
    circuit = qiskit.QuantumCircuit(qmr, qr, cr)

    # Initialize the qumode Fock state
    # qr[0] and cr[0] will init to zero
    qumode_initialize(circuit, 0, qmr[0:])

    circuit.h(qr[0])
    conditional_displacement_gate(circuit, alpha, -alpha, qr[0], qmr[0:])
    displacemnt_gate(circuit, 1j * alpha, qmr[0:])
    conditional_displacement_gate(circuit, -alpha, alpha, qr[0], qmr[0:])
    displacemnt_gate(circuit, -1j * alpha, qmr[0:])
    circuit.h(qr[0])

    circuit.save_statevector()

    if enable_measure:
        circuit.measure(qr[0], cr[0])

    backend = Aer.get_backend("aer_simulator")
    job = qiskit.execute(circuit, backend)
    result = job.result()
    state = result.get_statevector(circuit)
    counts = result.get_counts(circuit)

    assert len(state) > 0
    assert counts

    # print(state)
    # print(counts.int_outcomes())


def test_displacement_calibration(capsys):
    with capsys.disabled():
        run_displacement_calibration(False)
        run_displacement_calibration(True)
