import c2qa
import qiskit
from qiskit.quantum_info import Statevector
import numpy

    # TODO add legend for colors (make negative blue, positive red), figure out integral of Wigner function
    #      check cutoff, plot initial wigner function for original state, need to remove qubit state from statevector as it isn't part of fock state
    #      perform partial trace over qubit (look in qutip) to basically throw out qubit to disregard the qubit reigster

def test_plot_zero(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=2)
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr)

        # qr[0] and cr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        state = Statevector.from_instruction(circuit)
        print("Qumode initialized to zero:")
        print(state)
        c2qa.util.plot_wigner_fock_state(circuit, state, file="tests/zero.png")

def test_plot_one(capsys):
    with capsys.disabled(): 
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=2)
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr)

        # qr[0] and cr[0] will init to zero
        circuit.cv_initialize(1, qmr[0])

        state = Statevector.from_instruction(circuit)
        print("Qumode initialized to one:")
        print(state)
        c2qa.util.plot_wigner_fock_state(circuit, state, file="tests/one.png")

def test_partial_trace(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=2)
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr)

        # qr[0] and cr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        alpha = numpy.sqrt(numpy.pi)

        circuit.h(qr[0])
        circuit.cv_cnd_d(alpha, -alpha, qr[0], qmr[0])
        circuit.cv_d(1j * alpha, qmr[0])
        circuit.cv_cnd_d(-alpha, alpha, qr[0], qmr[0])
        circuit.cv_d(-1j * alpha, qmr[0])
        circuit.h(qr[0])
        circuit.measure(qr[0], cr[0])

        backend = qiskit.Aer.get_backend("statevector_simulator")
        job = qiskit.execute(circuit, backend)
        result = job.result()

        state = result.get_statevector(circuit)
        counts = result.get_counts(circuit)

        print(c2qa.util.cv_partial_trace(circuit, state))
