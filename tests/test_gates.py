
import c2qa
import numpy
import qiskit


def test_gates():
    """ Verify that we can use the gates, not that they are actually working. """

    # ===== Constants =====

    n_qubits_per_mode = 3
    n_qumodes = 2
    n_qubits = 3
    n_cbits = 6

    alpha = 1
    beta = -1
    phi = numpy.pi/2
    z_a = 1
    z_b = -1

    # ==== Initialize circuit =====

    qmr = c2qa.QumodeRegister(n_qumodes, n_qubits_per_mode)
    qr = qiskit.QuantumRegister(n_qubits)
    cr = qiskit.ClassicalRegister(n_cbits)
    circuit = c2qa.CVCircuit(qmr, qr, cr)
    circuit.initialize([0, 0])

    # ==== Build circuit ====

    # Basic Gaussian Operations on a Resonator
    circuit.cv_bs(phi, qmr[0], qmr[1])
    circuit.cv_d(alpha, qmr[0])
    circuit.cv_r(phi, qmr[0])
    circuit.cv_s(z_a, qmr[0])
    circuit.cv_s2(z_a, qmr[0], qmr[1])

    # Hybrid qubit-cavity gates
    circuit.cv_cnd_d(alpha, beta, qr[0], qmr[0], qmr[1])
    circuit.cv_cnd_s(z_a, z_b, qr[0], qmr[0], qmr[1])

    # ==== Compilation =====

    backend = qiskit.Aer.get_backend('statevector_simulator')
    job = qiskit.execute(circuit, backend)
    result = job.result()
    state = result.get_statevector(circuit)

    # ==== Tests ====

    assert(result.success)
    print(state)
