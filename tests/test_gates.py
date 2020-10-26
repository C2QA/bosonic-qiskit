
import c2qa
import numpy
import qiskit

def test_qiskit():
    qr = qiskit.QuantumRegister(6)
    cr = qiskit.ClassicalRegister(6)
    circuit = qiskit.circuit.QuantumCircuit(qr, cr) 
    circuit.cx(qr[0:1], qr[2])

    backend = qiskit.Aer.get_backend('statevector_simulator')
    job = qiskit.execute(circuit, backend)
    result = job.result()
    state = result.get_statevector(circuit)

    assert(True)
    print(state)

def test_gates():
    # ===== Constants =====

    n_qubits_per_mode = 3
    n_qumodes = 2
    n_qubits = 3
    alpha = 1
    phi = numpy.pi/2
    z = 1

    # ==== Initialize circuit =====
    
    qmr = c2qa.qumoderegister.QumodeRegister(num_qumodes = n_qumodes, num_qubits_per_mode = n_qubits_per_mode)
    qr = qiskit.QuantumRegister(size = n_qubits)
    cr = qiskit.ClassicalRegister(qmr.size)
    circuit = c2qa.circuit.CVCircuit(qmr, qr, cr)
    circuit.initialize([0, 0])

    # ==== Build circuit ====

    circuit.cv_bs(phi, qmr[0], qmr[1])
    circuit.cv_d(alpha, qmr[0])
    circuit.cv_r(phi, qmr[0])
    circuit.cv_s(z, qmr[0])
    circuit.cv_s2(z, qmr[0], qmr[1])
    print(circuit)

    # ==== Compilation =====

    backend = qiskit.Aer.get_backend('statevector_simulator')
    job = qiskit.execute(circuit, backend)
    result = job.result()
    state = result.get_statevector(circuit)

    # ==== Tests ====

    assert(True)
    print(state)