import qiskit


def test_qiskit():
    """ Verify we can do a simple QisKit circuit without our custom gates. """
    qr = qiskit.QuantumRegister(6)
    cr = qiskit.ClassicalRegister(6)
    circuit = qiskit.circuit.QuantumCircuit(qr, cr)
    circuit.cx(qr[0:1], qr[2])

    backend = qiskit.Aer.get_backend("statevector_simulator")
    job = qiskit.execute(circuit, backend)
    result = job.result()
    state = result.get_statevector(circuit)

    assert result.success
    print(state)
