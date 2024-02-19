import qiskit
import qiskit_aer


def test_qiskit():
    """Verify we can do a simple QisKit circuit without our custom gates."""
    qr = qiskit.QuantumRegister(6)
    cr = qiskit.ClassicalRegister(6)
    circuit = qiskit.circuit.QuantumCircuit(qr, cr)
    circuit.cx(qr[0:1], qr[2])
    circuit.save_statevector()

    backend = qiskit_aer.AerSimulator()
    job = backend.run(circuit)
    result = job.result()
    state = result.get_statevector(circuit)

    assert result.success
    print(state)


def test_initialize(capsys):
    with capsys.disabled():
        # Successful with Qiskit v0.34.2, raises error with v0.35+
        qr = qiskit.QuantumRegister(1)
        circuit = qiskit.circuit.QuantumCircuit(qr)
        circuit.initialize([0, 1], qr[0])
