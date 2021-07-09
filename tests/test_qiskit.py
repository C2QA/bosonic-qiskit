import qiskit
from qiskit import Aer


def test_qiskit():
    """Verify we can do a simple QisKit circuit without our custom gates."""
    qr = qiskit.QuantumRegister(6)
    cr = qiskit.ClassicalRegister(6)
    circuit = qiskit.circuit.QuantumCircuit(qr, cr)
    circuit.cx(qr[0:1], qr[2])
    circuit.save_statevector()

    backend = Aer.get_backend("aer_simulator")
    job = backend.run(circuit)
    result = job.result()
    state = result.get_statevector(circuit)

    assert result.success
    print(state)
