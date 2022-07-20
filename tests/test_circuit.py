import c2qa
import pytest
import qiskit
import numpy


def test_no_registers():
    with pytest.raises(ValueError):
        c2qa.CVCircuit()


def test_only_quantumregister():
    with pytest.raises(ValueError):
        qr = qiskit.QuantumRegister(1)
        c2qa.CVCircuit(qr)


def test_only_qumoderegister():
    c2qa.CVCircuit(c2qa.QumodeRegister(1, 1))


def test_multiple_qumoderegisters():
    with pytest.warns(UserWarning):
        c2qa.CVCircuit(c2qa.QumodeRegister(1, 1), c2qa.QumodeRegister(1, 1))


def test_correct():
    c2qa.CVCircuit(qiskit.QuantumRegister(1), c2qa.QumodeRegister(1, 1))


def test_with_classical():
    c2qa.CVCircuit(
        qiskit.QuantumRegister(1),
        c2qa.QumodeRegister(1, 1),
        qiskit.ClassicalRegister(1),
    )

def test_with_initialize():
    number_of_modes = 5
    number_of_qubits = number_of_modes
    number_of_qubits_per_mode = 2
    cutoff = 2 ** number_of_qubits_per_mode

    qmr = c2qa.QumodeRegister(num_qumodes=number_of_modes, num_qubits_per_qumode=number_of_qubits_per_mode)
    qbr = qiskit.QuantumRegister(size=number_of_qubits)
    cbr = qiskit.ClassicalRegister(size=1)
    circuit = c2qa.CVCircuit(qmr, qbr, cbr)

    sm = [0,0,1,0,0]
    for i in range(qmr.num_qumodes):
        circuit.cv_initialize(sm[i], qmr[i])

    circuit.initialize(numpy.array([0,1]), qbr[0])

    state, result = c2qa.util.simulate(circuit)
    assert result.success
