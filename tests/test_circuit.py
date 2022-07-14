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
    qmr = c2qa.QumodeRegister(1, 1)
    qbr = qiskit.QuantumRegister(1)
    cbr = qiskit.ClassicalRegister(1)
    circuit = c2qa.CVCircuit(qmr, qbr, cbr)
    circuit.initialize(numpy.array([0,1]), qbr[0])

    state, result = c2qa.util.simulate(circuit)
    assert result.success
