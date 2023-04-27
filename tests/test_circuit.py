import c2qa
import json
import numpy
import pytest
import qiskit
from qiskit.providers.ibmq.runtime.utils import RuntimeEncoder


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

    qmr = c2qa.QumodeRegister(
        num_qumodes=number_of_modes, num_qubits_per_qumode=number_of_qubits_per_mode
    )
    qbr = qiskit.QuantumRegister(size=number_of_qubits)
    cbr = qiskit.ClassicalRegister(size=1)
    circuit = c2qa.CVCircuit(qmr, qbr, cbr)

    sm = [0, 0, 1, 0, 0]
    for i in range(qmr.num_qumodes):
        circuit.cv_initialize(sm[i], qmr[i])

    circuit.initialize(numpy.array([0, 1]), qbr[0])

    state, result = c2qa.util.simulate(circuit)
    assert result.success


def test_with_delay(capsys):
    with capsys.disabled():
        number_of_modes = 1
        number_of_qubits = 1
        number_of_qubits_per_mode = 2

        qmr = c2qa.QumodeRegister(
            num_qumodes=number_of_modes, num_qubits_per_qumode=number_of_qubits_per_mode
        )
        qbr = qiskit.QuantumRegister(size=number_of_qubits)
        circuit = c2qa.CVCircuit(qmr, qbr)

        circuit.delay(100)
        circuit.cv_d(1, qmr[0])

        state, result = c2qa.util.simulate(circuit)
        assert result.success


def test_get_qubit_indices(capsys):
    with capsys.disabled():
        number_of_modes = 2
        number_of_qubits = 2
        number_of_qubits_per_mode = 2

        qmr = c2qa.QumodeRegister(
            num_qumodes=number_of_modes, num_qubits_per_qumode=number_of_qubits_per_mode
        )
        qbr = qiskit.QuantumRegister(size=number_of_qubits)
        cbr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qbr, cbr)  

        indices = circuit.get_qubit_indices([qmr[1]])
        print(f"qmr[1] indices = {indices}")
        assert indices == [2, 3]

        indices = circuit.get_qubit_indices([qmr[0]])
        print(f"qmr[0] indices = {indices}")
        assert indices == [0, 1]

        indices = circuit.get_qubit_indices([qbr[1]])
        print(f"qbr[1] indices = {indices}")
        assert indices == [5]

        indices = circuit.get_qubit_indices([qbr[0]])
        print(f"qbr[0] indices = {indices}")
        assert indices == [4]

def test_initialize_qubit_values(capsys):
    with capsys.disabled():
        print()

        number_of_modes = 1
        number_of_qubits_per_mode = 4

        for fock in range(pow(2, number_of_qubits_per_mode)):
            qmr = c2qa.QumodeRegister(
                num_qumodes=number_of_modes, num_qubits_per_qumode=number_of_qubits_per_mode
            )
            circuit = c2qa.CVCircuit(qmr)
            circuit.cv_initialize(fock, qmr[0])

            state, result = c2qa.util.simulate(circuit)
            assert result.success

            print(f"fock {fock} qubits {list(result.get_counts().keys())[0]}")

def test_serialize(capsys):
    with capsys.disabled():
        print()

        from qiskit.providers.ibmq.runtime.utils import RuntimeEncoder

        qumodes = c2qa.QumodeRegister(2)
        bosonic_circuit = c2qa.CVCircuit(qumodes)

        init_state = [0,2]
        for i in range(qumodes.num_qumodes):
            bosonic_circuit.cv_initialize(init_state[i], qumodes[i])

        phi = qiskit.circuit.Parameter('phi')
        bosonic_circuit.cv_bs(phi, qumodes[0], qumodes[1])

        #print(bosonic_circuit.draw())
        print('\nAttempt to serialize an unbound CVCircuit:')
        bosonic_serial = json.dumps(bosonic_circuit, cls=RuntimeEncoder)
        print(bosonic_serial)
