from pathlib import Path


import c2qa
import numpy
import qiskit


def assert_changed(state, result):
    assert result.success
    # print()
    # print(circuit.draw("text"))
    # print(state)

    # TODO - better understand what the state vector results should be
    assert count_nonzero(state) > 1


def count_nonzero(statevector: qiskit.quantum_info.Statevector):
    """Re-implement numpy.count_nonzero using numpy.isclose()."""
    nonzero = len(statevector.data)
    for state in statevector.data:
        if numpy.isclose(state, 0):
            nonzero -= 1

    return nonzero


def create_conditional(num_qumodes: int = 2, num_qubits_per_qumode: int = 2):
    qmr = c2qa.QumodeRegister(num_qumodes, num_qubits_per_qumode)
    qr = qiskit.QuantumRegister(2)
    circuit = c2qa.CVCircuit(qmr, qr)

    for qumode in range(num_qumodes):
        circuit.cv_initialize(0, qmr[qumode])

    circuit.initialize([0, 1], qr[1])  # qr[0] will init to zero

    return circuit, qmr, qr


def test_parameterized_displacement(capsys):
    with capsys.disabled():
        circuit, qmr, qr = create_conditional()

        alpha = qiskit.circuit.Parameter("alpha")
        circuit.cv_d(alpha, qmr[0])

        bound_circuit = circuit.assign_parameters({alpha: 3.14})

        state, result, fock_counts = c2qa.util.simulate(bound_circuit)
        assert_changed(state, result)


def test_complex_literals(capsys):
    with capsys.disabled():
        # a = qiskit.circuit.Parameter('𝛼')

        qmr = c2qa.QumodeRegister(1, num_qubits_per_qumode=4)
        qbr = qiskit.QuantumRegister(1)

        minimal_circuit = c2qa.CVCircuit(qmr, qbr)

        minimal_circuit.h(qbr[0])

        minimal_circuit.cv_c_d(1j * 1, qmr[0], qbr[0])

        # bound_circuit = minimal_circuit.assign_parameters({a: 1})

        c2qa.util.simulate(minimal_circuit)


def test_complex_parameters(capsys):
    with capsys.disabled():
        a = qiskit.circuit.Parameter("𝛼")

        qmr = c2qa.QumodeRegister(1, num_qubits_per_qumode=4)
        qbr = qiskit.QuantumRegister(1)

        minimal_circuit = c2qa.CVCircuit(qmr, qbr)

        minimal_circuit.h(qbr[0])

        minimal_circuit.cv_c_d(1j * a, qmr[0], qbr[0])

        bound_circuit = minimal_circuit.assign_parameters({a: 1})
        c2qa.util.simulate(bound_circuit)


def test_complex_parameters_float(capsys):
    with capsys.disabled():
        a = qiskit.circuit.Parameter("𝛼")

        qmr = c2qa.QumodeRegister(1, num_qubits_per_qumode=4)
        qbr = qiskit.QuantumRegister(1)

        minimal_circuit = c2qa.CVCircuit(qmr, qbr)

        minimal_circuit.h(qbr[0])

        minimal_circuit.cv_c_d(1j * a, qmr[0], qbr[0])

        bound_circuit = minimal_circuit.assign_parameters({a: 2})
        c2qa.util.simulate(bound_circuit)
