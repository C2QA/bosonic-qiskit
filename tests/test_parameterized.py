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

        bound_circuit = circuit.bind_parameters({alpha: 3.14})

        state, result = c2qa.util.simulate(bound_circuit)
        assert_changed(state, result)