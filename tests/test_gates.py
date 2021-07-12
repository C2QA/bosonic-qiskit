import random

import c2qa
import numpy
import qiskit


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


def create_unconditional(num_qumodes: int = 2, num_qubits_per_qumode: int = 2):
    qmr = c2qa.QumodeRegister(num_qumodes, num_qubits_per_qumode)
    circuit = c2qa.CVCircuit(qmr)
    for qumode in range(num_qumodes):
        circuit.cv_initialize(0, qmr[qumode])

    return circuit, qmr


def assert_changed(state, result):
    assert result.success
    # print()
    # print(circuit.draw("text"))
    # print(state)

    # TODO - better understand what the state vector results should be
    assert count_nonzero(state) > 1


def assert_unchanged(state, result):
    assert result.success
    # print()
    # print(circuit.draw("text"))
    # print(state)

    # TODO - better understand what the state vector results should be
    assert count_nonzero(state) == 1


def test_no_gates():
    circuit, qmr = create_unconditional()
    state, result = c2qa.util.simulate(circuit)
    assert_unchanged(state, result)


def test_beamsplitter_once():
    circuit, qmr = create_unconditional()

    phi = random.random()
    circuit.cv_bs(phi, qmr[0], qmr[1])

    state, result = c2qa.util.simulate(circuit)

    # TODO - Beam splitter gate does not change state vector
    #        Both Strawberry Fields & FockWits are the same, too.
    # assert_changed(state, result)
    assert_unchanged(state, result)


def test_conditional_beamsplitter():
    circuit, qmr, qr = create_conditional()

    phi = random.random()
    chi = random.random()
    circuit.cv_cnd_bs(phi, chi, qr[0], qmr[0], qmr[1])

    state, result = c2qa.util.simulate(circuit)

    # TODO - Beam splitter gate does not change state vector
    #        Both Strawberry Fields & FockWits are the same, too.
    # assert_changed(state, result)
    assert_unchanged(state, result)


def test_beamsplitter_twice():
    circuit, qmr = create_unconditional()

    phi = random.random()
    circuit.cv_bs(phi, qmr[0], qmr[1])
    circuit.cv_bs(-phi, qmr[0], qmr[1])

    state, result = c2qa.util.simulate(circuit)
    assert_unchanged(state, result)


def test_conditonal_displacement():
    circuit, qmr, qr = create_conditional()

    alpha = random.random()
    beta = random.random()
    circuit.cv_cnd_d(alpha, -beta, qr[0], qmr[0])
    circuit.cv_cnd_d(-alpha, beta, qr[0], qmr[0])

    circuit.cv_cnd_d(alpha, -beta, qr[1], qmr[0])
    circuit.cv_cnd_d(-alpha, beta, qr[1], qmr[0])

    state, result = c2qa.util.simulate(circuit)
    assert_unchanged(state, result)


def test_conditonal_squeezing():
    circuit, qmr, qr = create_conditional()

    alpha = random.random()
    beta = random.random()
    circuit.cv_cnd_s(alpha, -beta, qr[0], qmr[0])
    circuit.cv_cnd_s(-alpha, beta, qr[0], qmr[0])

    circuit.cv_cnd_s(alpha, -beta, qr[1], qmr[0])
    circuit.cv_cnd_s(-alpha, beta, qr[1], qmr[0])

    state, result = c2qa.util.simulate(circuit)
    assert_unchanged(state, result)


def test_displacement_once():
    circuit, qmr = create_unconditional()

    alpha = random.random()
    circuit.cv_d(alpha, qmr[0])

    state, result = c2qa.util.simulate(circuit)
    assert_changed(state, result)


def test_displacement_twice():
    circuit, qmr = create_unconditional()

    alpha = random.random()
    circuit.cv_d(alpha, qmr[0])
    circuit.cv_d(-alpha, qmr[0])

    state, result = c2qa.util.simulate(circuit)
    assert_unchanged(state, result)


def test_cond_displacement_gate_vs_two_separate():
    from qiskit.extensions import UnitaryGate

    alpha = numpy.sqrt(numpy.pi)
    beta = -alpha

    # Circuit using cnd_d
    qmr = c2qa.QumodeRegister(1, 2)
    qr = qiskit.QuantumRegister(1)
    cr = qiskit.ClassicalRegister(1)
    circuit = c2qa.CVCircuit(qmr, qr, cr)
    circuit.cv_initialize(0, qmr[0])  # qr[0] and cr[0] will init to zero
    circuit.cv_cnd_d(alpha, beta, qr[0], qmr[0])
    state, result = c2qa.util.simulate(circuit)
    assert result.success
    state_cnd = result.get_statevector(circuit)

    # Circuit using two controlled unitaries
    qmr = c2qa.QumodeRegister(1, 2)
    qr = qiskit.QuantumRegister(1)
    cr = qiskit.ClassicalRegister(1)
    circuit = c2qa.CVCircuit(qmr, qr, cr)
    circuit.cv_initialize(0, qmr[0])  # qr[0] and cr[0] will init to zero
    circuit.append(
        UnitaryGate(circuit.ops.d(alpha).toarray()).control(
            num_ctrl_qubits=1, ctrl_state=0
        ),
        [qr[0]] + qmr[0],
    )
    circuit.append(
        UnitaryGate(circuit.ops.d(beta).toarray()).control(
            num_ctrl_qubits=1, ctrl_state=1
        ),
        [qr[0]] + qmr[0],
    )
    state, result = c2qa.util.simulate(circuit)
    assert result.success
    state_unitary = result.get_statevector(circuit)

    assert numpy.allclose(state_cnd, state_unitary)


def test_displacement_calibration(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(1, 2)
        qr = qiskit.QuantumRegister(1)
        cr = qiskit.ClassicalRegister(1)
        circuit = c2qa.CVCircuit(qmr, qr, cr)

        # qr[0] and cr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        alpha = numpy.sqrt(numpy.pi)

        circuit.h(qr[0])
        circuit.cv_cnd_d(alpha, -alpha, qr[0], qmr[0])
        circuit.cv_d(1j * alpha, qmr[0])
        circuit.cv_cnd_d(-alpha, alpha, qr[0], qmr[0])
        circuit.cv_d(-1j * alpha, qmr[0])
        circuit.h(qr[0])
        circuit.measure(qr[0], cr[0])

        state, result = c2qa.util.simulate(circuit)
        assert result.success

        state = result.get_statevector(circuit)
        counts = result.get_counts(circuit)

        assert len(state) > 0
        assert counts

        # print()
        # print(circuit.draw("text"))
        # print(state)
        # print(counts.int_outcomes())


def test_rotation_once():
    circuit, qmr = create_unconditional()

    theta = random.random()
    circuit.cv_r(theta, qmr[0])

    state, result = c2qa.util.simulate(circuit)

    # TODO - Rotation gate does not change state vector.
    #        Both Strawberry Fields & FockWits are the same, too.
    # assert_changed(state, result)
    assert_unchanged(state, result)


def test_rotation_twice():
    circuit, qmr = create_unconditional()

    theta = random.random()
    circuit.cv_r(theta, qmr[0])
    circuit.cv_r(-theta, qmr[0])

    state, result = c2qa.util.simulate(circuit)
    assert_unchanged(state, result)


def test_squeezing_once():
    circuit, qmr = create_unconditional()

    z = random.random()
    circuit.cv_s(z, qmr[0])

    state, result = c2qa.util.simulate(circuit)
    assert_changed(state, result)


def test_squeezing_twice():
    circuit, qmr = create_unconditional()

    z = random.random()
    circuit.cv_s(z, qmr[0])
    circuit.cv_s(-z, qmr[0])

    state, result = c2qa.util.simulate(circuit)
    assert_unchanged(state, result)


def test_two_mode_squeezing_once():
    circuit, qmr = create_unconditional()

    z = random.random()
    circuit.cv_s2(z, qmr[0], qmr[1])

    state, result = c2qa.util.simulate(circuit)
    assert_changed(state, result)


def test_two_mode_squeezing_twice():
    circuit, qmr = create_unconditional()

    z = random.random()
    circuit.cv_s2(z, qmr[0], qmr[1])
    circuit.cv_s2(-z, qmr[0], qmr[1])

    state, result = c2qa.util.simulate(circuit)
    assert_unchanged(state, result)


def test_gates():
    """Verify that we can use the gates, not that they are actually working."""

    # ===== Constants =====
    alpha = 1
    beta = -1
    phi = numpy.pi / 2
    z_a = 1
    z_b = -1

    circuit, qmr, qr = create_conditional()

    # Basic Gaussian Operations on a Resonator
    circuit.cv_bs(phi, qmr[0], qmr[1])
    circuit.cv_d(alpha, qmr[0])
    circuit.cv_r(phi, qmr[0])
    circuit.cv_s(z_a, qmr[0])
    circuit.cv_s2(z_a, qmr[0], qmr[1])

    # Hybrid qubit-cavity gates
    circuit.cv_cnd_d(alpha, beta, qr[0], qmr[0])
    circuit.cv_cnd_d(alpha, beta, qr[0], qmr[1])
    circuit.cv_cnd_s(z_a, z_b, qr[0], qmr[0])
    circuit.cv_cnd_s(z_a, z_b, qr[0], qmr[1])

    state, result = c2qa.util.simulate(circuit)

    assert result.success
