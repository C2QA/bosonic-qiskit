import c2qa
import json
import numpy
import pytest
import qiskit
from qiskit_ibm_runtime.utils import RuntimeEncoder


def test_no_registers():
    with pytest.raises(ValueError):
        c2qa.CVCircuit()


def test_only_quantumregister():
    with pytest.raises(ValueError):
        qr = qiskit.QuantumRegister(1)
        c2qa.CVCircuit(qr)


def test_only_qumoderegister():
    c2qa.CVCircuit(c2qa.QumodeRegister(1, 1))


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

    state, result, _ = c2qa.util.simulate(circuit)
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

        state, result, _ = c2qa.util.simulate(circuit)
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
                num_qumodes=number_of_modes,
                num_qubits_per_qumode=number_of_qubits_per_mode,
            )
            circuit = c2qa.CVCircuit(qmr)
            circuit.cv_initialize(fock, qmr[0])

            state, result, _ = c2qa.util.simulate(circuit)
            assert result.success

            print(f"fock {fock} qubits {list(result.get_counts().keys())[0]}")


def test_serialize(capsys):
    with capsys.disabled():
        print()

        qumodes = c2qa.QumodeRegister(2)
        bosonic_circuit = c2qa.CVCircuit(qumodes)

        init_state = [0, 2]
        for i in range(qumodes.num_qumodes):
            bosonic_circuit.cv_initialize(init_state[i], qumodes[i])

        phi = qiskit.circuit.Parameter("phi")
        bosonic_circuit.cv_bs(phi, qumodes[0], qumodes[1])

        # print(bosonic_circuit.draw())
        print("\nAttempt to serialize an unbound CVCircuit:")
        bosonic_serial = json.dumps(bosonic_circuit, cls=RuntimeEncoder)
        print(bosonic_serial)


def test_cv_gate_from_matrix(capsys):
    with capsys.disabled():
        # This test picks random qumode/qubit to initialize xgate on, does fockcounts, and checks that results match expected values.
        xgate = [
            [0, 1],
            [1, 0],
        ]  # For qubit, flips |1> to |0> and vice versa. For qumode, flips fock |1> to fock |0> and vice versa

        num_qumode_registers = 2
        num_qumodes_per_register = 2
        num_qubits_per_qumode = 1

        num_qubits = 3

        total_qubits = (
            num_qumode_registers * num_qumodes_per_register * num_qubits_per_qumode
            + num_qubits
        )
        for i in range(10):  # Repeat test 10 times
            # Two qumode registers, One quantum register, One classical register. Hilbert space dimension = 2**(2 * 2 * 1 + 3) = 128
            qmr1 = c2qa.QumodeRegister(num_qumodes_per_register, num_qubits_per_qumode)
            qmr2 = c2qa.QumodeRegister(num_qumodes_per_register, num_qubits_per_qumode)
            q = qiskit.QuantumRegister(num_qubits)
            creg = qiskit.ClassicalRegister(total_qubits)

            # Initialize all qumodes to fock |1> and all qubits to |1> state
            circuit = c2qa.CVCircuit(qmr1, qmr2, q, creg)
            circuit.cv_initialize([0, 1], qmr1)
            circuit.cv_initialize([0, 1], qmr2)

            for i in range(num_qubits):
                circuit.initialize([0, 1], q[i])

            # Pick one of the qumode registers to act on
            register = numpy.random.randint(1, 3)

            # Pick one of the qumodes to act on
            qumode_no = numpy.random.randint(0, 2)

            # Pick one of the qubits to act on
            qubit_no = numpy.random.randint(0, 3)

            # Create string corresponding to expected results
            expect = ["1" for _ in range(total_qubits)]

            if register == 1:
                expect[qumode_no] = "0"
            elif register == 2:
                expect[qumode_no + 2] = "0"

            expect[4 + qubit_no] = "0"

            expect = "".join(reversed(expect))

            # Depending on quantum register chosen, assert result to be true
            if register == 1:
                circuit.cv_gate_from_matrix(xgate, qumodes=qmr1[qumode_no])
                circuit.cv_gate_from_matrix(xgate, qubits=q[qubit_no])

                circuit.cv_measure(qmr1[:] + qmr2[:] + q[:], creg)

                _, result, fock_counts = c2qa.util.simulate(circuit)

                # There should only be 1 result
                if len(list(fock_counts.keys())) > 1:
                    raise Exception

                assert list(fock_counts.keys())[0] == expect

            elif register == 2:
                circuit.cv_gate_from_matrix(xgate, qmr2[qumode_no])
                circuit.cv_gate_from_matrix(xgate, qubits=q[qubit_no])

                circuit.cv_measure(qmr1[:] + qmr2[:] + q[:], creg)

                _, result, fock_counts = c2qa.util.simulate(circuit)

                # There should only be 1 result
                if len(list(fock_counts.keys())) > 1:
                    raise Exception

                assert list(fock_counts.keys())[0] == expect

            else:
                raise Exception


def test_discretize_cv_gate_from_matrix(capsys):
    with capsys.disabled():
        # This test picks random qumode/qubit to initialize xgate on, does fockcounts, and checks that results match expected values.
        xgate = [[0, 1], [1, 0]]

        num_qumode_registers = 2
        num_qumodes_per_register = 2
        num_qubits_per_qumode = 1

        num_qubits = 3

        total_qubits = (
            num_qumode_registers * num_qumodes_per_register * num_qubits_per_qumode
            + num_qubits
        )

        # Two qumode registers, One quantum register, One classical register. Hilbert space dimension = 2**(2 * 2 * 1 + 3) = 128
        qmr1 = c2qa.QumodeRegister(num_qumodes_per_register, num_qubits_per_qumode)
        qmr2 = c2qa.QumodeRegister(num_qumodes_per_register, num_qubits_per_qumode)
        q = qiskit.QuantumRegister(num_qubits)
        creg = qiskit.ClassicalRegister(total_qubits)

        # Initialize all qumodes to fock |1> and all qubits to |1> state
        circuit = c2qa.CVCircuit(qmr1, qmr2, q, creg)

        circuit.cv_gate_from_matrix(xgate, qumodes=qmr1[0])

        discretized = c2qa.discretize.discretize_circuits(circuit)
        assert len(circuit.data) == len(discretized)


def test_cv_initialize(capsys):
    with capsys.disabled():
        qmr1 = c2qa.QumodeRegister(1, 2)
        qmr2 = c2qa.QumodeRegister(2, 2)
        qmr3 = c2qa.QumodeRegister(2, 3)  # <----- change to (2, 2) for no error
        qmr4 = c2qa.QumodeRegister(1, 2)
        qmr5 = c2qa.QumodeRegister(3, 2)
        qmr6 = c2qa.QumodeRegister(3, 2)
        circuit = c2qa.CVCircuit(qmr1, qmr2, qmr3, qmr4, qmr5, qmr6)

        circuit.cv_initialize([0, 1], qmr1[0])
        circuit.cv_initialize([0, 1], qmr2[0])
        circuit.cv_initialize([0, 1], qmr3[0])
        circuit.cv_initialize([0, 1], qmr4[0])
        circuit.cv_initialize([0, 1], qmr5[0])
        circuit.cv_initialize([0, 1], qmr6[0])

        # saving a state vector for all the registers takes a considerable amount of time
        state, result, fock_counts = c2qa.util.simulate(
            circuit, add_save_statevector=False, return_fockcounts=False
        )
        assert result.success
