import time

import c2qa
import pytest
import scipy
import qiskit
import qiskit_aer


@pytest.mark.skip(reason="Debug testing UnitaryGate vs ParameterizedUnitaryGate")
def test_custom_unitary(capsys):
    with capsys.disabled():
        circuit = qiskit.QuantumCircuit(2)

        # Passes
        gate = qiskit.circuit.library.UnitaryGate(_matrix().toarray(), label="foo")

        # Fails with `AerError: unknown instruction: foo`
        # gate = c2qa.parameterized_unitary_gate.ParameterizedUnitaryGate(_matrix, [0,1], 2, [], label="foo")

        print("gate name", gate.name, "label", gate.label)
        circuit.append(gate, [0, 1])

        # Fails with `AerError: unknown instruction: bar_circuit`
        #   This is a copy from ParameterizedUnitaryGate._define(), which in turn was adapted from Qiskit's UnitaryGate.
        # q = qiskit.QuantumRegister(circuit.num_qubits)
        # qc = qiskit.QuantumCircuit(q, name="bar_circuit")
        # rules = [
        #     (qiskit.circuit.library.UnitaryGate(_matrix(), "bar_gate"), [i for i in q], []),
        # ]
        # for instr, qargs, cargs in rules:
        #     qc._append(instr, qargs, cargs)
        # circuit.append(qc, [0, 1])

        start = time.perf_counter()

        # TODO Can we set custom AerSimulator target that includes our own basis_gates?
        #      Would that let us not need to transpile?
        # aerbackend.py lines 469-472
        # if self._target is not None:
        #     aer_circuits, idx_maps = assemble_circuits(circuits, self.configuration().basis_gates)
        # else:
        #     aer_circuits, idx_maps = assemble_circuits(circuits)
        # target = c2qa.BosonicQiskitTarget()
        target = {}

        # aerbackend.py lines 214-217
        # If config has custom instructions add them to
        # basis gates to include them for the qiskit transpiler
        # if hasattr(config, "custom_instructions"):
        #     config.basis_gates = config.basis_gates + config.custom_instructions
        configuration = (
            qiskit_aer.backendconfiguration.AerBackendConfiguration.from_dict(
                qiskit_aer.AerSimulator._DEFAULT_CONFIGURATION
            )
        )
        gate_names = []
        for gate in circuit.data:
            gate_names.append(gate.name)
        print(gate_names)
        configuration.basis_gates.extend(gate_names)

        # configuration = None
        # target = None
        # custom_instructions = []
        simulator = qiskit_aer.AerSimulator(
            configuration=configuration, target=target
        )  # , custom_instructions=custom_instructions)
        # circuit = qiskit.transpile(circuit, backend)
        job = simulator.run(circuit)
        end = time.perf_counter()
        print(f"[test_custom_unitary] runtime {end - start}")

        print(job.done())
        print(job.result())


def _matrix(q1=None, q2=None):
    return scipy.sparse.csr_matrix(
        [[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]]
    )


def test_cvcircuit_wo_transpile(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=6)
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr, force_parameterized_unitary_gate=False)

        dist = 3

        circuit.initialize([1, 0], qr[0])
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_c_d(dist, qmr[0], qr[0])
        circuit.cv_d(1j * dist, qmr[0])
        circuit.cv_c_d(-dist, qmr[0], qr[0])
        circuit.cv_d(-1j * dist, qmr[0])

        start = time.perf_counter()
        simulator = qiskit_aer.AerSimulator()
        # circuit = qiskit.transpile(circuit, backend)
        job = simulator.run(circuit)
        end = time.perf_counter()
        print(f"[test_cvcircuit_wo_transpile] {end - start}")

        assert job.result().success


def test_cvcircuit_util_simulate(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=6)
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr, force_parameterized_unitary_gate=False)

        dist = 3

        circuit.initialize([1, 0], qr[0])
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_c_d(dist, qmr[0], qr[0])
        circuit.cv_d(1j * dist, qmr[0])
        circuit.cv_c_d(-dist, qmr[0], qr[0])
        circuit.cv_d(-1j * dist, qmr[0])

        count = 10
        avg = 0
        for _ in range(count):
            start = time.perf_counter()
            _, result, _ = c2qa.util.simulate(circuit, return_fockcounts=False, add_save_statevector=False)
            end = time.perf_counter()
            print(f"[test_cvcircuit_util_simulate] {end - start}")
            avg += (end - start)
        print(f"[test_cvcircuit_util_simulate] average {avg / count}")

        assert result.success
