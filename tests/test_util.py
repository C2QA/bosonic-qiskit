import c2qa
import numpy
from pathlib import Path
import qiskit
from qiskit.visualization import plot_histogram


def test_partial_trace_zero(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=2)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        circuit.initialize([0, 1], qr[0])
        circuit.cv_initialize(0, qmr[0])

        state, _ = c2qa.util.simulate(circuit)
        trace = c2qa.util.cv_partial_trace(circuit, state)

        assert state.dims() == (2, 2, 2)
        assert trace.dims() == (2, 2)
        prob = trace.probabilities_dict()
        numpy.testing.assert_almost_equal(prob["00"], 1.0)

        # print("Partial trace Fock state zero")
        # print(DensityMatrix(state).probabilities_dict())
        # print(trace.probabilities_dict())


def test_partial_trace_one(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=2)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        circuit.initialize([1, 0], qr[0])
        circuit.cv_initialize(1, qmr[0])

        state, _ = c2qa.util.simulate(circuit)
        trace = c2qa.util.cv_partial_trace(circuit, state)

        assert state.dims() == (2, 2, 2)
        assert trace.dims() == (2, 2)
        prob = trace.probabilities_dict()
        numpy.testing.assert_almost_equal(prob["01"], 1.0)

        # print("Partial trace Fock state one")
        # print(DensityMatrix(state).probabilities_dict())
        # print(trace.probabilities_dict())


def test_measure_all_xyz(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=4)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        # dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)
        dist = numpy.sqrt(2)

        # qr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_c_d(dist, qmr[0], qr[0])

        (
            (state_x, result_x),
            (state_y, result_y),
            (state_z, result_z),
        ) = c2qa.util.measure_all_xyz(circuit)

        print("state_x.probabilities_dict()")
        print(state_x.probabilities_dict())

        print("result_x.get_counts() calculated probabilities")
        print(c2qa.util.get_probabilities(result_x))

        print("result_x.to_dict()")
        print(result_x.to_dict())

        plot_histogram(result_x.get_counts(), title="X", figsize=(9, 7)).savefig(
            "tests/plot_histogram_x.png"
        )
        plot_histogram(result_y.get_counts(), title="Y", figsize=(9, 7)).savefig(
            "tests/plot_histogram_y.png"
        )
        plot_histogram(result_z.get_counts(), title="Z", figsize=(9, 7)).savefig(
            "tests/plot_histogram_z.png"
        )


def test_stateread(capsys):
    with capsys.disabled():
        num_qumodes = 2
        qubits_per_mode = 3

        qmr = c2qa.QumodeRegister(
            num_qumodes=num_qumodes, num_qubits_per_qumode=qubits_per_mode, name="qmr"
        )

        circuit = c2qa.CVCircuit(qmr)

        circuit.cv_initialize(2, qmr[0])
        circuit.cv_initialize(0, qmr[1])
        state, result = c2qa.util.simulate(circuit)
        c2qa.util.stateread(
            state,
            numberofqubits=0,
            numberofmodes=num_qumodes,
            cutoff=circuit.cutoff,
            verbose=True,
        )
