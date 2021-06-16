from qiskit.quantum_info.states.densitymatrix import DensityMatrix
import c2qa
import matplotlib.pyplot as plt
import numpy
from pathlib import Path
import pytest
import scipy.special as ssp
import qiskit
from qiskit.visualization import plot_state_city, plot_histogram
from qiskit.providers.aer.library.save_instructions.save_statevector import save_statevector

def test_partial_trace_zero(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=2)
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
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=2)
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


def test_plot_zero(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=4)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        # dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)
        dist = 1.0

        # qr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])

        state, _ = c2qa.util.simulate(circuit)
        trace = c2qa.util.cv_partial_trace(circuit, state)
        c2qa.util.plot_wigner_fock_state(circuit, trace, file="tests/zero.png", trace=False)


def test_plot_one(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=2)
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr)

        # qr[0] and cr[0] will init to zero
        circuit.cv_initialize(1, qmr[0])

        state, _ = c2qa.util.simulate(circuit)
        # print("Qumode initialized to one:")
        # print(state)
        c2qa.util.plot_wigner_fock_state(circuit, state, file="tests/one.png")


@pytest.mark.skip(reason="Work in progress, not operational yet.")
def test_plot_projection_old(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=5)
        qr = qiskit.QuantumRegister(size=1)
        # cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        # dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)
        dist = 1

        # qr[0] and cr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])
        # circuit.cv_d(dist, qmr[0])

        state, _ = c2qa.util.simulate(circuit)

        c2qa.util.plot_wigner_interference_old(
            circuit, state, file="tests/projection.png"
        )


@pytest.mark.skip(reason="GitHub actions build environments do not have ffmpeg")
def test_animate(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=4)
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr, animation_segments=10)

        dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)

        circuit.initialize([1, 0], qr[0])
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])
        circuit.cv_d(1j * dist, qmr[0])
        circuit.cv_cnd_d(-dist, dist, qr[0], qmr[0])
        circuit.cv_d(-1j * dist, qmr[0])
        circuit.h(qr[0])
        circuit.measure(qr[0], cr[0])

        backend = qiskit.Aer.get_backend("statevector_simulator")
        job = qiskit.execute(circuit, backend)
        result = job.result()

        c2qa.util.animate_wigner_fock_state(
            circuit, result, file="tests/displacement.mp4"
        )


def test_plot_wigner_interference_manual(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=4)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        # dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)
        dist = 1.0

        # qr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])

        state, _ = c2qa.util.simulate(circuit)
        trace = c2qa.util.cv_partial_trace(circuit, state)
        state_h = state.data.conjugate().transpose()

        # Duplicate circuit for Pauli Z
        qmr_z = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=4)
        qr_z = qiskit.QuantumRegister(size=1)
        circuit_z = c2qa.CVCircuit(qmr_z, qr_z)

        # qr[0] will init to zero
        circuit_z.cv_initialize(0, qmr_z[0])

        circuit_z.h(qr_z[0])
        circuit_z.cv_cnd_d(dist, -dist, qr_z[0], qmr_z[0])
        circuit_z.z(qr_z[0])

        state_z, _ = c2qa.util.simulate(circuit_z)
        temp_z = state_z.data * state_h
        trace_z = c2qa.util.cv_partial_trace(circuit_z, temp_z)

        # print("state")
        # print(DensityMatrix(state).data)
        # print("state_z")
        # print(DensityMatrix(state_z).data)
        # numpy.testing.assert_almost_equal(state.data, state_z.data)

        # print("trace")
        # print(trace.data)
        # print("trace_z")
        # print(trace_z.data)
        # numpy.testing.assert_almost_equal(trace.data, trace_z.data)

        projection_zero = (trace.data + trace_z.data) / 2
        projection_one = (trace.data - trace_z.data) / 2

        # Duplicate circuit for Pauli X
        qmr_x = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=4)
        qr_x = qiskit.QuantumRegister(size=1)
        circuit_x = c2qa.CVCircuit(qmr_x, qr_x)

        # qr[0] will init to zero
        circuit_x.cv_initialize(0, qmr_x[0])

        circuit_x.h(qr_x[0])
        circuit_x.cv_cnd_d(dist, -dist, qr_x[0], qmr_x[0])
        circuit_x.x(qr_x[0])

        state_x, _ = c2qa.util.simulate(circuit_x)
        temp_x = state_x.data * state_h
        trace_x = c2qa.util.cv_partial_trace(circuit_x, temp_x)

        projection_plus = (trace.data + trace_x.data) / 2
        projection_minus = (trace.data - trace_x.data) / 2

        # Calculate Wigner functions
        xvec = numpy.linspace(-5, 5, 200)
        wigner_zero = c2qa.util._wigner(projection_zero, xvec, xvec, circuit.cutoff)
        wigner_one = c2qa.util._wigner(projection_one, xvec, xvec, circuit.cutoff)
        wigner_plus = c2qa.util._wigner(projection_plus, xvec, xvec, circuit.cutoff)
        wigner_minus = c2qa.util._wigner(projection_minus, xvec, xvec, circuit.cutoff)

        # Plot using matplotlib on four subplots, at double the default width & height
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12.8, 12.8))

        cont = ax0.contourf(xvec, xvec, wigner_zero, 200, cmap='RdBu_r')
        ax0.set_xlabel("x")
        ax0.set_ylabel("p")
        ax0.set_title("Projection onto zero")
        fig.colorbar(cont, ax=ax0)

        cont = ax1.contourf(xvec, xvec, wigner_one, 200, cmap='RdBu_r')
        ax1.set_xlabel("x")
        ax1.set_ylabel("p")
        ax1.set_title("Projection onto one")
        fig.colorbar(cont, ax=ax1)

        cont = ax2.contourf(xvec, xvec, wigner_plus, 200, cmap='RdBu_r')
        ax2.set_xlabel("x")
        ax2.set_ylabel("p")
        ax2.set_title("Projection onto plus")
        fig.colorbar(cont, ax=ax2)

        cont = ax3.contourf(xvec, xvec, wigner_minus, 200, cmap='RdBu_r')
        ax3.set_xlabel("x")
        ax3.set_ylabel("p")
        ax3.set_title("Projection onto minus")
        fig.colorbar(cont, ax=ax3)

        plt.savefig("tests/interference_copy.png")

        # c2qa.util.plot_wigner_interference(circuit, qr[0], file="tests/interference.png")


def test_plot_wigner_interference(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=4)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        # dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)
        dist = 1.0

        # qr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])
        # circuit.cv_d(dist, qmr[0])

        c2qa.util.plot_wigner_interference(circuit, qr[0], file="tests/interference.png")


def _generate_cat(parity: int, cutoff: int):
        # Prepare cat state
        #   See https://github.com/XanaduAI/strawberryfields/blob/896557c42f6ab07efed79fa402503628bd75bb23/strawberryfields/ops.py#L861-L884
        alpha = numpy.sqrt(3)

        # Zero is even cat state, 1 is odd
        phi = numpy.pi * parity
        l = numpy.arange(cutoff)[:, numpy.newaxis]

        # normalization constant
        temp = numpy.exp(-0.5 * numpy.abs(alpha) ** 2)
        N = temp / numpy.sqrt(2 * (1 + numpy.cos(phi) * temp ** 4))

        # coherent states
        # Need to cast  alpha to float before exponentiation to avoid overflow
        c1 = ((1.0 * alpha) ** l) / numpy.sqrt(ssp.factorial(l))
        c2 = ((-1.0 * alpha) ** l) / numpy.sqrt(ssp.factorial(l))
        # add them up with a relative phase
        ket = (c1 + numpy.exp(1j * phi) * c2) * N

        # in order to support broadcasting, the batch axis has been located at last axis, but backend expects it up as first axis
        ket = numpy.transpose(ket)

        # drop dummy batch axis if it is not necessary
        ket = numpy.squeeze(ket)

        # Create QisKit Statevector from cat state
        return qiskit.quantum_info.Statevector(ket)


def test_wigner_cat_state(capsys):
    with capsys.disabled():
        cutoff = 2 ** 4

        even_filename = "tests/wigner_even.png"
        state_even = _generate_cat(0, cutoff)
        dense_even = DensityMatrix(state_even)
        c2qa.util.plot_wigner(state_even, cutoff, file=even_filename)
        plot_state_city(state_even, figsize=(9, 7)).savefig("tests/plot_state_even.png")
        assert Path(even_filename).is_file()

        odd_filename = "tests/wigner_odd.png"
        state_odd = _generate_cat(1, cutoff)
        c2qa.util.plot_wigner(state_odd, cutoff, file=odd_filename)
        assert Path(odd_filename).is_file()


def test_simulate_plot(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=3)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        # dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)
        dist = 1.0

        # qr[0] will init to zero
        # circuit.cv_initialize(0, qmr[0])
        # circuit.initialize([0,1], qr[0])
        circuit.initialize([0,1], qmr[0][0])

        # circuit.h(qr[0])
        # circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])

        state, result = c2qa.util.simulate(circuit)

        print(state)
        plot_state_city(state).savefig("tests/plot_state_city.png")
        plot_histogram(result.get_counts(), figsize=(9, 7)).savefig("tests/plot_histogram.png")

def test_circuit_cv_cat_state(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=4)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        dist = numpy.sqrt(3)

        # qr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])

        state, result = c2qa.util.simulate(circuit, shots=8192)

        trace = c2qa.util.cv_partial_trace(circuit, state)

        print(state)
        plot_state_city(state).savefig("tests/plot_state_city.png")
        plot_histogram(result.get_counts(), figsize=(9, 7)).savefig("tests/plot_counts.png")
        plot_histogram(trace.sample_counts(256), figsize=(9, 7)).savefig("tests/plot_trace.png")

        wigner_filename = "tests/wigner_cv_cat.png"
        c2qa.util.plot_wigner_fock_state(circuit, state, file=wigner_filename)
        assert Path(wigner_filename).is_file()


def test_circuit_cv_cat_state_paper(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=4)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        dist = numpy.sqrt(3)

        # qr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        circuit.ry(numpy.pi / 2, qr[0])
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])

        # Fanout CX on qmr
        for qubit in qmr[0]:
            circuit.cx(qr[0], qubit)

        state, result = c2qa.util.simulate(circuit, shots=8192)

        trace = c2qa.util.cv_partial_trace(circuit, state)

        print(state)
        plot_state_city(state).savefig("tests/plot_state_city.png")
        plot_histogram(result.get_counts(), figsize=(9, 7)).savefig("tests/plot_counts.png")
        plot_histogram(trace.sample_counts(256), figsize=(9, 7)).savefig("tests/plot_trace.png")

        wigner_filename = "tests/wigner_cv_cat.png"
        c2qa.util.plot_wigner_fock_state(circuit, state, file=wigner_filename)
        assert Path(wigner_filename).is_file()

def test_circuit_cat_state(capsys):
    with capsys.disabled():
        qr = qiskit.QuantumRegister(size=4)
        circuit = qiskit.circuit.QuantumCircuit(qr)

        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[2], qr[3])

        state, result = c2qa.util.simulate(circuit)
        print(state)
        plot_state_city(state).savefig("tests/plot_state_city.png")
        plot_histogram(result.get_counts(), figsize=(9, 7)).savefig("tests/plot_histogram.png")
        # c2qa.util.plot_wigner(state, 3, file="tests/wigner_cat.png")


def test_measure_all_xyz(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=4)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        # dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)
        dist = numpy.sqrt(2)

        # qr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])

        (state_x, result_x), (state_y, result_y), (state_z, result_z) = c2qa.util.measure_all_xyz(circuit)

        print("state_x.probabilities_dict()")
        print(state_x.probabilities_dict())

        print("result_x.get_counts() calculated probabilities")
        print(c2qa.util.get_probabilities(result_x))

        print("result_x.to_dict()")
        print(result_x.to_dict())

        plot_histogram(result_x.get_counts(), title="X", figsize=(9, 7)).savefig("tests/plot_histogram_x.png")
        plot_histogram(result_y.get_counts(), title="Y", figsize=(9, 7)).savefig("tests/plot_histogram_y.png")
        plot_histogram(result_z.get_counts(), title="Z", figsize=(9, 7)).savefig("tests/plot_histogram_z.png")

def test_repeat_until_success(capsys):
    with capsys.disabled():
        success = False
        num_qubits_per_qumode = 6

        while not success:
            qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=num_qubits_per_qumode)
            qr = qiskit.QuantumRegister(size=1)
            cr = qiskit.ClassicalRegister(size=1)
            circuit = c2qa.CVCircuit(qmr, qr, cr)

            dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)

            circuit.initialize([1, 0], qr[0])
            circuit.cv_initialize(0, qmr[0])

            circuit.h(qr[0])
            circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])
            circuit.h(qr[0])
            save_statevector(circuit)
            circuit.measure(qr[0], cr[0])

            state, result = c2qa.util.simulate(circuit, shots=1, add_save_statevector=False)
            counts = result.get_counts()
            print(counts)
            success = "0" in counts and counts["0"] == 1

            if success:
                qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=num_qubits_per_qumode)
                qr = qiskit.QuantumRegister(size=1)
                cr = qiskit.ClassicalRegister(size=1)
                circuit = c2qa.CVCircuit(qmr, qr, cr)
                circuit.initialize(state)

                wigner_filename = "tests/repeat_wigner.png"
                c2qa.util.plot_wigner_fock_state(circuit, state, file=wigner_filename, trace=True)
                assert Path(wigner_filename).is_file()

                wigner_filename = "tests/repeat_projection_wigner.png"
                c2qa.util.plot_wigner_interference(circuit, qr[0], file=wigner_filename)
                assert Path(wigner_filename).is_file()