import c2qa
import numpy
import pytest
import qiskit
from qiskit.quantum_info import Statevector



def test_partial_trace_zero(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=2)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        circuit.initialize([1, 0], qr[0])
        circuit.cv_initialize(0, qmr[0])

        state = qiskit.quantum_info.Statevector.from_instruction(circuit)
        trace = c2qa.util.cv_partial_trace(circuit, state)
        assert trace

        # print("Partial trace Fock state zero")
        # print(state)
        # print(state.data)
        # print(trace)


def test_partial_trace_one(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=2)
        qr = qiskit.QuantumRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        circuit.initialize([1, 0], qr[0])
        circuit.cv_initialize(1, qmr[0])

        state = qiskit.quantum_info.Statevector.from_instruction(circuit)
        trace = c2qa.util.cv_partial_trace(circuit, state)
        assert trace

        # print("Partial trace Fock state one")
        # print(state)
        # print(state.data)
        # print(trace)


def test_plot_zero(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=5)
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr)

        # dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)
        dist = 0.5

        # qr[0] and cr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        # circuit.h(qr[0])
        # circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])
        circuit.cv_h()
        circuit.cv_d(dist, qmr[0])

        state = Statevector.from_instruction(circuit)
        # print("Qumode initialized to zero:")
        # print(state)
        c2qa.util.plot_wigner_fock_state(circuit, state, file="tests/zero.png")


def test_plot_one(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=2)
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr)

        # qr[0] and cr[0] will init to zero
        circuit.cv_initialize(1, qmr[0])

        state = Statevector.from_instruction(circuit)
        # print("Qumode initialized to one:")
        # print(state)
        c2qa.util.plot_wigner_fock_state(circuit, state, file="tests/one.png")

# @pytest.mark.skip(reason="Work in progress, not operational yet.")
def test_plot_projection(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=5)
        qr = qiskit.QuantumRegister(size=1)
        # cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        # dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)
        dist = 0.5

        # qr[0] and cr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        # circuit.h(qr[0])
        # circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])
        circuit.cv_d(dist, qmr[0])

        state = Statevector.from_instruction(circuit)

        c2qa.util.plot_wigner_interference(
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






# pauli z average
def test_pauli(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=5)
        qr = qiskit.QuantumRegister(size=1)
        # cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr)

        # dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)
        dist = 0.5

        # qr[0] and cr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        # circuit.h(qr[0])
        # circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])
        circuit.cv_d(dist, qmr[0])

        state = Statevector.from_instruction(circuit)

        # TODO make sure we get a copy so z doesn't get in there
        circuitx = circuit

        circuit.z(qr[0])
        state_p = Statevector.from_instruction(circuit)

        proj = c2qa.util.cv_partial_trace(circuit, state)

        proj_p = c2qa.util.cv_partial_trace(circuit, state_p)

        # |0X0| is +
        # |1X1| is -
        proj_avg = (proj - proj_p) / 2


        circuitx.x(qr[0])
        state_p = Statevector.from_instruction(circuitx)

        proj = c2qa.util.cv_partial_trace(circuitx, state)

        proj_p = c2qa.util.cv_partial_trace(circuitx, state_p)

        # |+X+| is +
        # |-X-| is -
        proj_avg = (proj - proj_p) / 2




        c2qa.util.plot_wigner_fock_state(
            circuit, proj_avg, trace = False, file="tests/pauli.png"
        )