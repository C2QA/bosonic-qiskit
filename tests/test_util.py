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
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_mode=2)
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr)

        # qr[0] and cr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

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

def test_plot_projection(capsys):
    with capsys.disabled():
        qmr = c2qa.QumodeRegister(num_qumodes=2, num_qubits_per_mode=4)
        qr = qiskit.QuantumRegister(size=1)
        cr = qiskit.ClassicalRegister(size=1)
        circuit = c2qa.CVCircuit(qmr, qr, cr)

        dist = numpy.sqrt(numpy.pi) / numpy.sqrt(2)

        # qr[0] and cr[0] will init to zero
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])

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
