import qiskit
from qiskit.ignis.verification.tomography import (
    state_tomography_circuits,
    StateTomographyFitter,
)


def test_state_tomograph_from_counts(capsys):
    """See https://qiskit.org/documentation/tutorials/noise/8_tomography.html"""
    with capsys.disabled():
        print()

        qr = qiskit.QuantumRegister(1)
        circuit = qiskit.circuit.QuantumCircuit(qr)
        circuit.h(qr[0])

        qst_circuit = state_tomography_circuits(circuit, qr)

        job = qiskit.execute(
            qst_circuit, qiskit.Aer.get_backend("qasm_simulator"), shots=1024
        )
        result_dict = job.result().to_dict()
        print(result_dict)

        result = qiskit.result.Result.from_dict(result_dict)
        stf = StateTomographyFitter(result, qst_circuit)
        state = stf.fit()
        print(state)
