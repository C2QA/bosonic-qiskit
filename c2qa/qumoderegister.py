from qiskit import QuantumRegister


class QumodeRegister:
    def __init__(self, num_qumodes: int, num_qubits_per_mode: int = 2, name: str = None):
        self.size = num_qumodes * num_qubits_per_mode
        self.num_qumodes = num_qumodes
        self.num_qubits_per_mode = num_qubits_per_mode
        self.cutoff = 2**self.num_qubits_per_mode

        # Aggregate the QuantumRegister representing these qumodes as
        # extending the class confuses QisKit when overriding __getitem__().
        # It doesn't expect a list of Qubit back when indexing a single value 
        # (i.e., qmr[0] is represented by multiple qubits).
        self.qreg = QuantumRegister(size=self.size, name=name)


    def __getitem__(self, key: int):
        start = self.num_qubits_per_mode * key
        stop = start + self.num_qubits_per_mode

        return self.qreg[start:stop]
