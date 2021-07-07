from qiskit import QuantumRegister


class QumodeRegister:
    def __init__(
        self, num_qumodes: int, num_qubits_per_qumode: int = 2, name: str = None
    ):
        self.size = num_qumodes * num_qubits_per_qumode
        self.num_qumodes = num_qumodes
        self.num_qubits_per_qumode = num_qubits_per_qumode
        self.cutoff = 2 ** self.num_qubits_per_qumode

        # Aggregate the QuantumRegister representing these qumodes as
        # extending the class confuses QisKit when overriding __getitem__().
        # It doesn't expect a list of Qubit back when indexing a single value
        # (i.e., qmr[0] is represented by multiple qubits).
        self.qreg = QuantumRegister(size=self.size, name=name)

    def __getitem__(self, key):
        start = None
        stop = self.size
        step = None

        if isinstance(key, slice):
            start = self.num_qubits_per_qumode * key.start
            stop = (self.num_qubits_per_qumode * key.stop) + self.num_qubits_per_qumode
            step = (key.step * self.num_qubits_per_qumode) if key.step else None
        elif isinstance(key, int):
            start = self.num_qubits_per_qumode * key
            stop = start + self.num_qubits_per_qumode
        else:
            raise ValueError("Must provide slice or int.")

        return self.qreg[start:stop:step]
