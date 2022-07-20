from qiskit import QuantumRegister


class QumodeRegister:
    """Wrapper to QisKit QuantumRegister to represent multiple qubits per qumode.

    Implements __getitem__ to make QumodeRegister appear to work just like QuantumRegister with instances of CVCircuit.
    """

    def __init__(
        self, num_qumodes: int, num_qubits_per_qumode: int = 2, name: str = None
    ):
        """Initialize QumodeRegister

        Args:
            num_qumodes (int): total number of qumodes
            num_qubits_per_qumode (int, optional): Number of qubits representing each qumode. Defaults to 2.
            name (str, optional): Name of register. Defaults to None.
        """
        self.size = num_qumodes * num_qubits_per_qumode
        self.num_qumodes = num_qumodes
        self.num_qubits_per_qumode = num_qubits_per_qumode
        self.cutoff = 2 ** self.num_qubits_per_qumode

        # Aggregate the QuantumRegister representing these qumodes as
        # extending the class confuses QisKit when overriding __getitem__().
        # It doesn't expect a list of Qubit back when indexing a single value
        # (i.e., qmr[0] is represented by multiple qubits).
        self.qreg = QuantumRegister(size=self.size, name=name)

    def __iter__(self):
        """Iterate over the list of lists representing the qubits for each qumode in the register"""
        return QumodeIterator(self)

    def __getitem__(self, key):
        """Return a list of QisKit Qubit for each indexed qumode

        Args:
            key (slice or int): index into qumode register

        Raises:
            ValueError: if slice or int not provided

        Returns:
            list: ;ost pf qubits from QuantumRegister representing qumode
        """
        start = None
        stop = self.size
        step = None

        if isinstance(key, slice):
            start_index = key.start if key.start else 0
            stop_index = key.stop if key.stop else self.size
            start = self.num_qubits_per_qumode * start_index
            stop = (self.num_qubits_per_qumode * stop_index) + self.num_qubits_per_qumode
            step = (key.step * self.num_qubits_per_qumode) if key.step else None
        elif isinstance(key, int):
            start = self.num_qubits_per_qumode * key
            stop = start + self.num_qubits_per_qumode
        else:
            raise ValueError("Must provide slice or int.")

        return self.qreg[start:stop:step]
    
    def __len__(self):
        """The length of a QumodeRegister is the number of qumodes (not the num_qumodes * num_qubits_per_qumode)"""
        return self.num_qumodes


class QumodeIterator:
    """Iterate over the list of lists representing the qubits for each qumode in the register"""

    def __init__(self, register: QumodeRegister):
        self._index = 0
        self._register = register

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < self._register.num_qumodes:
            next = self._register[self._index]
            self._index += 1

            return next
        else:
            raise StopIteration