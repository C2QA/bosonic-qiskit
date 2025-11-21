from collections.abc import Iterator, Sequence
from typing import overload

from qiskit import QuantumRegister

from .typing import Qubit, Qumode


class QumodeRegister(Sequence[Qumode]):
    """Wrapper to QisKit QuantumRegister to represent multiple qubits per qumode.

    Implements __getitem__ to make QumodeRegister appear to work just like QuantumRegister with instances of CVCircuit.
    """

    def __init__(
        self, num_qumodes: int, num_qubits_per_qumode: int = 2, name: str | None = None
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

        # Aggregate the QuantumRegister representing these qumodes as
        # extending the class confuses QisKit when overriding __getitem__().
        # It doesn't expect a list of Qubit back when indexing a single value
        # (i.e., qmr[0] is represented by multiple qubits).
        self.qreg = QuantumRegister(size=self.size, name=name)

    @property
    def name(self) -> str:
        return self.qreg.name

    @property
    def qubits(self) -> QuantumRegister:
        return self.qreg

    @property
    def cutoff(self) -> int:
        return self.calculate_cutoff(self.num_qubits_per_qumode)

    @staticmethod
    def calculate_cutoff(num_qubits_per_qumode: int) -> int:
        return 2**num_qubits_per_qumode

    def get_qumode_index(self, qubit: Qubit) -> int:
        """Get the qumode index for the given qubit in this register"""
        qubit_index = self.qreg.index(qubit)
        return qubit_index // self.num_qubits_per_qumode

    def __iter__(self) -> Iterator[Qumode]:
        """Iterate over the list of lists representing the qubits for each qumode in the register"""
        return QumodeIterator(self)

    @overload
    def __getitem__(self, key: int) -> Qumode: ...

    @overload
    def __getitem__(self, key: slice) -> list[Qumode]: ...

    def __getitem__(self, key):
        """Return a list of QisKit Qubit for each indexed qumode

        Args:
            key (slice or int): index into qumode register

        Raises:
            ValueError: if slice or int not provided

        Returns:
            list: list pf qubits from QuantumRegister representing qumode
        """
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or self.size
            step = key.step or 1

            return [self[i] for i in range(start, stop, step)]

        if isinstance(key, int):
            start = self.num_qubits_per_qumode * key
            stop = start + self.num_qubits_per_qumode
            return self.qreg[start:stop]

        raise KeyError("Must provide slice or int.")

    def __len__(self):
        """The length of a QumodeRegister is the number of qumodes (not the num_qumodes * num_qubits_per_qumode)"""
        return self.num_qumodes

    def __contains__(self, qubit: Qubit):
        """Return true if this QumodeRegister contains the given qubit. This allows callers to use `in` python syntax."""
        return qubit in self.qreg

    def __add__(self, other: Sequence[Qumode]) -> list[Qumode]:
        return [*self, *other]

    def __repr__(self) -> str:
        return f"QumodeRegister({self.num_qumodes}, {self.num_qubits_per_qumode}, '{self.name}')"


class QumodeIterator(Iterator[Qumode]):
    """Iterate over the list of lists representing the qubits for each qumode in the register"""

    def __init__(self, register: QumodeRegister):
        self._index: int = 0
        self._register = register

    def __iter__(self) -> Iterator[Qumode]:
        return self

    def __next__(self):
        if self._index < self._register.num_qumodes:
            next = self._register[self._index]
            self._index += 1

            return next
        else:
            raise StopIteration
