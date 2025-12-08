from collections.abc import Sequence

from qiskit.circuit import Qubit

from bosonic_qiskit import QumodeRegister


class TestQumodeRegister:
    def test_qumode_iterate(self, capsys):
        with capsys.disabled():
            num_qubits_per_qumode = 2
            num_qumodes = 3

            register = QumodeRegister(
                num_qubits_per_qumode=num_qubits_per_qumode, num_qumodes=num_qumodes
            )

            assert isinstance(register, Sequence)
            assert len(register) == num_qumodes
            for qumode in register:
                assert len(qumode) == num_qubits_per_qumode

    def test_qubits(self, capsys):
        with capsys.disabled():
            reg = QumodeRegister(2, 2)
            assert len(reg.qubits) == 4

            for qubit in reg.qubits:
                assert isinstance(qubit, Qubit)
                assert qubit in reg

    def test_add(self, capsys):
        with capsys.disabled():
            reg1 = QumodeRegister(2, 2)
            reg2 = QumodeRegister(3, 1)

            all_qumodes = reg1 + reg2
            assert len(all_qumodes) == 5
            assert len(all_qumodes[0]) == 2
            assert len(all_qumodes[1]) == 2
            assert len(all_qumodes[2]) == 1
            assert len(all_qumodes[3]) == 1
            assert len(all_qumodes[4]) == 1

    def test_getitem(self, capsys):
        with capsys.disabled():
            reg = QumodeRegister(3, 2)

            # Implicitly tests getitem with scalar index
            for qumode in reg:
                assert len(qumode) == 2

            # Slicing
            qumodes = reg[:2]
            assert len(qumodes) == 2
            for qumode in qumodes:
                assert len(qumode) == 2

    def test_cutoff(self, capsys):
        with capsys.disabled():
            reg = QumodeRegister(3, 2)
            assert reg.cutoff == 4
