import c2qa


def test_qumode_iterate(capsys):
    with capsys.disabled():
        num_qubits_per_qumode = 2
        num_qumodes = 3

        register = c2qa.QumodeRegister(num_qubits_per_qumode=num_qubits_per_qumode, num_qumodes=num_qumodes)
        index = 0

        for qumode in register:
            print(qumode)
            index += 1
            assert len(qumode) == num_qubits_per_qumode

        assert index == num_qumodes
