# Bosonic Qiskit

NQI C2QA project to simulate hybrid boson-qubit systems within QisKit.

## Installation

Bosonic-qiskit can be installed from PyPI:

```bash
pip install bosonic-qiskit
```

We recommend the use of a virtual environment.

### Development

First, checkout the code from Github and use the provided script to create a virtual environment with the necessary dependencies:

```bash
git clone https://github.com/C2QA/bosonic-qiskit.git
cd bosonic-qiskit
./install-dependencies.sh
```

Then, install development dependencies and bosonic-qiskit in editable mode (after activating the newly-created virtual environment)

```bash
source venv/bin/activate
pip install -r requirements_dev.txt
pip install -e .
```

### Dependency Version Compatibility

The Bosonic Qiskit software has not been extensively tested with different versions of its [dependencies](requirements.txt); however, some success has been achieved with both newer and older versions of Qiskit. Do note that some features require newer versions. For example, the noise modelling requires Qiskit v0.34.2+. Using older versions will cause `ModuleNotFoundError` at runtime.

## Tutorials

Jupyter Notebook tutorials can be found in the [tutorials](tutorials) folder. JupyterLab is a dependency found in [requirements.txt](requirements.txt), so after installing and activating the virtual environment, to run the tutorials simply start Jupyter with `jupyter lab` and then navigate to the desired tutorial.

See our paper presented at IEEE HPEC 2022 on [arXiv](https://arxiv.org/abs/2209.11153) for more information on using bosonic-qiskit.

## How to add gates

The code is structured to separate generation of the operator matrices from creating instances of QisKit Gate. 

The first step in adding a new gate is to develop software to build a unitary operator matrix. These matrices must be unitary in order for QisKit to simulate them. Non unitary matrices will fail during simulation. Existing operator matrices are built in the CVOperators class found in [operators.py](c2qa/operators.py). Included in CVOperators are functions to build the bosonic creation and annihilation operators based on a provided cutoff. The order of the data in your operators must match the order of the qumodes (QisKit qubits) sent in as QisKit gate parameters found in [circuit.py](c2qa/circuit.py), as described next.

Once you've written software to build the operator matrix, a new function is added to the CVCircuit class found in [circuit.py](c2qa/circuit.py). This class extends the QisKit QuantumCircuit class to add the bosonic gates available in this library. The previusly defined operators are parameterized by user input, as needed, and appended to the QuantumCircuit as unitary gates. The CVCircuit class includes functions to easily make your new gates conditional based on a control qubit.

See examples of software building new gates in the previously mentioned [operators.py](c2qa/operators.py) and [circuit.py](c2qa/circuit.py). Examples using the library's gates can be found in both the PyTest [test cases](tests) and Jupyter Notebook [tutorials](tutorials) folders

## Available Gates

Current gates available for simulation are documented at https://c2qa.github.io/bosonic-qiskit/c2qa.circuit.CVCircuit.html
