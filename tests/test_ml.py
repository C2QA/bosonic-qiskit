#importing
import os
import sys
import c2qa
import qiskit
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import pytest
# from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import algorithm_globals

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN


def test_bosnic_qiskit(capsys):
    with capsys.disabled():
        algorithm_globals.random_seed = 42

        #Plotting the sine function 
        num_samples = 20
        eps = 0.2
        lb, ub = -np.pi, np.pi
        X_ = np.linspace(lb, ub, num=50).reshape(50, 1)
        f = lambda x: np.sin(x)

        X = (ub - lb) * algorithm_globals.random.random([num_samples, 1]) + lb
        y = f(X[:, 0]) + eps * (2 * algorithm_globals.random.random(num_samples) - 1)

        plt.plot(X_, f(X_), "r--")
        plt.plot(X, y, "bo")
        # plt.show()

        # construct simple feature map
        param_x = Parameter("x")
        feature_qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=1)
        feature_map = c2qa.CVCircuit(feature_qmr, name="fm")
        # feature_map.cv_d(param_x, feature_qmr[0])
        feature_map.cv_r(param_x, feature_qmr[0])

        # construct simple ansatz
        param_y = Parameter("y")
        ansatz_qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=1)
        ansatz = c2qa.CVCircuit(ansatz_qmr, name="vf")
        # ansatz.cv_d(param_y, ansatz_qmr[0])
        ansatz.cv_r(param_y, ansatz_qmr[0])

        # construct a circuit
        qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=1)
        qc = c2qa.CVCircuit(qmr)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        # qc.draw()

        # construct QNN
        regression_estimator_qnn = EstimatorQNN(
            circuit=qc, input_params=feature_map.parameters, weight_params=ansatz.parameters
        )

        # callback function that draws a live plot when the .fit() method is called
        def callback_graph(weights, obj_func_eval):
            # clear_output(wait=True)
            objective_func_vals.append(obj_func_eval)
            plt.title("Objective function value against iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Objective function value")
            plt.plot(range(len(objective_func_vals)), objective_func_vals)
            # plt.show()


        # construct the regressor from the neural network
        regressor = NeuralNetworkRegressor(
            neural_network=regression_estimator_qnn,
            loss="squared_error",
            optimizer=L_BFGS_B(maxiter=5),
            callback=callback_graph,
        )


        # create empty array for callback to store evaluations of the objective function
        objective_func_vals = []
        plt.rcParams["figure.figsize"] = (12, 6)

        # fit to data
        regressor.fit(X, y)

        # return to default figsize
        plt.rcParams["figure.figsize"] = (6, 4)

        # score the result
        regressor.score(X, y)


def test_qiskit(capsys):
    """Duplicate test from Qiskit, without bosonic-qiskit https://qiskit.org/documentation/machine-learning/tutorials/02_neural_network_classifier_and_regressor.html#Regression"""
    with capsys.disabled():
        num_samples = 20
        eps = 0.2
        lb, ub = -np.pi, np.pi
        X_ = np.linspace(lb, ub, num=50).reshape(50, 1)
        f = lambda x: np.sin(x)

        X = (ub - lb) * algorithm_globals.random.random([num_samples, 1]) + lb
        y = f(X[:, 0]) + eps * (2 * algorithm_globals.random.random(num_samples) - 1)

        plt.plot(X_, f(X_), "r--")
        plt.plot(X, y, "bo")
        # plt.show()

        # construct simple feature map
        param_x = Parameter("x")
        feature_map = QuantumCircuit(1, name="fm")
        feature_map.ry(param_x, 0)

        # construct simple ansatz
        param_y = Parameter("y")
        ansatz = QuantumCircuit(1, name="vf")
        ansatz.ry(param_y, 0)

        # construct a circuit
        qc = QuantumCircuit(1)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        # construct QNN
        regression_estimator_qnn = EstimatorQNN(
            circuit=qc, input_params=feature_map.parameters, weight_params=ansatz.parameters
        )

        # callback function that draws a live plot when the .fit() method is called
        def callback_graph(weights, obj_func_eval):
            # clear_output(wait=True)
            objective_func_vals.append(obj_func_eval)
            plt.title("Objective function value against iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Objective function value")
            plt.plot(range(len(objective_func_vals)), objective_func_vals)
            # plt.show()

        # construct the regressor from the neural network
        regressor = NeuralNetworkRegressor(
            neural_network=regression_estimator_qnn,
            loss="squared_error",
            optimizer=L_BFGS_B(maxiter=5),
            callback=callback_graph,
        )

        # create empty array for callback to store evaluations of the objective function
        objective_func_vals = []
        plt.rcParams["figure.figsize"] = (12, 6)

        # fit to data
        regressor.fit(X, y)

        # return to default figsize
        plt.rcParams["figure.figsize"] = (6, 4)

        # score the result
        regressor.score(X, y)


def test_qiskit_without_ml(capsys):
    """Use pytest.raises context to test that unbound parameter raises QiskitError"""
    with pytest.raises(qiskit.exceptions.QiskitError), capsys.disabled():
        theta = Parameter("theta")
        circuit = QuantumCircuit(1, name="fm")
        circuit.r(theta, np.pi / 2, 0)

        simulator = qiskit.Aer.get_backend('aer_simulator')
        # circuit = qiskit.transpile(circuit, simulator)

        # You can't simulate a Qiskit circuit with an unbound parameter
        # simulator.run(circuit_transpiled.bind_parameters([0.2]))
        simulator.run(circuit)


def test_parameterized_ndarray(capsys):
    with capsys.disabled():
        param = Parameter("test")
        array = np.random.rand(4, 4)

        result = param * array
        print(result)


def test_parameterized_sparse_array(capsys):
    import scipy.sparse

    with pytest.raises(ValueError), capsys.disabled():
        param = Parameter("test")
        
        data = np.sqrt(range(4))
        array = scipy.sparse.spdiags(
            data=data, diags=[1], m=len(data), n=len(data)
        ).tocsc()

        result = param * array
        print(result)