from typing import Iterable
import warnings

import numpy
import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.parameter import ParameterExpression


class ParameterizedUnitaryGate(Gate):
    """UnitaryGate sublcass that stores the operator matrix for later reference by animation utility."""

    def __init__(
        self,
        op_func,
        params,
        num_qubits,
        cutoffs,
        label=None,
        duration=100,
        unit="ns",
        discretized_param_indices: list = [],
    ):
        """Initialize ParameterizedUnitaryGate

        Args:
            op_func (function): function to build operator matrix
            params (List): List of parameters to pass to op_func to build operator matrix (supports instances of Qiskit Parameter to be bound later)
            num_qubits (int): Number of qubits in the operator -- this would likely equate to (num_qubits_per_qumode * num_qumodes + num_ancilla).
            label (string, optional): Gate name. Defaults to None.
            duration (int, optional): Duration of gate used for noise modeling. Defaults to 100.
            unit (string, optional): Unit of duration (only supports those allowed by Qiskit).
            discretized_param_indices (list): list of int indices into self.params for parameters to be discretized. An empty list will discretize all params.
        """
        super().__init__(name=label, num_qubits=num_qubits, params=params, label=label)

        self.op_func = op_func

        self._parameterized = any(
            isinstance(param, ParameterExpression) and param.parameters
            for param in params
        )

        self.duration = duration
        self.unit = unit
        self.discretized_param_indices = discretized_param_indices
        self.cutoffs = cutoffs

    def __array__(self, dtype=None):
        """Call the operator function to build the array using the bound parameter values."""
        # return self.op_func(*map(complex, self.params)).toarray()
        values = []

        # Add parameters for op_func call
        for param in self.params:
            if isinstance(param, ParameterExpression):
                # if param.is_real():
                #     values.append(float(param))
                # else:
                #     values.append(complex(param))
                values.append(
                    complex(param)
                )  # just cast everything to complex to avoid errors in Ubuntu/MacOS vs Windows
            else:
                values.append(param)

        # Add cutoff for each parameter
        values.extend(self.cutoffs)

        # Conver array to tupple
        values = tuple(values)

        return self.op_func(*values).toarray()

    def _define(self):
        try:
            mat = self.to_matrix()
            q = QuantumRegister(self.num_qubits)
            qc = QuantumCircuit(q, name=self.name)
            rules = [
                (UnitaryGate(mat, self.label), [i for i in q], []),
            ]
            for instr, qargs, cargs in rules:
                qc._append(instr, qargs, cargs)

            self.definition = qc
        except:
            # warnings.warn("Unable to define gate, setting definition to None to prevent serialization errors for parameterized unitary gates.")
            self.definition = None

    def validate_parameter(self, parameter):
        """Gate parameters should be int, float, complex, or ParameterExpression"""
        if numpy.iscomplexobj(parameter):
            # Turn all numpy complex values into native python complex objects so that
            # they can't be cast to float without raising an error.
            if isinstance(parameter, Iterable):
                return [complex(p) for p in parameter]

            return complex(parameter)
        elif isinstance(parameter, ParameterExpression) and not parameter.is_real():
            return parameter
        elif isinstance(parameter, (str, list)):  # accept strings as-is
            return parameter
        else:
            return super().validate_parameter(parameter)

    def calculate_matrix(
        self, current_step: int = 1, total_steps: int = 1, keep_state: bool = False
    ):
        """Calculate the operator matrix by executing the selected function.
        Increment the parameters based upon the current and total steps.

        Args:
            current_step (int, optional): Current step within total_steps. Defaults to 1.
            total_steps (int, optional): Total steps to increment parameters. Defaults to 1.

        Returns:
            ndarray: operator matrix
        """
        if self.is_parameterized():
            raise NotImplementedError(
                "Unable to calculate incremental operator matrices for parameterized gate"
            )

        values = self.calculate_segment_params(current_step, total_steps, keep_state)

        # if self.inverse:
        #     result = scipy.sparse.linalg.inv(self.op_func(*values))
        # else:
        #     result = self.op_func(*values)
        result = self.op_func(*values)

        if hasattr(result, "toarray"):
            result = result.toarray()

        return result


def __calculate_segment_params(
    self, current_step: int = 1, total_steps: int = 1, keep_state: bool = False
):
    """
    Calculate the parameters at the current step. Return a tuples of the values.

     Args:
        current_step (int): 0-based current step index of the discretization
        total_steps (int): total number of discretization steps
        keep_state (bool): true if the state should be kept between discretization steps (i.e., if the discretization value should be 1/total_steps vs current_step/total_steps)

    Returns:
        discretized parameter values as tuple
    """
    if keep_state:
        param_fraction = 1 / total_steps
    else:
        param_fraction = current_step / total_steps

    values = []
    for index, param in enumerate(self.params):
        if (
            not hasattr(self, "discretized_param_indices")
            or len(self.discretized_param_indices) == 0
            or index in self.discretized_param_indices
        ):
            values.append(param * param_fraction)
        else:
            values.append(param)

    return tuple(values)


def __calculate_segment_duration(
    self, current_step: int = 1, total_steps: int = 1, keep_state: bool = False
):
    """Calculate the duration at the current step. Return a tuple of the (duration, unit)."""
    frame_duration = None

    if self.duration:
        if keep_state:
            fraction = 1 / total_steps
        else:
            fraction = current_step / total_steps

        frame_duration = self.duration * fraction

    return frame_duration, self.unit


# Monkey patch Qiskit Instruction to support animating base Qiskit Instruction
qiskit.circuit.instruction.Instruction.calculate_segment_params = (
    __calculate_segment_params
)
qiskit.circuit.instruction.Instruction.calculate_segment_duration = (
    __calculate_segment_duration
)
