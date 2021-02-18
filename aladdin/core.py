#!/usr/bin/env python3

from collections import defaultdict
import functools
from typing import List, Optional, Union, Dict, NamedTuple

import numpy as np

_ZBASIS = {"0": np.array([1.0, 0.0]), "1": np.array([0.0, 1.0])}


class StateVector:
    def __init__(self, num_qubits: int):
        """
        Creates a state vector representing the `num_qubits` zero state.

        Args:
        num_qubits (int): Number of qubits.
        """
        self.num_qubits = num_qubits
        self.data = np.zeros(2 ** num_qubits)
        self.data[0] = 1

    def __repr__(self):
        return f"{self.data}"

    def __str__(self):
        return f"{self.data}"

    @classmethod
    def from_label(cls, label: str):
        """
        Creates a state vector from a z-basis label.

        Args:
        label (str): String representing the state in the z-basis, or binary. For
        example "0", "00", "101".
        """
        num_qubits = len(label)
        sv = cls(num_qubits)
        sv.data = functools.reduce(np.kron, [_ZBASIS[c] for c in label])
        return sv


def measure(sv: StateVector, seed: Optional[int] = None) -> str:
    """
    Returns the result of a z-basis measurement on a quantum state. The original quantum
    state generarlly changes after the measurement (collapse of a wave function).

    Args:
    sv (StateVector): A `StateVector` encoding the quantum state.
    seed (int, optional): Seed for the pseudorandom number generator.

    Returns:
    The label of the resulting state.
    """
    label = _single_measure(sv.data, seed)
    sv.data = StateVector.from_label(label).data
    return label


def get_counts(
    sv: StateVector, num_shots: int = 1024, seed: Optional[int] = None
) -> Dict[str, int]:
    """
    Returns the counts of each z-basis quantum state measured during `num_shots`
    independent measurements. Unlike `measure` does not change the original quantum
    state.

    Args:
    sv (StateVector): A `StateVector` encoding the quantum state.
    num_shots (int, optional): The number of independent measurements made on the same
    initial state.
    seed (int, optional): Seed for the pseudorandom number generator.

    Returns:
    A dict with the counts of each z-basis quantum state measured.
    """
    counts = defaultdict(int)
    for _ in range(num_shots):
        measured_state = _single_measure(sv.data, seed)
        counts[measured_state] += 1
    return counts


def _single_measure(amplitudes: np.ndarray, seed: Optional[int] = None) -> str:
    """
    Helper function that does the actual measurement on a single quantum state. Returns
    the z-basis label of the resulting quantum state, without changing the original.

    Args:
    amplitudes (np.ndarray): A 1-dimensional np.ndarray containing the
    amplitudes of the quantum state.
    seed (int, optional): Seed for the pseudorandom number generator.

    Returns:
    The label of the resulting quantum state.
    """
    if amplitudes.ndim != 1:
        raise ValueError("The state vector must be 1-dimensional.")

    num_qubits = int(np.log2(len(amplitudes)))
    probabilities = np.real(amplitudes * np.conjugate(amplitudes))
    probabilities /= np.sum(probabilities)

    basis_states = np.arange(len(probabilities))
    rng = np.random.default_rng(seed)
    measured_state = rng.choice(basis_states, p=probabilities)

    return np.binary_repr(measured_state, num_qubits)


def general_unitary_gate(theta: float, phi: float, gamma: float) -> np.ndarray:
    """
    Returns the matrix corresponding to the most general single qubit unitary:

        |cos(θ/2)                -exp(-iγ) * sin(θ/2)|
        |exp(iφ) * sin(θ/2)   exp(iγ + iφ) * cos(θ/2)|.

    Args:
    {theta, phi, gamma} (float): Euler angles for the Bloch sphere.
    Returns:
    A np.ndarray encoding the matrix given above.
    """
    a = np.cos(theta / 2)
    b = -np.exp(-1.0j * gamma) * np.sin(theta / 2)
    c = np.exp(1.0j * phi) * np.sin(theta / 2)
    d = np.exp(1.0j * gamma + 1.0j * phi) * np.cos(theta / 2)
    return np.real_if_close(np.array([[a, b], [c, d]], dtype=np.csingle))


rx_gate = functools.partial(general_unitary_gate, phi=-np.pi / 2, gamma=np.pi / 2)
rx_gate.__doc__ = "Gate for rotation around X axis."
ry_gate = functools.partial(general_unitary_gate, phi=0.0, gamma=0.0)
ry_gate.__doc__ = "Gate for rotation around Y axis."
rz_gate = functools.partial(general_unitary_gate, theta=0.0, gamma=0.0)
rz_gate.__doc__ = "Gate for rotation around Z axis."

_SINGLE_QUBIT_GATES = {
    "I": general_unitary_gate(0.0, 0.0, 0.0),
    "X": general_unitary_gate(np.pi, 0.0, np.pi),
    "Y": general_unitary_gate(np.pi, np.pi / 2, np.pi / 2),
    "Z": general_unitary_gate(0.0, 0.0, np.pi),
    "H": general_unitary_gate(np.pi / 2, 0.0, np.pi),
    "S": general_unitary_gate(0.0, 0.0, np.pi / 2),
    "Sdag": general_unitary_gate(0.0, 0.0, -np.pi / 2),
    "T": general_unitary_gate(0.0, 0.0, np.pi / 4),
    "Tdag": general_unitary_gate(0.0, 0.0, -np.pi / 4),
}


class CircuitLayer(NamedTuple):
    gate: Union[str, np.ndarray]
    iscontrolled: bool = False
    target_qubits: List[int] = []


def evolve(sv: StateVector, circuit: List[CircuitLayer]):
    """
    Evolves a quantum state through a quantum circuit.

    Args:
    sv (StateVector): A `StateVector` encoding a quantum state.
    circuit (List[CircuitLayer]): The quantum circuit is presented as a list of
    `CircuitLayer`s.
    """
    for layer in circuit:
        sv.data = _evolve_single_layer(sv, layer)


def _evolve_single_layer(sv: StateVector, layer: CircuitLayer):
    """
    Helper function that evolves a quantum state through a single layer of a quantum
    circuit.

    Args:
    sv (StateVector): A `StateVector` encoding a quantum state.
    layer (CircuitLayer): A `CircuitLayer` encoding a layer of the circuit. A
    `CircuitLayer`, contains three fields: `gate`, `iscontrolled`, and `target_qubits`,
    `gate` can be either string or a 2 x 2 np.ndarray. If a string then it must be one
    of the gates defined in _SINGLE_QUANTUM_GATES. The boolean `iscontrolled`, set to
    False by default, tells if `gate` is controlled or not. If controlled then
    `target_qubits` should be in the form [control_qubit, target_qubit].
    """
    if layer.iscontrolled:
        if len(layer.target_qubits) != 2:
            raise ValueError(
                "target_qubits for controlled gates must be in the form of [control_qubit, target_qubit]."
            )
        return _apply_controlled_gate(sv, layer.gate, *layer.target_qubits)
    return _apply_single_qubit_gates(sv, layer.gate, layer.target_qubits)


def _apply_single_qubit_gates(
    sv: StateVector,
    gate: Union[str, np.ndarray],
    target_qubits: List[int],
) -> None:
    """
    Applies a single qubit gate to the target qubits in the quantum state. This
    generally changes the quantum state.

    Args:
    sv (StateVector): A `StateVector` encoding a quantum state.
    gate (str, np.ndarray): A string representing the gate, or a np.ndarray of shape
    (2, 2) containing the matrix representation of the gate. If a string then it can
    be either I, X, Y, Z, H, S, Sdag, T, or Tdag.
    target_qubits (List[int]): Qubits on which the gate is to be applied.
    """
    num_qubits = sv.num_qubits
    if len(target_qubits) > num_qubits:
        raise ValueError(
            "Mismatch in number of qubits: len(target_qubit) cannot be larger than number of qubits."
        )
    if max(target_qubits) >= num_qubits or min(target_qubits) < 0:
        raise IndexError("The target qubits must be between 0 and num_qubits - 1")
    if isinstance(gate, str) and gate not in _SINGLE_QUBIT_GATES:
        raise ValueError(
            f"Gate not defined. Either choose a value from {_SINGLE_QUBIT_GATES.items()} or pass a matrix."
        )
    if isinstance(gate, np.ndarray):
        gate = gate.squeeze()
        if gate.shape != (2, 2):
            raise ValueError("Gate must be a 2 x 2 matrix.")

    single_gate_matrix = (
        gate if isinstance(gate, np.ndarray) else _SINGLE_QUBIT_GATES[gate]
    )
    gate_list = [
        (
            lambda x: single_gate_matrix
            if x in target_qubits
            else _SINGLE_QUBIT_GATES["I"]
        )(i)
        for i in range(num_qubits)
    ]
    gate_matrix = np.real_if_close(functools.reduce(np.kron, gate_list))
    return np.real_if_close(gate_matrix @ sv.data)


def _apply_controlled_gate(
    sv: StateVector,
    gate: Union[str, np.ndarray],
    control_qubit: int,
    target_qubit: int,
) -> None:
    """
    Applies a CU gate controlled on the `control qubit` to the `target_qubit`. This
    generally changes the quantum state.

    Args:
    sv (StateVector): A `StateVector` encoding the quantum state.
    gate (str, np.ndarray): A string representing the gate, or a np.ndarray of shape
    (2, 2) containing the matrix representation of the gate. If a string then it can
    be either I, X, Y, Z, H, S, Sdag, T, or Tdag.
    control_qubit (int): Control qubit for the CNOT gate.
    target_qubit (int): Target qubit for the CNOT gate.
    """
    num_qubits = sv.num_qubits
    if num_qubits < 2:
        raise ValueError(
            "A controlled gate cannot be applied to a state with less than 2 qubits."
        )
    if (
        max(control_qubit, target_qubit) >= num_qubits
        or min(control_qubit, target_qubit) < 0
    ):
        raise IndexError(
            "The control and target qubits must be between 0 and num_qubits - 1"
        )
    if isinstance(gate, str) and gate not in _SINGLE_QUBIT_GATES:
        raise ValueError(
            f"Gate not defined. Either choose a value from {_SINGLE_QUBIT_GATES.items()} or pass a matrix."
        )
    if isinstance(gate, np.ndarray) and gate.shape != (2, 2):
        raise ValueError("Gate must be a 2 x 2 matrix.")

    zero_proj = np.outer(_ZBASIS["0"], _ZBASIS["0"])
    zero_list = [
        (lambda x: zero_proj if x == control_qubit else _SINGLE_QUBIT_GATES["I"])(i)
        for i in range(num_qubits)
    ]

    single_gate_matrix = (
        gate if isinstance(gate, np.ndarray) else _SINGLE_QUBIT_GATES[gate]
    )
    one_proj = np.outer(_ZBASIS["1"], _ZBASIS["1"])
    one_list = [
        (
            lambda x: one_proj
            if x == control_qubit
            else single_gate_matrix
            if x == target_qubit
            else _SINGLE_QUBIT_GATES["I"]
        )(i)
        for i in range(num_qubits)
    ]

    gate_matrix = np.add(
        np.real_if_close(functools.reduce(np.kron, zero_list)),
        np.real_if_close(functools.reduce(np.kron, one_list)),
    )
    return np.real_if_close(gate_matrix @ sv.data)
