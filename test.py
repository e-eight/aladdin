from aladdin import *
from scipy.optimize import minimize

single_qubit_sv = StateVector(1)
coin_toss = [CircuitLayer("H", target_qubits=[0])]
evolve(single_qubit_sv, coin_toss)
get_counts(single_qubit_sv, num_shots=2048)

two_qubit_sv = StateVector(2)
phi_plus = [
    CircuitLayer("H", target_qubits=[0]),
    CircuitLayer("X", iscontrolled=True, target_qubits=[0, 1]),
]
evolve(two_qubit_sv, phi_plus)
two_qubit_sv

get_counts(two_qubit_sv, num_shots=2048)

two_qubit_sv = StateVector.from_label("01")
swap = [
    CircuitLayer("X", True, [0, 1]),
    CircuitLayer("X", True, [1, 0]),
    CircuitLayer("X", True, [0, 1]),
]
evolve(two_qubit_sv, swap)
get_counts(two_qubit_sv)

three_qubit_sv = StateVector(3)
ghz = [
    CircuitLayer("H", target_qubits=[0]),
    CircuitLayer("X", True, target_qubits=[0, 1]),
    CircuitLayer("X", True, target_qubits=[1, 2]),
]
evolve(three_qubit_sv, ghz)
get_counts(three_qubit_sv, num_shots=2048)

def cost_function(theta: float) -> float:
    # Prepare ansatz
    two_qubit_sv = StateVector(2)
    parametric_gate = rx_gate(theta)
    rx_ansatz = [
        CircuitLayer("H", target_qubits=[0]),
        CircuitLayer("X", iscontrolled=True, target_qubits=[0, 1]),
        CircuitLayer(parametric_gate, target_qubits=[0]),
    ]
    evolve(two_qubit_sv, rx_ansatz)

    # Compute expectation value
    num_shots = 2048
    counts = get_counts(two_qubit_sv, num_shots=num_shots)
    exp_val = (counts["00"] + counts["11"] - counts["01"] - counts["10"]) / num_shots
    return exp_val


theta = np.pi / 4
minimum = minimize(cost_function, theta, method="COBYLA", tol=1e-6)

print(minimum)
