Aladdin is a state vector based toy quantum computing simulator written
in Python with NumPy. This is my submission for the QOSF 2021 mentorship
program. At present only state vector and some single qubit gates have
been implemented. No double qubit gates are implemented directly. The
actions of controlled-unitary gates are evaluated by using the relation
*C**U* = \|0⟩⟨0\| ⊗ *I* + \|1⟩⟨1\| ⊗ *U*. Aladdin uses the big-endian
notation.

Why Aladdin?
============

Because quantum is magic. And this project is all about demystifying
that as much as possible.

State vectors
=============

The `StateVector` class provides all the functionalities for
initializing state vectors. State vectors can be either initialized by
the number of qubits `StateVector(2)` or from a z-basis label
`StateVector.from_label("10")`.

Quantum gates
=============

Single qubit gates are implemented as special cases of the most general
single qubit unitary:
<img src="https://latex.codecogs.com/svg.image?\begin{equation}U&space;=&space;\begin{pmatrix}\cos&space;(\theta&space;/&space;2)&space;&&space;-e^{i\gamma}&space;\sin&space;(\theta&space;/&space;2)&space;\\&space;e^{i\phi}&space;\sin&space;(\theta&space;/&space;2)&space;&&space;e^{i\gamma&space;&plus;&space;i\phi}&space;\cos&space;(\theta&space;/&space;2)\end{pmatrix}\end{equation}&space;" title="\begin{equation}U = \begin{pmatrix}\cos (\theta / 2) & -e^{i\gamma} \sin (\theta / 2) \\ e^{i\phi} \sin (\theta / 2) & e^{i\gamma + i\phi} \cos (\theta / 2)\end{pmatrix}\end{equation} " />

This gate is implemented in the `general_unitary_gate` function which
takes in as argument the three angles `theta`, `phi`, and `gamma`. Any
single qubit unitary can be formed by providing proper values of those
angles. Additionally the rotational gates *R*<sub>*x*</sub>,
*R*<sub>*y*</sub>, and *R*<sub>*z*</sub> have also been provided for
ease. Other single qubit gates implemented internally, but not
accessible to the user directly are:

-   Pauli gates: *I*, *X*, *Y*, *Z*
-   Clifford gates: *H*, *S*, *S*<sup>†</sup>
-   C3 gates: *T*, *T*<sup>†</sup>

The way to use these gates is detailed below.

Quantum circuit
===============

Quantum circuits are implemented as list of `CircuitLayer`'s. The class
`CircuitLayer` contains three fields:

-   `gate`: `gate` can either be a string from
    `I, X, Y, Z, H, S, Sdag, T, Tdag` representing the gates mentioned
    above or an `ndarray` of shape `(2, 2)` representing any arbitrary
    user-defined gate,
-   `iscoupled`: `iscoupled` is a flag that tells if the gate is
    controlled or not (default is `False`),
-   `target_qubits`: `target_qubits` is either a list of qubits on which
    `gate` will act if not controlled or `[control_qubit, target_qubit]`
    if the gate is controlled.

Two evolve a quantum state through a quantum circuit use `evolve(state,
circuit)`, `evolve` modifies the state vector in place.

Examples
========

Here are some examples to demonstrate the capabilities of Aladdin. All
the results can be reproduced by running `test.py`. This has been tested
on Python 3.7.9, with NumPy 1.20.1 and SciPy 1.6.1. You can install the
requirements by executing `pip install requirements.txt`, preferably in
your local Python environment.

    from aladdin import *
    from scipy.optimize import minimize

Coin toss
---------

Coin toss is demonstrated by repeated measurements of the \| + ⟩ state.

    single_qubit_sv = StateVector(1)
    coin_toss = [CircuitLayer("H", target_qubits=[0])]
    evolve(single_qubit_sv, coin_toss)
    counts = get_counts(single_qubit_sv, num_shots=2048)
    print(*counts.items())
    
    # output
    ('0', 1023) ('1', 1025)

Bell basis states
-----------------

The Bell basis states are the maximally entangled two qubit Bell states.
Here I will show how to produce the \|*Φ*<sup>+</sup>⟩ ∝ \|00⟩ + \|11⟩
state.

    two_qubit_sv = StateVector(2)
    phi_plus = [
        CircuitLayer("H", target_qubits=[0]),
        CircuitLayer("X", iscontrolled=True, target_qubits=[0, 1]),
    ]
    evolve(two_qubit_sv, phi_plus)
    print(two_qubit_sv)
    
    # output
    [7.07106769e-01 0.00000000e+00 4.32978040e-17 7.07106769e-01]

An easier way to see that it is indeed the \|*Φ*<sup>+</sup>⟩ state is
to do repeated measurements - \|00⟩ and \|11⟩ should appear in roughly
equal numbers (actually that is also true for the \|*Φ*<sup>−</sup>⟩
state.)

    counts = get_counts(two_qubit_sv, num_shots=2048)
    print(*counts.items())
    
    # output
    ('11', 1044) ('00', 1004)

SWAP gate
---------

The SWAP gate swaps the states of two qubits, for example \|01⟩ becomes
\|10⟩. The SWAP gate is not internally implemented in Aladdin, but it can
be represented by three consecutive CNOT gates. The control and target
qubits are flipped for the second CNOT gate in comparison to the other
two.

    two_qubit_sv = StateVector.from_label("01")
    swap = [
        CircuitLayer("X", True, [0, 1]),
        CircuitLayer("X", True, [1, 0]),
        CircuitLayer("X", True, [0, 1]),
    ]
    evolve(two_qubit_sv, swap)
    counts = get_counts(two_qubit_sv)
    print(*counts.items())
    
    # output
    ('10', 1024)

GHZ state
---------

The three qubit GHZ state is an entangled state defined by
$$
\\vert\\mathrm{GHZ}\\rangle = \\frac{1}{\\sqrt{2}}( \|000\\rangle + \|111\\rangle).
$$

    three_qubit_sv = StateVector(3)
    ghz = [
        CircuitLayer("H", target_qubits=[0]),
        CircuitLayer("X", True, target_qubits=[0, 1]),
        CircuitLayer("X", True, target_qubits=[1, 2]),
    ]
    evolve(three_qubit_sv, ghz)
    counts = get_counts(three_qubit_sv, num_shots=2048)
    print(*counts.items())

    # output
    ('111', 1026) ('000', 1022)

Variational quantum eigensolver
-------------------------------

Variational quantum eigensolver (VQE) is a hybrid quantum-classical
algorithm that can possibly show quantum advantage on NISQ devices. It
is used to find the ground state energy , or lowest eigenvalue, of
Hamiltonians. I will use VQE to find the lowest eigenvalue of a very
simple Hamiltonian *Z**Z*. The lowest eigenvalue of this Hamiltonian can
be easily classically calculated. It is -1. For this demonstration I
will use an ansatz consisting of *R*<sub>*x*</sub>, *H*, and *C**X*
gates.

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

    # output
    fun: -1.0
    maxcv: 0.0
    message: 'Optimization terminated successfully.'
    nfev: 19
    status: 1
    success: True
    x: array(3.16039816)
