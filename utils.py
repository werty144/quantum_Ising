import numpy as np
from qiskit import *
from itertools import product
import pandas as pd

def make_neighbour_indices(n):
	return np.dstack((np.arange(n)[:-1], np.arange(n)[1:])).reshape((-1, 2))


def interaction_matrix_from_dict(n, pair_interaction):
    interaction_matrix = np.zeros((n, n))
    for ((i, j), interaction) in pair_interaction.items():
        interaction_matrix[i, j] = interaction
    return interaction_matrix


def H(spins, interactions, magnetic_fields=None):
	n = len(spins)

	if isinstance(interactions, dict):
		interactions = interaction_matrix_from_dict(n, interactions)
	
	if magnetic_fields is None:
		magnetic_fields = np.zeros(n)
		
	return (- spins.T @ interactions @ spins - magnetic_fields @ spins).item()


def Z(n, interactions, magnetic_fields=None, beta=1):
	z = 0
	for spins in product([-1, 1], repeat=n):
		z += np.exp(-beta * H(np.array(spins), interactions, magnetic_fields))
	return z


def Pr(spins, interactions, magnetic_fields=None, beta=1):
	return np.exp(-beta * H(spins, interactions, magnetic_fields)) / Z(len(spins), interactions, magnetic_fields, beta)


def get_probs_distribution(n, interactions, magnetic_fields=None, beta=1):
	configs, probs = [], []
	for spins in product([-1, 1], repeat=n):
		configs.append(list(spins))   
		probs.append(Pr(np.array(spins), interactions, magnetic_fields, beta))

	display(pd.DataFrame({'s': configs, 'Pr[s]': probs}))


def S_gate(J=1, beta=1):
	x = np.exp(beta/2)
	c = 0.5 * Z(2, {(0, 1): J}, beta=beta)
	mat = 1/c**(1/2) * np.array(
	[
		[x**J, x**(-J), 0, 0],
		[-x**(-J), x**J, 0, 0],
		[0, 0, x**(-J), x**J],
		[0, 0, -x**J, x**(-J)]
	]
	)
	s_gate = extensions.UnitaryGate(mat, 'S')
	return s_gate


def X_gate():
    mat = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    x_gate = qiskit.extensions.UnitaryGate(mat, 'X')
    return x_gate


def linear_circuit(n, interactions, magnetic_fields=None, beta=1):
	qr = QuantumRegister(n, 'q')
	circ = QuantumCircuit(qr)
	circ.h(0)
	for i in range(1, n):
		circ.append(S_gate(interactions[i - 1, i], beta), [qr[i], qr[i - 1]])
	return circ


def circled_circuit(n, interactions, magnetic_fields=None, beta=1):
	circ = linear_circuit(n, interactions, magnetic_fields, beta)
	w = QuantumRegister(1, 'w')
	circ.add_register(w)
	circ.append(S_gate(interactions[n - 1, 0], beta), [w[0], circ.qubits[-2]])
	circ.append(X_gate(), [w[0], circ.qubits[0]])

	return circ


def process_circled_result(result):
	filtered_results = {key: value for (key, value) in result.get_counts().items() if key[0] == '1'}
	summa = sum(filtered_results.values())
	normalized_filtered = {key[1:]: value/summa for (key, value) in filtered_results.items()}
	return normalized_filtered
