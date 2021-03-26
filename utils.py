import numpy as np
from qiskit import *
from itertools import product, combinations
import pandas as pd
from scipy.stats import chisquare


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


def spins_energy(spins, interactions, magnetic_fields=None, beta=1):
	return np.exp(-beta * H(spins, interactions, magnetic_fields))


def Z(n, interactions, magnetic_fields=None, beta=1):
	z = 0
	for spins in product([-1, 1], repeat=n):
		spins = np.array(spins)
		z += spins_energy(spins, interactions, magnetic_fields, beta)
	return z


def Pr(spins, interactions, magnetic_fields=None, beta=1):
	return spins_energy(spins, interactions, magnetic_fields, beta) / Z(len(spins), interactions, magnetic_fields, beta)


def get_probs_distribution(n, interactions, magnetic_fields=None, beta=1):
	configs, probs = [], []
	for spins in product([-1, 1], repeat=n):
		configs.append(string01_from_spins(spins))  
		probs.append(Pr(np.array(spins), interactions, magnetic_fields, beta))

	return pd.DataFrame({'s': configs, 'Pr[s]': probs})


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


def Omega_gate(J=1, beta=1):
	from qiskit.quantum_info.operators import Operator
	from qiskit.extensions import UnitaryGate

	x = np.exp(beta/2)
	c = 0.5 * Z(2, {(0, 1): J}, beta=beta)
	S_mat = 1/c**(1/2) * np.array(
	[
		[x**J, x**(-J), 0, 0],
		[-x**(-J), x**J, 0, 0],
		[0, 0, x**(-J), x**J],
		[0, 0, -x**J, x**(-J)]
	]
	)
	X_mat = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ]
	)
    
	S_operator, X_operator = Operator(S_mat), Operator(X_mat)

	res_operator = Operator(np.eye(8)).compose(S_operator, qargs=[2, 0]).compose(X_operator, qargs=[2, 1])

	return UnitaryGate(res_operator, 'Omega')


def linear_circuit(n, interactions, beta=1):
	qr = QuantumRegister(n, 'q')
	circ = QuantumCircuit(qr)
	circ.h(0)
	for i in range(1, n):
		circ.append(S_gate(interactions[i - 1, i], beta), [qr[i], qr[i - 1]])
	return circ


def circled_circuit(n, interactions, beta=1):
	circ = linear_circuit(n, interactions, beta)
	w = QuantumRegister(1, 'w')
	circ.add_register(w)
	circ.append(Omega_gate(interactions[n - 1, 0], beta), [circ.qubits[-2], circ.qubits[0], w[0]])

	return circ


def lattice_draft(plaquettes_interactions, hv_edges_n, beta=1):
	VERTICES_IN_PLAQUETTE = 4

	plaquettes_n = len(plaquettes_interactions)
	qr = QuantumRegister(plaquettes_n * VERTICES_IN_PLAQUETTE, 'q')
	w_close = QuantumRegister(plaquettes_n, 'wClose')
	circ = QuantumCircuit(qr, w_close)
	for i, interactions in enumerate(plaquettes_interactions):
		pairs = list(interactions.keys())
		begin_qubit = pairs[0][0]
		end_qubit = None
		circ.h(qr[begin_qubit])
		for frm, to in pairs:
			if to == begin_qubit:
				end_qubit = frm
				continue
			circ.append(S_gate(interactions[frm, to], beta), [qr[to], qr[frm]])

		circ.append(Omega_gate(interactions[end_qubit, begin_qubit], beta), [qr[end_qubit], qr[begin_qubit], w_close[i]])

	return circ


def lattice_curcuit(plaquettes_interactions, hv_interactions, beta=1):
	VERTICES_IN_PLAQUETTE = 4

	hv_edges_n = len(hv_interactions.keys())
	circ = lattice_draft(plaquettes_interactions, hv_edges_n, beta)
	qr = circ.qregs[0]

	w_connect = QuantumRegister(hv_edges_n, 'wConnect')
	circ.add_register(w_connect)

	for i, ((frm, to), interaction) in enumerate(hv_interactions.items()):
		circ.append(Omega_gate(interaction, beta), [qr[frm], qr[to], w_connect[i]])

	return circ


def process_lattice_result(result, work_bits_n):
	return  {key[work_bits_n:]: value for (key, value) in result.items() if key[:work_bits_n] == '1' * work_bits_n}



def process_circled_result(result):
	return  {key[1:]: value for (key, value) in result.items() if key[0] == '1'}


def spins_from_string01(s):
	return np.array([-1 if c == '0' else 1 for c in s])


def string01_from_spins(spins):
	return ''.join(['0' if spin == -1 else '1' for spin in spins])


def distribution_pseudo_pvalue(probabilities, interactions, magnetic_fields=None, beta=1):
	measurements = probabilities.keys()
	real_energies = dict()
	for mes in measurements:
		real_energies[mes] = spins_energy(spins_from_string01(mes), interactions, magnetic_fields, beta)
	ratio_sum = 0
	combs = 0
	for (mes1, mes2) in combinations(measurements, r=2):
		combs += 1
		obtained_probs_ratio = probabilities[mes1] / probabilities[mes2]
		real_probs_ratio = real_energies[mes1] / real_energies[mes2]
		ratio_sum += max(obtained_probs_ratio, real_probs_ratio) / min(obtained_probs_ratio, real_probs_ratio)

	return 1 / (ratio_sum / combs)


def result_from_csv(csv_file_name):
	df = pd.read_csv(csv_file_name, dtype={'Computational basis states': str, 'Measurement outcome': str})
	result = {row[0]: row[1] for (ind, row) in df.iterrows()}
	return result


def distribution_chi2_pvalue(observed_frequencies, interactions, magnetic_fields=None, beta=1):
    f_obs, energies = [], []
    shots_n = sum(observed_frequencies.values())
    for observation, frequency in observed_frequencies.items():
        f_obs.append(frequency)
        energies.append(spins_energy(spins_from_string01(observation), interactions, magnetic_fields, beta))

    energy_sum = sum(energies)
    f_exp = list(map(lambda energy: energy/energy_sum * shots_n, energies))
    return chisquare(f_obs, f_exp)


def to_csv(result, file_name):
	result_df = pd.DataFrame(zip(result.keys(), result.values()), columns=['Measurement outcome', 'Frequency'])
	with open(file_name, 'w+') as f:
		result_df.to_csv(f, index=False)
