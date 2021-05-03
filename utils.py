import numpy as np
from qiskit import *
from itertools import product, combinations
import pandas as pd
from scipy.stats import chisquare
from collections import defaultdict
from os import path, getcwd
from tqdm import tqdm
from math import log, sqrt, comb, exp


def make_neighbour_indices(n):
	return np.dstack((np.arange(n)[:-1], np.arange(n)[1:])).reshape((-1, 2))


def interaction_matrix_from_dict(n, pair_interaction):
    interaction_matrix = np.zeros((n, n))
    for ((i, j), interaction) in pair_interaction.items():
        interaction_matrix[i, j] = interaction
    return interaction_matrix


def H(spins, interactions, magnetic_fields=None):
	return -sum([interaction * spins[i] * spins[j] for (i, j), interaction in interactions.items()])


def spins_energy(spins, interactions, magnetic_fields=None, beta=1):
	return np.exp(-beta * H(spins, interactions, magnetic_fields))


def spins_beta_H(spins, interactions, magnetic_fields=None, beta=1):
	return -beta * H(spins, interactions, magnetic_fields)


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


def add_h_cnot_to_all(circuit):
	b = QuantumRegister(1, 'b')
	circuit.add_register(b)
	circuit.h(b[0])
	for qubit in circuit.qubits[:-1]:
		circuit.cnot(b[0], qubit)


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


def process_balanced_result(result):
	return {key[1:]: value for (key, value) in result.items()}


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


def draw_energy_hist(observations, interactions, magnetic_fields=None, beta=1):
	energies = []
	for observation, frequency in tqdm(observations.items()):
		energy = spins_beta_H(spins_from_string01(observation), interactions, magnetic_fields, beta)
		energies += [energy] * frequency
	df = pd.DataFrame(energies, columns=['energy'])
	df.hist()


def to_csv(result, file_name):
	result_df = pd.DataFrame(zip(result.keys(), result.values()), columns=['Measurement outcome', 'Frequency'])
	with open(file_name, 'w+') as f:
		result_df.to_csv(f, index=False)


def run_and_write_to_csv(circuit, backend_name, provider, result_file_name=None, shots_n=8192):
	backend = provider.get_backend(backend_name)
	job = qiskit.execute(circuit, backend, shots=shots_n)
	qiskit.tools.job_monitor(job)
	result = job.result().get_counts()

	if result_file_name is None:
		result_file_name = path.join(getcwd(), 'measured_data', backend_name + '_' + str(shots_n) + '_' + str(job.job_id()) + '.csv')
	to_csv(result, result_file_name)


def estimate_beta(result, interactions):
	H_frequency = defaultdict(lambda: 0)
	for observation, frequency in result.items():
		cur_H = H(spins_from_string01(observation), interactions)
		H_frequency[cur_H] += frequency

	Xs, Ys = [], []
	pairs = list(H_frequency.items())
	if len(pairs) == 1:
		return None

	for i in range(1, len(pairs)):
		H1, freq1 = pairs[i - 1]
		H2, freq2 = pairs[i]
		Xs.append(-(H1 - H2))
		Ys.append(log(freq1 / freq2))

	numerator = 0
	denominator = 0
	for i in range(len(Xs)):
		numerator += Xs[i] * Ys[i]
		denominator += Xs[i] ** 2

	return numerator / denominator


def ouctomes_frequencies(result):
	outcomes, frequencies = [], []
	for outcome, frequency in result.items():
		outcomes.append(outcome)
		frequencies.append(frequency)

	total = sum(frequencies)

	return outcomes, [frequency/total for frequency in frequencies]


def kullback_leibler_divergence(result, interactions, probability_denomenator=None, beta=1):
	outcomes, frequencies = ouctomes_frequencies(result)

	n = len(outcomes[0])
	all_obtained = True
	if not len(outcomes) == 2**n:
		all_obtained = False

	energy_map = {}
	for outcome in outcomes:
		energy_map[outcome] = spins_energy(spins_from_string01(outcome), interactions, beta=beta)

	if probability_denomenator is None:
		if n <= 15:
			probability_denomenator = Z(n, interactions, beta=beta)
		else:
			probability_denomenator = sum(energy_map.values())

	kl_sum = 0

	for outcome, frequency in zip(outcomes, frequencies):
		P = frequency
		Q = energy_map[outcome] / probability_denomenator
		kl_sum += P * log(P / Q)

	return kl_sum, all_obtained


def hellinger_distance(result, interactions, probability_denomenator=None, beta=1):
	outcomes, frequencies = ouctomes_frequencies(result)

	n = len(outcomes[0])
	all_obtained = True
	if not len(outcomes) == 2**n:
		all_obtained = False

	energy_map = {}
	for outcome in outcomes:
		energy_map[outcome] = spins_energy(spins_from_string01(outcome), interactions, beta=beta)

	if probability_denomenator is None:
		if n <= 15:
			probability_denomenator = Z(n, interactions, beta=beta)
		else:
			probability_denomenator = sum(energy_map.values())

	h_sum = 0
	for outcome, frequency in zip(outcomes, frequencies):
		P = frequency
		Q = energy_map[outcome] / probability_denomenator
		h_sum += (sqrt(P) - sqrt(Q))**2

	return 1/sqrt(2) * sqrt(h_sum), all_obtained


def total_variation_distance(result, interactions, probability_denomenator=None, beta=1):
	outcomes, frequencies = ouctomes_frequencies(result)

	n = len(outcomes[0])
	all_obtained = True
	if not len(outcomes) == 2**n:
		all_obtained = False

	energy_map = {}
	for outcome in outcomes:
		energy_map[outcome] = spins_energy(spins_from_string01(outcome), interactions, beta=beta)

	if probability_denomenator is None:
		if n <= 15:
			probability_denomenator = Z(n, interactions, beta=beta)
		else:
			probability_denomenator = sum(energy_map.values())

	positive_P_Q_diffs, negative_P_Q_diffs = [], []
	for outcome, frequency in zip(outcomes, frequencies):
		P = frequency
		Q = energy_map[outcome] / probability_denomenator
		if P - Q > 0:
			positive_P_Q_diffs.append(P - Q)
		else:
			negative_P_Q_diffs.append(P - Q)

	return max(sum(positive_P_Q_diffs), sum(negative_P_Q_diffs)), all_obtained



def kullback_leibler_divergence_energy(result, beta):
	# Only for linear Ising chain with all neighbours interactions equal to 1

	def get_alternations_n(observation: str):
		return sum([1 for i in range(1, len(observation)) if observation[i - 1] != observation[i]])

	def trivial_H(n, alternations_n):
		return -((n - 1) - 2 * alternations_n)

	def get_theoretical_enumerator(n, alternations_n, beta):
		cur_H = trivial_H(n, alternations_n)
		return 2 * comb(n - 1, alternations_n) * np.exp(-beta * cur_H)


	outcomes, frequencies = ouctomes_frequencies(result)
	n = len(outcomes[0])

	empiric_energy_prob = defaultdict(lambda: 0)
	for outcome, frequency in zip(outcomes, frequencies):
		alternations_n = get_alternations_n(outcome)
		empiric_energy_prob[alternations_n] += frequency

	theoretical_denomenator = 0
	for alternations_n in range(n):
		theoretical_denomenator += get_theoretical_enumerator(n, alternations_n, beta)

	kl_sum = 0
	for alternations_n, empiric_frequency in empiric_energy_prob.items():
		P = empiric_frequency
		Q = get_theoretical_enumerator(n, alternations_n, beta) / theoretical_denomenator
		kl_sum += P * log(P / Q)

	return kl_sum


def hellinger_distance_energy(result, beta):
	# Only for linear Ising chain with all neighbours interactions equal to 1

	def get_alternations_n(observation: str):
		return sum([1 for i in range(1, len(observation)) if observation[i - 1] != observation[i]])

	def trivial_H(n, alternations_n):
		return -((n - 1) - 2 * alternations_n)

	def get_theoretical_enumerator(n, alternations_n, beta):
		cur_H = trivial_H(n, alternations_n)
		return 2 * comb(n - 1, alternations_n) * np.exp(-beta * cur_H)


	outcomes, frequencies = ouctomes_frequencies(result)
	n = len(outcomes[0])

	empiric_energy_prob = defaultdict(lambda: 0)
	for outcome, frequency in zip(outcomes, frequencies):
		alternations_n = get_alternations_n(outcome)
		empiric_energy_prob[alternations_n] += frequency

	theoretical_denomenator = 0
	for alternations_n in range(n):
		theoretical_denomenator += get_theoretical_enumerator(n, alternations_n, beta)

	h_sum = 0
	for alternations_n, empiric_frequency in empiric_energy_prob.items():
		P = empiric_frequency
		Q = get_theoretical_enumerator(n, alternations_n, beta) / theoretical_denomenator
		h_sum += (sqrt(P) - sqrt(Q))**2

	return 1/sqrt(2) * sqrt(h_sum)

