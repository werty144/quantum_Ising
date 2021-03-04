import numpy as np
from qiskit import *
from itertools import product
import pandas as pd

def make_neighbour_indices(n):
	return np.dstack((np.arange(n)[:-1], np.arange(n)[1:])).reshape((-1, 2))

def H(spins, pair_indices, couplings=None, magnetic_fields=None):
	n = len(spins)

	if couplings is None:
		couplings = np.zeros((n, n))
		couplings[pair_indices[:,0], pair_indices[:,1]] = np.ones(len(pair_indices))
	
	if magnetic_fields is None:
		magnetic_fields = np.zeros(n)
		
	return (- spins.T @ couplings @ spins - magnetic_fields @ spins).item()

def Z(n, pair_indices, beta=1, **kwargs):
	z = 0
	for spins in product([-1, 1], repeat=n):
		z += np.exp(-beta * H(np.array(spins), pair_indices, **kwargs))
	return z

def Pr(spins, pair_indices, beta=1, **kwargs):
	return np.exp(-beta * H(spins, pair_indices, **kwargs)) / Z(len(spins), pair_indices, beta=beta, **kwargs)

def get_probs_distribution(n, pair_indices=None):
	if pair_indices is None:
		pair_indices = make_neighbour_indices(n)
	configs, probs = [], []
	for spins in product([-1, 1], repeat=n):
		configs.append(list(spins))   
		probs.append(Pr(np.array(spins), pair_indices))

	display(pd.DataFrame({'s': configs, 'Pr[s]': probs}))

def S_gate(J=1, beta=1):
	x = np.exp(beta/2)
	c = 0.5 * Z(2, np.array([[0, 1]]))
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

def linear_circuit(n):
	qr = QuantumRegister(n, 'q')
	circ = QuantumCircuit(qr)
	circ.h(0)
	for i in range(1, n):
		circ.append(S_gate(), [qr[i], qr[i - 1]])
	return circ