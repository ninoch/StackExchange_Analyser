import numpy as np
from scipy.stats import norm 
import pickle

acceptance_over_session = []

def load_acceptance_values(file_name):
	global acceptance_over_session

	with open(file_name, 'rb') as f:
		tmp = pickle.load(f)

	acceptance_over_session = tmp[2]


def z_test(p2, p1, n2, n1):
	p = (p1 * n1 + p2 * n2) / (n1 + n2)
	z = (p2 - p1) / np.sqrt(p * (1.0 - p) * ((1.0 / n1) + (1.0 / n2)))
	pval = norm.sf(z)
	return pval

print "_______________________________________________________"
load_acceptance_values("Randomized_over_one_user/all_variables.obj")
print "Z test output for randomized over one user: "
print z_test(acceptance_over_session[1][0], acceptance_over_session[1][1], 1.6 * 1000 * 1000, 1.6 * 1000 * 1000)

print "_______________________________________________________"
load_acceptance_values("Randomized_all_data/all_variables.obj")
print "Z test output for randomized all data: "
print z_test(acceptance_over_session[1][0], acceptance_over_session[1][1], 1.6 * 1000 * 1000, 1.6 * 1000 * 1000)

print "_______________________________________________________"
load_acceptance_values("Original_data/all_variables.obj")
print "Z test output for original data: "
print z_test(acceptance_over_session[1][0], acceptance_over_session[1][1], 1.6 * 1000 * 1000, 1.6 * 1000 * 1000)

print "_______________________________________________________"