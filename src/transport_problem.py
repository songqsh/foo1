# Transport problem
import numpy as np
from simplex import simplex_solver

# setup TP
source_constraint_list = [75, 125, 100]
destination_constraint_list = [80, 65, 70, 85]
cost_tab = [[464, 513, 654, 867],
            [352, 416, 690, 791],
            [995, 682, 388, 685]
            ]

n_source = len(source_constraint_list)
n_destination = len(destination_constraint_list)
n_decision_var = n_source * n_destination


def convert_2d_1d(i_source, i_destination):
    return i_source * n_destination + i_destination


# setup LP
c = []
for i_src in range(n_source):
    c += cost_tab[i_src]
c = - np.array(c)

b = source_constraint_list + destination_constraint_list

A = []
for i_src in range(n_source):
    row = [0] * n_decision_var
    for i_dst in range(n_destination):
        row[convert_2d_1d(i_src, i_dst)] = 1
    A += [row]
for i_dst in range(n_destination):
    row = [0] * n_decision_var
    for i_src in range(n_source):
        row[convert_2d_1d(i_src, i_dst)] = 1
    A += [row]

# initial tableau with big M
M = 10000.

# first row of the tableau
tab01 = - np.array(c).reshape((1, len(c))) - M * np.ones((1, len(A))) @ np.array(A)
tab0 = np.append(np.array([1]).reshape((1, 1)), tab01, axis=1)
tab0 = np.append(tab0, np.zeros((1, len(b))), axis=1)
tab0 = np.append(tab0, - M * np.ones((1, len(b))) @ np.array(b).reshape((len(b), 1)), axis=1)

tab1 = np.append(np.zeros((len(b), 1)), np.array(A), axis=1)
tab1 = np.append(tab1, np.eye(len(b)), axis=1)
tab1 = np.append(tab1, np.array(b).reshape((len(b), 1)), axis=1)

tab = np.append(tab0, tab1, axis=0)

simplex_solver(tab, display=1)