# solve sudoku 4 by 4 using linear programing

import numpy as np
from scipy.optimize import linprog

# sudoku setup
sudoku_dim = 4
box_row_n = box_col_n = 2


# convert 3d to 1d
def conversion_3d_1d(row_i, col_i, num_i):
    return int(sudoku_dim ** 2 * row_i + sudoku_dim * col_i + num_i)


# coefficient matrix A and b for Ax = b
n_decision_variables = sudoku_dim ** 3


# produce a row of A by checking squares
def square_condition(row_i, col_i):
    row = [0] * n_decision_variables  # initialize a row
    for num_i in range(sudoku_dim):
        ind = conversion_3d_1d(row_i, col_i, num_i)
        row[ind] = 1.  # coefficient from the constraint
    return row


# produce a row of A by checking columns
def column_condition(row_i, num_i):
    row = [0] * n_decision_variables  # initialize a row
    for col_i in range(sudoku_dim):
        ind = conversion_3d_1d(row_i, col_i, num_i)
        row[ind] = 1.  # coefficient from the constraint
    return row


# produce a row of A by checking rows
def row_condition(col_i, num_i):
    row = [0] * n_decision_variables  # initialize a row
    for row_i in range(sudoku_dim):
        ind = conversion_3d_1d(row_i, col_i, num_i)
        row[ind] = 1.  # coefficient from the constraint
    return row


# produce a row of A by checking small boxes
def box_condition(box_i, num_i):
    row = [0] * n_decision_variables  # initialize a row
    start_row_i = int(box_i/2)
    start_col_i = int(box_i % 2)
    for element in range(sudoku_dim):
        ind = conversion_3d_1d(start_row_i + int(element / 2), start_col_i + int(element % 2), num_i)
        row[ind] = 1.
    return row


# construct matrix A as a list
mat_A = []

for i in range(sudoku_dim):
    for j in range(sudoku_dim):
        mat_A += [square_condition(i, j)]
        mat_A += [column_condition(i, j)]
        mat_A += [row_condition(i, j)]
        mat_A += [box_condition(i, j)]

vec_b = [1] * len(mat_A)

# a list with given members [row_index, col_index, value]
given_nums = [[0, 0, 1],
              [0, 1, 2],
              [3, 2, 2],
              [3, 3, 4]]
# Set up the objective function
vec_c = [0]*n_decision_variables
for a_given_num in given_nums:
    ind = conversion_3d_1d(a_given_num[0], a_given_num[1], a_given_num[2] - 1)
    vec_c[ind] = -10.

res = linprog(c=vec_c, A_eq= mat_A, b_eq=vec_b, bounds=[0., 1.])
print(res)

