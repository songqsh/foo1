# simplex method in linear programing

# import package
import numpy as np
import pandas as pd
import warnings

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})

# ignore by message
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")


# pivoting with (i,j)
def pivoting(A, i, j):
    A[i] = A[i] / A[i, j]  # scale to get one in (i,j)
    n_rows, _ = A.shape
    for k in range(n_rows):
        if k == i:
            continue  # skip i-row
        A[k] = A[k] - A[i] * A[k, j]  # replacement to get zero


# minimum ratio test
def min_ratio_test(ratio_list):
    pivot_row = 0
    min_ratio = np.inf
    for i in range(len(ratio_list)):
        if 0 < ratio_list[i] < min_ratio:
            pivot_row = i + 1
            min_ratio = ratio_list[i]
    if min_ratio is np.inf:
        return 0  # no leaving variable
    else:
        return pivot_row


def new_pivot(M_mat):  # M is the tableau
    row_n, col_n = M_mat.shape
    row_n = row_n - 1  # number of constraints in the augmented form
    col_n = col_n - 2  # number of variables in the augmented form
    optimal_test = min(M_mat[0, range(1, col_n + 1)])

    if optimal_test < 0:
        pivot_col = np.argmin(M_mat[0, range(1, col_n + 1)]) + 1
    else:
        print(f'pass the optimal test')
        return 0
    ratio_list = np.divide(M_mat[range(1, row_n + 1), -1], M_mat[range(1, row_n + 1), pivot_col])

    pivot_row = min_ratio_test(ratio_list)
    if pivot_row is 0:
        print(f'no leaving variable, here is the ratio list {ratio_list}')
        return 0
    else:
        return pivot_row, pivot_col


# simplex solver
# input: augmented matrix
# output: print the updated augmented matrix
def simplex_solver(M_mat):
    print(f'{M_mat.shape[0] - 1} constraints and {M_mat.shape[1] - 2} variables')
    print(pd.DataFrame(M_mat))  # print augmented matrix

    pivot_n = new_pivot(M_mat)
    while pivot_n is not 0:
        print(f'new pivot is {pivot_n}')
        pivoting(M_mat, pivot_n[0], pivot_n[1])
        print(pd.DataFrame(M_mat))
        print(f'=======================')
        pivot_n = new_pivot(M_mat)


# test

# input initial tableau
M = np.array([
    [1, -3, -5, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 4],
    [0, 0, 2, 0, 1, 0, 12],
    [0, 3, 2, 0, 0, 1, 18]
], dtype=float)

simplex_solver(M)
