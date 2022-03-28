# simplex method in linear programing

# import package
import numpy as np
import pandas as pd
import warnings

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})

# ignore by message
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")


# pivoting with (row_i,col_i)
def pivoting(A_mat, row_i, col_i):
    if A_mat[row_i, col_i] == 0:
        print(f'err msg: no pivoting due to division by zero')
        return 0
    A_mat[row_i] = A_mat[row_i] / A_mat[row_i, col_i]  # scale to get one in (row_i,col_i)
    n_rows, _ = A_mat.shape
    for k in range(n_rows):
        if k == row_i:
            continue  # skip i-row
        A_mat[k] = A_mat[k] - A_mat[row_i] * A_mat[k, col_i]  # replacement to get zero


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
    row_n, col_n = M_mat.shape - np.array([1, 2])  # number of constraints and number of variables
    optimal_test = min(M_mat[0, range(1, col_n + 1)])

    if optimal_test < 0:
        pivot_col = np.argmin(M_mat[0, range(1, col_n + 1)]) + 1
    else:
        print(f'=================================')
        print(f'pass the optimal test')
        print(f'optimal value is {M_mat[0, -1]}')
        print(f'The final tableau is \n {pd.DataFrame(M_mat)}')
        print(f'=================================')
        return 0

    ratio_list = np.divide(M_mat[range(1, row_n + 1), -1], M_mat[range(1, row_n + 1), pivot_col])

    pivot_row = min_ratio_test(ratio_list)
    if pivot_row is 0:
        print(f'no leaving variable, here is the ratio list:')
        print(f'{ratio_list}')
        return 0
    else:
        return pivot_row, pivot_col


# simplex solver
# input:
#   M_mat - initial tableau,
#   display - 0: no display of intermediate tableau, 1: print all intermediate tableau
# output: print the updated augmented matrix
def simplex_solver(M_mat, display=0):
    print(f'{M_mat.shape[0] - 1} constraints and {M_mat.shape[1] - 2} variables')
    print(f'initial tableau is:')
    print(f'=======================')
    print(pd.DataFrame(M_mat))  # print augmented matrix

    pivot_n = new_pivot(M_mat)  # returns new pivot if not pass optimal test, otherwise return zero
    while pivot_n is not 0:

        pivoting(M_mat, pivot_n[0], pivot_n[1])
        if display is not 0:
            print(f'new pivot is {pivot_n}')
            print(f'=======================')
            print(pd.DataFrame(M_mat))

        pivot_n = new_pivot(M_mat)

# Test: simplex_v1.ipynb
