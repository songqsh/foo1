# simplex method in linear programing

# import package
import numpy as np
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
        return 0  # no leaving variable, this means optimum is unbounded
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
        print(f'The final tableau is \n {M_mat}')
        print(f'=================================')
        return 0

    ratio_list = np.divide(M_mat[range(1, row_n + 1), -1], M_mat[range(1, row_n + 1), pivot_col])

    pivot_row = min_ratio_test(ratio_list)
    if pivot_row is 0:
        print(f'no leaving variable, which means z is unbounded. Here is the ratio list:')
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
    n_constraints, n_decision_var = np.array(M_mat.shape) - [1, 2]
    print(f'{n_constraints} constraints and {n_decision_var} variables')
    # default pivot columns
    pivots_column_list = [n_decision_var - n_constraints + 1 + i for i in range(n_constraints)]
    print(f'pivot columns are: {pivots_column_list}')
    print(f'initial tableau is:')
    print(f'=======================')
    print(f'{M_mat}')  # print initial tableau

    pivot_n = new_pivot(M_mat)  # returns new pivot if not pass optimal test, otherwise return zero
    while pivot_n is not 0:
        pivots_column_list[pivot_n[0]-1] = pivot_n[1]
        print(f'pivot columns are: {pivots_column_list}')
        pivoting(M_mat, pivot_n[0], pivot_n[1])
        if display is not 0:
            print(f'new pivot is {pivot_n}')
            print(f'=======================')
            print(M_mat)

        pivot_n = new_pivot(M_mat)


def init_tab_standard_lp(A_mat, b_vec, c_vec):
    n = len(c_vec)  # number of decision variables
    m = len(A_mat)  # number of constraints
    row_0 = np.append([1], -1. * np.array(c_vec + [0] * m + [0])).reshape((1, n + m + 2))
    row_1 = np.append(np.zeros((m, 1)), A_mat, axis=1)
    row_1 = np.append(row_1, np.eye(m), axis=1)
    row_1 = np.append(row_1, np.array(b_vec).reshape((m, 1)), axis=1)
    return np.append(row_0, row_1, axis=0)


def main():
    # setup for WG
    c = [3., 5]
    A = [[1., 0], [0, 2], [3, 2]]
    b = [4, 12., 18]
    init_tab = init_tab_standard_lp(A, b, c)
    simplex_solver(init_tab, display=1)


# More test: simplex_v1.ipynb
if __name__ == "__main__":
    main()
