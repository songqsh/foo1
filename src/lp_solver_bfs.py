# lp solver by bfs enumeration
# max z = c^T x
# s.j.
# Ax = b
# x\ge 0

import numpy as np
import numpy.linalg as la
import itertools

np.set_printoptions(suppress=True)


# A function solving for bfs
# inputs:
#   list_of_bv: an index list of basic variables
#   mat_A: 2d numpy array for matrix A
#   vec_b: 1d numpy array with the length equal to row number of A
# return:
#   either basic solution or error message
def bfs(list_of_bv, mat_A, vec_b):
    try:
        basic_solution = la.solve(mat_A[:, list_of_bv], vec_b)

        if min(basic_solution) < 0:
            raise (ValueError('Infeasible solution'))
        return basic_solution

    except la.LinAlgError as err:
        return err

    except ValueError as err:
        return err


# A function solving for bfs
# inputs:
#   mat_A: 2d numpy array for matrix A
#   vec_b: 1d numpy array with the length equal to row number of A
#   vec_c: 1d numpy array with the length equal to column number of A
# return:
#   [list of optimal bvs, optimal value]
def lp_solver(mat_A, vec_b, vec_c):
    m, n = mat_A.shape
    bvs = itertools.combinations(range(n), m)

    list_of_optimal_bvs = []
    optimal_z = 0

    for bv in bvs:
        res = bfs(bv, mat_A, vec_b)
        if type(res) is np.ndarray:
            z = np.dot(vec_c[bv, ], res)
            if z > optimal_z:
                optimal_z = z
                list_of_optimal_bvs = [bv]
            elif z == optimal_z:
                list_of_optimal_bvs += [bv]

    return list_of_optimal_bvs, optimal_z


# test
A = np.array([[2., 1., 1, 0, 0], [1., 1, 0, 1, 0], [1., 0, 0, 0, 1]])
b = np.array([100., 80, 40])
c = np.array([3, 2, 0., 0, 0])

opt = lp_solver(A, b, c)
print(f'optimal value is {opt[1]}')
for bv in opt[0]:
    res = bfs(bv, A, b)
    print(f'--- optimal bv is {bv}, and bfs is {res}')
