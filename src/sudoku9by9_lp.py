# solve sudoku 9 by 9 using linear programing
from scipy.optimize import linprog

# sudoku setup
sudoku_dim = 9
box_row_n = box_col_n = 3

n_decision_variables = sudoku_dim ** 3  # coefficient matrix A and b for Ax = b


# convert 3d to 1d
def conversion_3d_1d(i_row, i_col, i_num):
    return int(sudoku_dim ** 2 * i_row + sudoku_dim * i_col + i_num)


# convert a sudoku table of 9 by 9 array into 9^3 by 1 binary array
def conversion_tab_solution(sudoku_tab):
    sudoku_solution = []
    for i_row in range(sudoku_dim):
        for i_col in range(sudoku_dim):
            cell_binary = [0] * sudoku_dim
            if sudoku_tab[i_row][i_col] > 0:
                cell_binary[int(sudoku_tab[i_row][i_col] - 1)] = 1
            sudoku_solution += cell_binary
    return sudoku_solution


# convert a 9^3 by 1 sudoku solution to 9 by 9 table
def conversion_solution_tab(sudoku_solution):
    sudoku_table = []
    i_solution = 0
    for i_row in range(sudoku_dim):
        current_row = []
        for i_col in range(sudoku_dim):
            current_num = 0
            for i_num in range(sudoku_dim):
                if 1.1 > sudoku_solution[i_solution] > 0.9:
                    current_num = i_num + 1
                i_solution += 1
            current_row += [current_num]
        sudoku_table += [current_row]
    return sudoku_table


# produce a row of the matrix A by checking the specified cell
def cell_condition(i_row, i_col):
    row = [0.] * n_decision_variables  # initialize a row
    for i_num in range(sudoku_dim):
        ind1 = conversion_3d_1d(i_row, i_col, i_num)
        row[ind1] = 1.  # coefficient from the constraint
    return row


# produce a row of the matrix A by checking a specified column and a specified number
def column_condition(i_col, i_num):
    row = [0.] * n_decision_variables  # initialize a row
    for i_row in range(sudoku_dim):
        ind1 = conversion_3d_1d(i_row, i_col, i_num)
        row[ind1] = 1.  # coefficient from the constraint
    return row


# produce a row of the matrix A by checking a specified row and a specified number
def row_condition(i_row, i_num):
    row = [0.] * n_decision_variables  # initialize a row
    for i_col in range(sudoku_dim):
        ind1 = conversion_3d_1d(i_row, i_col, i_num)
        row[ind1] = 1.  # coefficient from the constraint
    return row


# produce a row of matrix A by checking a given box and a number
def box_condition(i_box, i_num):
    row = [0.] * n_decision_variables  # initialize a row
    i_start_row = int(i_box / box_col_n) * box_row_n
    i_start_col = int(i_box % box_col_n) * box_col_n
    for element in range(sudoku_dim):
        ind1 = conversion_3d_1d(i_start_row + int(element / box_col_n), i_start_col + int(element % box_col_n), i_num)
        row[ind1] = 1.
    return row


# Set up the vector c in the objective function
def objective_vector(sudoku_table):
    out = [0.] * n_decision_variables
    for i_row in range(sudoku_dim):
        for i_col in range(sudoku_dim):
            if sudoku_table[i_row][i_col] > 0:
                ind1 = conversion_3d_1d(i_row, i_col, sudoku_table[i_row][i_col] - 1)
                out[ind1] = -1.
    return out


mat_A = []
for i in range(sudoku_dim):
    for j in range(sudoku_dim):
        mat_A += [cell_condition(i, j)]
        mat_A += [column_condition(i, j)]
        mat_A += [row_condition(i, j)]
        mat_A += [box_condition(i, j)]

vec_b = [1.] * len(mat_A)

# An example of a sudoku table
tab = [[0, 5, 0, 9, 1, 0, 0, 0, 0],
       [0, 1, 0, 0, 3, 0, 5, 8, 0],
       [7, 4, 0, 0, 0, 0, 1, 2, 0],
       [4, 3, 0, 0, 0, 9, 0, 0, 7],
       [2, 0, 0, 0, 5, 8, 0, 0, 0],
       [9, 8, 1, 3, 0, 4, 2, 0, 5],
       [0, 0, 3, 0, 6, 5, 0, 7, 2],
       [0, 6, 7, 0, 0, 3, 0, 5, 1],
       [5, 0, 4, 0, 0, 0, 6, 0, 0]
       ]

vec_c = objective_vector(tab)

res = linprog(c=vec_c, A_eq=mat_A, b_eq=vec_b, bounds=(0, 1), options={"disp": False})
print(f'nit is {res.nit}')
print(f'optimal value is {res.fun}')
tab1 = conversion_solution_tab(res.x)
for i in range(sudoku_dim):
    print(tab1[i])
