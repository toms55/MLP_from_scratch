from c_wrapper import (
    py_list_to_c_matrix,
    c_matrix_to_py_list,
    add_matrices,
    multiply_matrices,
    free_py_matrix,
)

if __name__ == "__main__":
    py_mat1 = [[1.0, 2.0], [3.0, 4.0]]
    py_mat2 = [[5.0, 6.0], [7.0, 8.0]]
    
    c_mat1, mat1_rows, mat1_cols = py_list_to_c_matrix(py_mat1)
    c_mat2, mat2_rows, mat2_cols = py_list_to_c_matrix(py_mat2)

    try:
        c_result = multiply_matrices(c_mat1, mat1_rows, mat1_cols, c_mat2, mat2_rows, mat2_cols)
        
        result_rows = mat1_rows
        result_cols = mat2_cols
        py_result = c_matrix_to_py_list(c_result, result_rows, result_cols)
        
        print("Matrix 1:")
        print(py_mat1)
        print("\nMatrix 2:")
        print(py_mat2)
        print("\nResult of matrix multiplication (from C):")
        print(py_result)

    finally:
        free_py_matrix(c_mat1, mat1_rows)
        free_py_matrix(c_mat2, mat2_rows)
        free_py_matrix(c_result, result_rows)
