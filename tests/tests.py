import sys
sys.path.append('./python')


from c_wrapper import (
    py_list_to_c_matrix,
    c_matrix_to_py_list,
    add_py_matrices,
    subtract_py_matrices,
    multiply_py_matrices,
    transpose_py_matrix,
    scalar_multiply_py_matrix,
    free_py_matrix,
    create_zero_py_matrix,
    py_sigmoid,
    py_sigmoid_derivative
)

if __name__ == "__main__":
    py_mat1 = [[1.2, 2.99], [3.1, 4.25]]
    py_mat2 = [[5.01, 6.91], [7.25, 8.765]]
    
    c_mat1 = None
    c_mat2 = None
    c_result = None
    c_zero_mat = None

    try:
        c_mat1 = py_list_to_c_matrix(py_mat1)
        c_mat2 = py_list_to_c_matrix(py_mat2)
        
        c_result = multiply_py_matrices(c_mat1, c_mat2)
        py_result = c_matrix_to_py_list(c_result)

        c_zero_mat = create_zero_py_matrix(3, 3)
        zero_mat = c_matrix_to_py_list(c_zero_mat)
        
        print("Matrix 1:")
        print(py_mat1)
        print("\nMatrix 2:")
        print(py_mat2)
        print("\nResult of matrix multiplication:")
        print(py_result)
        print("\nZero Matrix:")
        print(zero_mat)

        c_sum_mat = add_py_matrices(c_mat1, c_mat2)
        print("\nResult of matrix addition:")
        print(c_matrix_to_py_list(c_sum_mat))
        
        free_py_matrix(c_sum_mat)

        print("Testing Sigmoid")
        print(f"{py_sigmoid(3)} should equal 0.95257")

        print("Testing Sigmoid Derivative")
        print(f"{py_sigmoid_derivative(3)} should equal 0.04517665973")

    finally:
        if c_mat1:
            free_py_matrix(c_mat1)
        if c_mat2:
            free_py_matrix(c_mat2)
        if c_result:
            free_py_matrix(c_result)
        if c_zero_mat:
            free_py_matrix(c_zero_mat)
