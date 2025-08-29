import ctypes
import os

lib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'libmlp.so')

try:
    lib = ctypes.CDLL(lib_path)
except OSError as e:
    raise RuntimeError(f"Error loading shared library: {e}. "
                       "Make sure you have run 'make all' to build it.") from e

DoublePtr = ctypes.POINTER(ctypes.c_double)
DoublePtrPtr = ctypes.POINTER(DoublePtr)

# C: double** create_matrix(int rows, int cols)
lib.create_matrix.argtypes = [ctypes.c_int, ctypes.c_int]
lib.create_matrix.restype = DoublePtrPtr

# C: void free_matrix(double** matrix, int rows)
lib.free_matrix.argtypes = [DoublePtrPtr, ctypes.c_int]
lib.free_matrix.restype = None

# C: double** matrix_add(double** matrix1, double** matrix2, int rows, int cols)
lib.matrix_add.argtypes = [DoublePtrPtr, DoublePtrPtr, ctypes.c_int, ctypes.c_int]
lib.matrix_add.restype = DoublePtrPtr

# C: double** matrix_sub(double** matrix1, double** matrix2, int rows, int cols)
lib.matrix_sub.argtypes = [DoublePtrPtr, DoublePtrPtr, ctypes.c_int, ctypes.c_int]
lib.matrix_sub.restype = DoublePtrPtr

# C: double** matrix_multiply(double** matrix1, double** matrix2, int matrix1_rows, int matrix1_cols, int matrix2_cols)
lib.matrix_multiply.argtypes = [DoublePtrPtr, DoublePtrPtr, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.matrix_multiply.restype = DoublePtrPtr

# C: double** matrix_transpose(double** matrix, int rows, int cols)
lib.matrix_transpose.argtypes = [DoublePtrPtr, ctypes.c_int, ctypes.c_int]
lib.matrix_transpose.restype = DoublePtrPtr

# C: double** matrix_scalar_multiply(double** matrix, double scalar, int rows, int cols)
lib.matrix_scalar_multiply.argtypes = [DoublePtrPtr, ctypes.c_double, ctypes.c_int, ctypes.c_int]
lib.matrix_scalar_multiply.restype = DoublePtrPtr

def py_list_to_c_matrix(py_matrix):
    if not isinstance(py_matrix, list) or not all(isinstance(row, list) for row in py_matrix):
        raise TypeError("Input must be a list of lists.")
    
    rows = len(py_matrix)
    if rows == 0:
        return None, 0, 0
    cols = len(py_matrix[0])
    
    c_matrix = lib.create_matrix(rows, cols)
    
    for i in range(rows):
        for j in range(cols):
            c_matrix[i][j] = py_matrix[i][j]
            
    return c_matrix, rows, cols

def c_matrix_to_py_list(c_matrix, rows, cols):
    if not c_matrix:
        return []
    
    py_matrix = []
    for i in range(rows):
        row_list = []
        for j in range(cols):
            row_list.append(c_matrix[i][j])
        py_matrix.append(row_list)
        
    return py_matrix

def create_py_matrix(rows, cols):
    return lib.create_matrix(rows, cols)

def add_matrices(mat1, rows1, cols1, mat2, rows2, cols2):
    # Add validation if sizes don't match
    return lib.matrix_add(mat1, mat2, rows1, cols1)

def subtract_matrices(mat1, rows1, cols1, mat2, rows2, cols2):
    return lib.matrix_sub(mat1, mat2, rows1, cols1)

def multiply_matrices(mat1, mat1_rows, mat1_cols, mat2, mat2_rows, mat2_cols):
    # Add validation if dimensions don't match
    return lib.matrix_multiply(mat1, mat2, mat1_rows, mat1_cols, mat2_cols)

def transpose_matrix(mat, rows, cols):
    return lib.matrix_transpose(mat, rows, cols)

def scalar_multiply_matrix(mat, scalar, rows, cols):
    # Note: This function modifies the matrix in place
    lib.matrix_scalar_multiply(mat, scalar, rows, cols)

def free_py_matrix(mat, rows):
    lib.free_matrix(mat, rows)
