import ctypes
import os
import numpy as np

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

# C: double** create_zero_matrix(int rows, int cols)
lib.create_zero_matrix.argtypes = [ctypes.c_int, ctypes.c_int]
lib.create_zero_matrix.restype = DoublePtrPtr

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

# C: double** sigmoid(double x)
lib.sigmoid.argtypes = [ctypes.c_double]
lib.sigmoid.restype = ctypes.c_double

lib.sigmoid_derivative.argtypes = [ctypes.c_double]
lib.sigmoid_derivative.restype = ctypes.c_double

class Matrix:
    def __init__(self, c_ptr, rows, cols):
        self.c_ptr = c_ptr
        self.rows = rows
        self.cols = cols

    def __repr__(self):
        return f"Matrix({self.rows}*{self.cols})"

def py_list_to_c_matrix(py_matrix):
   rows = len(py_matrix)
   cols = len(py_matrix[0]) if rows > 0 else 0
   c_matrix = lib.create_matrix(rows, cols)

   for row in range(rows):
       for col in range(cols):
           c_matrix[row][col] = py_matrix[row][col]
   return Matrix(c_matrix, rows, cols)

def c_matrix_to_py_list(c_matrix):
    if not isinstance(c_matrix, Matrix):
        raise TypeError("Did not pass a Matrix to c_matrix_to_py_list")
    if not c_matrix.c_ptr:
        return []
    
    py_matrix = []
    for i in range(c_matrix.rows):
        row_list = []
        for j in range(c_matrix.cols):
            row_list.append(c_matrix.c_ptr[i][j])
        py_matrix.append(row_list)
        
    return py_matrix


def create_py_matrix(rows, cols):
    c_ptr = lib.create_matrix(rows, cols)
    return Matrix(c_ptr, rows, cols)

def create_zero_py_matrix(rows, cols):
    c_ptr = lib.create_zero_matrix(rows, cols)
    return Matrix(c_ptr, rows, cols)

def add_py_matrices(mat1, mat2):
    if mat1.rows != mat2.rows or mat1.cols != mat2.cols:
        raise ValueError("Matrices must have the same dimensions to be added.")
    c_result_ptr = lib.matrix_add(mat1.c_ptr, mat2.c_ptr, mat1.rows, mat1.cols)
    return Matrix(c_result_ptr, mat1.rows, mat1.cols)

def subtract_py_matrices(mat1, mat2):
    if mat1.rows != mat2.rows or mat1.cols != mat2.cols:
        raise ValueError("Matrices must have the same dimensions to be subtracted.")
    c_result_ptr = lib.matrix_sub(mat1.c_ptr, mat2.c_ptr, mat1.rows, mat1.cols)
    return Matrix(c_result_ptr, mat1.rows, mat1.cols)

def multiply_py_matrices(mat1, mat2):
    if mat1.cols != mat2.rows:
        raise ValueError("Matrix dimensions are not compatible for multiplication.")
    c_result_ptr = lib.matrix_multiply(mat1.c_ptr, mat2.c_ptr, mat1.rows, mat1.cols, mat2.cols)
    return Matrix(c_result_ptr, mat1.rows, mat2.cols)

def transpose_py_matrix(mat):
    c_result_ptr = lib.matrix_transpose(mat.c_ptr, mat.rows, mat.cols)
    return Matrix(c_result_ptr, mat.cols, mat.rows)

def scalar_multiply_py_matrix(mat, scalar):
    lib.matrix_scalar_multiply(mat.c_ptr, scalar, mat.rows, mat.cols)

def free_py_matrix(mat):
    lib.free_matrix(mat.c_ptr, mat.rows)
    
def py_sigmoid(x):
    return lib.sigmoid(x)

def py_sigmoid_derivative(x):
    return lib.sigmoid_derivative(x)

def from_numpy(np_array: np.ndarray) -> Matrix:
    if not isinstance(np_array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    
    # Ensure the data type is float64 (c_double)
    np_array = np_array.astype(np.float64)

    rows, cols = np_array.shape
    c_mat_ptr = lib.create_matrix(rows, cols)

    for r in range(rows):
        for c in range(cols):
            c_mat_ptr[r][c] = np_array[r, c]

    return Matrix(c_mat_ptr, rows, cols)

def to_numpy(c_matrix: Matrix) -> np.ndarray:
    if not isinstance(c_matrix, Matrix):
        raise TypeError("Input must be a Matrix object.")

    np_array = np.empty((c_matrix.rows, c_matrix.cols), dtype=np.float64)
    
    for r in range(c_matrix.rows):
        for c in range(c_matrix.cols):
            np_array[r, c] = c_matrix.c_ptr[r][c]
            
    return np_array
