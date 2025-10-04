#ifndef MATRIX_H
#define MATRIX_H

// create and free space
double** create_matrix(int rows, int cols);
void free_matrix(double** matrix, int rows);
double** create_zero_matrix(int rows, int cols);

// Operations
double** matrix_add(double** A, double** B, int rows, int cols);
double** matrix_sub(double** A, double** B, int rows, int cols);
double** matrix_multiply(double** A, double** B, int a_rows, int a_cols, int b_cols);
double** matrix_transpose(double** A, int rows, int cols);
double** matrix_hadamard(double** matrix1, double** matrix2, int matrix1_rows, int matrix2_cols);
double** matrix_scalar_multiply(double** A, double scalar, int rows, int cols);
double** add_weights_and_biases(double** weights, double** biases, int rows, int cols);
#endif
