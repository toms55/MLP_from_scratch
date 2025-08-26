#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

// create and free space
double** create_matrix(int rows, int cols){
  double** matrix = malloc(rows * sizeof(double*));

  for (int i = 0; i < rows; i++) {
    matrix[i] = malloc(cols * sizeof(double));
  }

  return matrix;
}

void free_matrix(double** matrix, int rows) {
  for (int i = 0; i < rows; i++) {
    free(matrix[i]);
  }

  free(matrix);
}

// Operations
double** matrix_add(double** matrix1, double** matrix2, int rows, int cols) {
  double** added_matrix = create_matrix(rows, cols);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      added_matrix[i][j] = matrix1[i][j] + matrix2[i][j];
    }
  } 

  return added_matrix;
}

double** matrix_sub(double** A, double** B, int rows, int cols);
double** matrix_multiply(double** A, double** B, int a_rows, int a_cols, int b_cols);
double** matrix_transpose(double** A, int rows, int cols);

double** matrix_scalar_multiply(double** A, double scalar, int rows, int cols);
