#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

// create and free space
double** create_matrix(int rows, int cols){
  double** matrix = malloc(rows * sizeof(double*));

  for (int i = 0; i < rows; ++i) {
    matrix[i] = malloc(cols * sizeof(double));
  }

  return matrix;
}

void free_matrix(double** matrix, int rows){
  for (int i = 0; i < rows; ++i) {
    free(matrix[i]);
  }

  free(matrix);
}

double** create_zero_matrix(int rows, int cols){
  double** zero_matrix = create_matrix(rows, cols);

  for (int i = 0; i < rows; ++i){
    for (int j = 0; j < cols; ++j){
      zero_matrix[i][j] = 0;
    }
  }

  return zero_matrix;
}

// Operations
double** matrix_add(double** matrix1, double** matrix2, int rows, int cols){
  double** added_matrix = create_matrix(rows, cols);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      added_matrix[i][j] = matrix1[i][j] + matrix2[i][j];
    }
  } 

  return added_matrix;
}

double** matrix_sub(double** matrix1, double** matrix2, int rows, int cols){
  double** subbed_matrix = create_matrix(rows, cols);

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        subbed_matrix[i][j] = matrix1[i][j] - matrix2[i][j];
      }
    } 

  return subbed_matrix;
}

double** matrix_multiply(double** matrix1, double** matrix2, int matrix1_rows, int matrix1_cols, int matrix2_cols){
  double** multiplied_matrix = create_matrix(matrix1_rows, matrix2_cols);

  for (int i = 0; i < matrix1_rows; ++i) {
    for (int j = 0; j < matrix2_cols; ++j) {
      double sum = 0.0;

      for (int k = 0; k < matrix1_cols; ++k) {
        sum += matrix1[i][k] * matrix2[k][j];
      }
      multiplied_matrix[i][j] = sum;
    }
  }

  return multiplied_matrix;
}

double** matrix_hadamard(double** matrix1, double** matrix2, int matrix1_rows, int matrix2_cols){
  double** hadamard_matrix = create_matrix(matrix1_rows, matrix2_cols);

  for (int i = 0; i < matrix1_rows; ++i) {
    for (int j = 0; j < matrix2_cols; ++j) {
      hadamard_matrix[i][j] = matrix1[i][j] * matrix2[i][j];
    }  
  }

  return hadamard_matrix;
}

double** matrix_transpose(double** matrix, int rows, int cols){
  double** transposed_matrix = create_matrix(cols, rows);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      transposed_matrix[j][i] = matrix[i][j];
    }
  }

  return transposed_matrix;
}

double** matrix_scalar_multiply(double** matrix, double scalar, int rows, int cols){
  double** multiplied_matrix = create_matrix(rows, cols);

  for (int i = 0; i < rows; ++i){
    for (int j = 0; j < cols; ++j){
      multiplied_matrix[i][j] = matrix[i][j] * scalar;
    } 
  }
  return multiplied_matrix;
}

double* sum_matrix_columns(double** matrix, int rows, int cols){
  double* summed_vector = (double*)malloc(rows * sizeof(double));

  for (int i = 0; i < rows; ++i){
    double column_sum = 0;
    for (int j = 0; j < cols; ++j){
      column_sum += matrix[i][j];
    }
    summed_vector[i] = column_sum;
  }

  return summed_vector;
}


double** add_weights_and_biases(double** weights, double** biases, int rows, int cols){
  double** result = create_matrix(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      result[i][j] = weights[i][j] + biases[i][0];
    }
  }
  return result;
}
