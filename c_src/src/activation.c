#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "activation.h"
#include "matrix.h"


double sigmoid(double x){
  double result = 1 / (1 + exp(-x));
  return result;
}

double sigmoid_derivative(double x) {
  double s = sigmoid(x);
  return s * (1 - s);
}

double** matrix_sigmoid(double** matrix, int rows, int cols){
  double** result = create_matrix(rows, cols);

  for(int i = 0; i < rows; ++i){
    for(int j = 0; j < cols; ++j){
      result[i][j] = sigmoid(matrix[i][j]);
    }
  }

  return result;
}

double** matrix_sigmoid_derivative(double** matrix, int rows, int cols){
  double** result = create_matrix(rows, cols);

  for(int i = 0; i < rows; ++i){
    for(int j = 0; j < cols; ++j){
      result[i][j] = sigmoid_derivative(matrix[i][j]);
    }
  }

  return result;
}
