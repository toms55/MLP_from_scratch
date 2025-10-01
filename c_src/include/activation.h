#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "activation.h"

double sigmoid(double x);
double sigmoid_derivative(double x);

double** matrix_sigmoid(double** matrix, int rows, int cols);
double** matrix_sigmoid_derivative(double** matrix, int rows, int cols);

double** matrix_relu(double** matrix, int rows, int cols);
double** matrix_relu_derivative(double** matrix, int rows, int cols);

#endif
