#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "activation.h"

// Activation functions
double sigmoid(double x);
double sigmoid_derivative(double x);

double** matrix_sigmoid(double** matrix, int rows, int cols);
double** matrix_sigmoid_derivative(double** matrix, int rows, int cols);

double relu(double x);
double relu_derivative(double x);

double tanh_activation(double x);
double tanh_derivative(double x);

#endif
