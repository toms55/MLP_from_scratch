#ifndef LOSS_H
#define LOSS_H

// Loss functions
double mean_squared_error(double* y_true, double* y_pred, int size);
double cross_entropy_loss(double* y_true, double* y_pred, int size);

// Derivatives
double* mse_derivative(double* y_true, double* y_pred, int size);
double* cross_entropy_derivative(double* y_true, double* y_pred, int size);

#endif
