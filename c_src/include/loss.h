#ifndef LOSS_H
#define LOSS_H

double mean_squared_error(double* y_true, double* y_pred, int size);
double mean_absolute_percentage_error(double* y_true, double* y_pred, int size);

#endif
