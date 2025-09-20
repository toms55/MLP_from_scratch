#include "loss.h"

double mean_squared_error(double* y_true, double* y_pred, int size){
  double MSE = 0.0;
  for (int i = 0; i < size; ++i) {
    double diff = (y_true[i] - y_pred[i]);
    MSE += diff * diff;
  }

  MSE = MSE / size;
  return MSE;
}
