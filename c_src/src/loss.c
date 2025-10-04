#include "loss.h"

double mean_squared_error(double** y_true, double** y_pred, int size){
  double MSE = 0.0;
  for (int i = 0; i < size; ++i) {
    double diff = (y_true[0][i] - y_pred[0][i]);
    MSE += diff * diff;
  }

  MSE = MSE / size;
  return MSE;
}

double mean_absolute_percentage_error(double** y_true, double** y_pred, int size){
  double MAPE = 0.0;
  for (int i = 0; i < size; ++i){
    double diff = (y_true[0][i] - y_pred[0][i]) / (y_true[0][i] + 1e-8);
    MAPE += diff;
  }

  MAPE = MAPE / size;
  return MAPE;
}
