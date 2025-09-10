#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "activation.h"

double sigmoid(double x){
  double result = 1 / (1 + exp(-x));
  return result;
}

double sigmoid_derivative(double x) {
  double s = sigmoid(x);
  return s * (1 - s);
}
