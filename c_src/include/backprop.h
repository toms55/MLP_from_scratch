#ifndef BACKPROP_H
#define BACKPROP_H

// Forward + backward step helpers
double** compute_layer_output(double** weights, double* inputs, double* biases,
                              int in_size, int out_size, double (*activation)(double));

double** compute_weight_gradients(double* inputs, double* deltas,
                                  int in_size, int out_size);

double* compute_deltas(double** weights, double* next_deltas, double* activations,
                       int in_size, int out_size, double (*activation_derivative)(double));

#endif
