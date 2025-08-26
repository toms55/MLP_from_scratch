#ifndef UTILS_H
#define UTILS_H

// Random initialization
double** random_matrix(int rows, int cols, double min, double max);
double* random_vector(int size, double min, double max);

// Helper for shuffling training data
void shuffle_data(double** X, double** y, int samples);

#endif
