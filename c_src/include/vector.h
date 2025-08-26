#ifndef VECTOR_H
#define VECTOR_H

double* vector_add(double* v1, double* v2, int size);
double* vector_sub(double* v1, double* v2, int size);
double dot_product(double* v1, double* v2, int size);

double* vector_scalar_mul(double* v, double scalar, int size);

#endif
