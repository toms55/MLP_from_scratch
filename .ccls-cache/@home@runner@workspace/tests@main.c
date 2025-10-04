#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

void test_matrix_add() {
  int rows = 2, cols = 2;

  double** A = create_matrix(rows, cols);
  double** B = create_matrix(rows, cols);

  A[0][0] = 1; A[0][1] = 2;
  A[1][0] = 3; A[1][1] = 4;

  B[0][0] = 5; B[0][1] = 6;
  B[1][0] = 7; B[1][1] = 8;

  // Add them
  double** C = matrix_add(A, B, rows, cols);

  // Expected values: A + B = [[6, 8], [10, 12]]
  printf("Testing matrix_add...\n");
  printf("C[0][0] = %f (expected 6)\n", C[0][0]);
  printf("C[0][1] = %f (expected 8)\n", C[0][1]);
  printf("C[1][0] = %f (expected 10)\n", C[1][0]);
  printf("C[1][1] = %f (expected 12)\n", C[1][1]);

  // Clean up
  free_matrix(A, rows);
  free_matrix(B, rows);
  free_matrix(C, rows);
}

void test_matrix_multiply() {
  int rows1 = 2, cols1 = 3;
  int rows2 = 3, cols2 = 2;

  double** A = create_matrix(rows1, cols1);
  double** B = create_matrix(rows2, cols2);

  // A = [[1, 2, 3],
  //      [4, 5, 6]]
  A[0][0] = 1; A[0][1] = 2; A[0][2] = 3;
  A[1][0] = 4; A[1][1] = 5; A[1][2] = 6;

  // B = [[7, 8],
  //      [9, 10],
  //      [11, 12]]
  B[0][0] = 7;  B[0][1] = 8;
  B[1][0] = 9;  B[1][1] = 10;
  B[2][0] = 11; B[2][1] = 12;

  // Multiply them
  double** C = matrix_multiply(A, B, rows1, cols1, cols2);

  // Expected result:
  // [[58, 64],
  //  [139, 154]]
  printf("Testing matrix_multiply...\n");
  printf("C[0][0] = %f (expected 58)\n", C[0][0]);
  printf("C[0][1] = %f (expected 64)\n", C[0][1]);
  printf("C[1][0] = %f (expected 139)\n", C[1][0]);
  printf("C[1][1] = %f (expected 154)\n", C[1][1]);

  // Clean up
  free_matrix(A, rows1);
  free_matrix(B, rows2);
  free_matrix(C, rows1);
}
void test_matrix_sub() {
  int rows = 2, cols = 2;

  double** A = create_matrix(rows, cols);
  double** B = create_matrix(rows, cols);

  A[0][0] = 5; A[0][1] = 7;
  A[1][0] = 9; A[1][1] = 11;

  B[0][0] = 1; B[0][1] = 2;
  B[1][0] = 3; B[1][1] = 4;

  // Subtract them
  double** C = matrix_sub(A, B, rows, cols);

  // Expected values: A - B = [[4, 5], [6, 7]]
  printf("Testing matrix_sub...\n");
  printf("C[0][0] = %f (expected 4)\n", C[0][0]);
  printf("C[0][1] = %f (expected 5)\n", C[0][1]);
  printf("C[1][0] = %f (expected 6)\n", C[1][0]);
  printf("C[1][1] = %f (expected 7)\n", C[1][1]);

  // Clean up
  free_matrix(A, rows);
  free_matrix(B, rows);
  free_matrix(C, rows);
}

void test_matrix_transpose() {
  int rows = 2, cols = 3;

  double** A = create_matrix(rows, cols);

  // A = [[1, 2, 3],
  //      [4, 5, 6]]
  A[0][0] = 1; A[0][1] = 2; A[0][2] = 3;
  A[1][0] = 4; A[1][1] = 5; A[1][2] = 6;

  // Transpose it
  double** T = matrix_transpose(A, rows, cols);

  // Expected result:
  // [[1, 4],
  //  [2, 5],
  //  [3, 6]]
  printf("Testing matrix_transpose...\n");
  printf("T[0][0] = %f (expected 1)\n", T[0][0]);
  printf("T[0][1] = %f (expected 4)\n", T[0][1]);
  printf("T[1][0] = %f (expected 2)\n", T[1][0]);
  printf("T[1][1] = %f (expected 5)\n", T[1][1]);
  printf("T[2][0] = %f (expected 3)\n", T[2][0]);
  printf("T[2][1] = %f (expected 6)\n", T[2][1]);

  // Clean up
  free_matrix(A, rows);
  free_matrix(T, cols);
}

int main() {
  test_matrix_add();
  test_matrix_sub();
  test_matrix_multiply();
  test_matrix_transpose();
  return 0;
}

