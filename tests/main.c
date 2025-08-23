#include <stdio.h>
#include "matrix.h"

int main() {
    int rows = 2, cols = 3;
    double** mat = create_matrix(rows, cols);

    // put something in it
    mat[0][0] = 1.0;
    mat[0][1] = 2.0;
    mat[1][2] = 3.0;

    printf("mat[0][1] = %f\n", mat[0][1]); // expect 2.0

    free_matrix(mat, rows);
    
    return 0;
}
