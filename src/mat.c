#include <stdlib.h>
#include <stdio.h>

#include "./mat.h"

//---------- Matrix manipulation ----------//
mat *new_mat(unsigned int r, unsigned int c) {
  mat *m = malloc(sizeof(mat));
  m->rows = r;
  m->cols = c;
  m->data = calloc(r * c, sizeof(float));

  return m;
}

void free_mat(mat *matrix) {
  free(matrix->data);
  free(matrix);
  
  return;
}

void print_mat(mat *matrix) {
  for(unsigned int i = 0; i < matrix->rows; i += 1) {
    for(unsigned int j = 0; j < matrix->cols; j += 1) {
      printf(" %f", matrix->data[i*matrix->cols + j]);
    }
    printf("\n");
  }

  return;
}

void fill_mat(mat *matrix, float x) {
  for(unsigned int i = 0; i < matrix->rows; i += 1) {
    for(unsigned int j = 0; j < matrix->cols; j += 1) {
      matrix->data[i*matrix->cols + j] = x;
    }
  }

  return;
}

int copy_mat(mat *output, mat *matrix) {
  if(!eqdims_mat(output, matrix)) { return 0; }

  for(unsigned int i = 0; i < matrix->rows; i += 1) {
    for(unsigned int j = 0; j < matrix->cols; j += 1) {
      output->data[i*output->cols + j] = matrix->data[i*matrix->cols + j];
    }
  }

  return 1;
}

//---------- Matrix operations ----------//
mat *transpose(mat *matrix) {
  mat *matrixT = new_mat(matrix->cols, matrix->rows);
  unsigned int i, j;

  for(i = 0; i < matrixT->rows; ++i) {
    for(j = 0; j < matrixT->cols; ++j) {
      matrixT->data[i * matrixT->cols + j] = matrix->data[j * matrix->cols + i];
    }
  }

  return matrixT;
}

int eqdims_mat(mat *mat1, mat *mat2) {
  return (mat1->cols == mat2->cols && mat1->rows == mat2->rows);
}

int eq_mat(mat *mat1, mat *mat2) {
  if(!eqdims_mat(mat1, mat2)) { return 0; }

  for(unsigned int i = 0; i < mat1->rows; i += 1) {
    for(unsigned int j = 0; j < mat1->cols; j += 1) {
      if(mat1->data[i*mat1->cols + j] != mat2->data[i*mat2->cols + j]) { return 0; }
    }
  }

  return 1;
}

int add_mat(mat *output, mat *mat1, mat *mat2) {
  if(!eqdims_mat(mat1, mat2)) { return 0; }
  if(!eqdims_mat(mat1, output)) { return 0; }

  for(unsigned int i = 0; i < mat1->rows; i += 1) {
    for(unsigned int j = 0; j < mat1->cols; j += 1) {
      output->data[i*output->cols + j] = mat1->data[i*mat1->cols + j] + mat2->data[i*mat2->cols + j];
    }
  }

  return 1;
}

int sub_mat(mat *output, mat *mat1, mat *mat2) {
  if(!eqdims_mat(mat1, mat2)) { return 0; }
  if(!eqdims_mat(mat1, output)) { return 0; }

  for(unsigned int i = 0; i < mat1->rows; i += 1) {
    for(unsigned int j = 0; j < mat1->cols; j += 1) {
      output->data[i*output->cols + j] = mat1->data[i*mat1->cols + j] - mat2->data[i*mat2->cols + j];
    }
  }

  return 1;
}

int mul_mat(mat *output, mat *mat1, mat *mat2) {
  if(mat1->cols != mat2->rows) { return 0; }
  if(output->rows != mat1->rows || output->cols != mat2->cols) { return 0; }
  
  fill_mat(output, 0.0f);

  unsigned int i, j, k;
  for(i = 0; i < mat1->rows; ++i) {
    for(j = 0; j < mat2->cols; ++j) {
      for(k = 0; k < mat1->cols; ++k) {
        output->data[i * output->cols + j] += mat1->data[i * mat1->cols + k] * mat2->data[k * mat2->cols + j];
      } 
    }
  }

  return 1;
}

int scale_mat(mat *matrix, float scale) {
  for(unsigned int i = 0; i < matrix->rows; i += 1) {
    for(unsigned int j = 0; j < matrix->cols; j += 1) {
      matrix->data[i*matrix->cols + j] = matrix->data[i*matrix->cols + j] * scale;
    }
  }

  return 1;
}
