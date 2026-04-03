#ifndef MAT_H
#define MAT_H

typedef struct mat_s {
  unsigned int rows;
  unsigned int cols;
  float *data;
} mat;

//---------- Matrix manipulation ----------//
mat *new_mat(unsigned int r, unsigned int c);
void free_mat(mat *matrix);
void print_mat(mat *matrix);
void fill_mat(mat *matrix, float x);
int copy_mat(mat *mat1, mat *mat2);

//---------- Matrix operations ----------//
mat *transpose(mat *matrix);
int eqdims_mat(mat *mat1, mat *mat2);
int eq_mat(mat *mat1, mat *mat2);
int add_mat(mat *output, mat *mat1, mat *mat2);
int sub_mat(mat *output, mat *mat1, mat *mat2);
int mul_mat(mat *output, mat *mat1, mat *mat2);
int scale_mat(mat *matrix, float scale);

#endif //MAT_H
