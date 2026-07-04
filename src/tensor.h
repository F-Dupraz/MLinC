// This file is inspired in the micrograd.c from Jaykef and the tensor.h from apoorvnandan.
// This may look redundant with the mat files, but it a learning project so everything stays.

typedef struct tensor_s {
  float *values;
  int *shape;
  unsigned int ndim;
  unsigned int size;
} tensor;

tensor *new_ten(float *values, int *shape, int ndim);
void free_ten(tensor *ten);

tensor *add_ten(tensor *ten1, tensor *ten2);
tensor *sub_ten(tensor *ten1, tensor *ten2);
tensor *mul_ten(tensor *ten1, tensor *ten2);
tensor *scale_ten(float p, tensor *ten);
tensor *matmul_ten(tensor *ten1, tensor *ten2);  // Only works with matrices (ndim == 2)
tensor *transpose_ten(tensor *ten); // Only works with matrices (ndim == 2)
tensor *mean_ten(tensor *ten);
tensor *sum_ten(tensor *ten);
tensor *relu_ten(tensor *ten);
tensor *sigmoid_ten(tensor *ten);

void add_inplace_ten(tensor *des, tensor *src);
void sub_inplace_ten(tensor *des, tensor *src);
void mul_inplace_ten(tensor *des, tensor *src);
