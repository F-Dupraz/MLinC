#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "./tensor.h"

tensor *new_ten(float *values, int *shape, int ndim) {
  tensor *ten = malloc(sizeof(tensor));
  if(ten == NULL) { return NULL; }

  ten->ndim = ndim;
  
  int counter = 1;
  for(int i = 0; i < ndim; ++i) {
    counter = counter * shape[i];
  }
  ten->size = counter;

  ten->shape = malloc(ndim * sizeof(int));
  if(ten->shape == NULL) {
    free(ten->shape);
    free(ten);
    return NULL;
  }
  memcpy(ten->shape, shape, ndim * sizeof(int));

  ten->values = calloc(ten->size, sizeof(float));
  if(ten->values == NULL) {
    free(ten->shape);
    free(ten);
    return NULL;
  }
  if(values != NULL) {
    memcpy(ten->values, values, ten->size * sizeof(float));
  }
  return ten;
}

void free_ten(tensor *ten) {
  if(ten == NULL) return;

  free(ten->values);
  free(ten->shape);
  free(ten);

  return;
}

// ---------- Operations ---------- //

tensor *add_ten(tensor *ten1, tensor *ten2) {
  if(ten1->size != ten2->size) {
    return NULL;
  } else if(ten1->ndim != ten2->ndim) {
    return NULL;
  }

  for(unsigned int i = 0; i < ten1->ndim; ++i) {
    if(ten1->shape[i] != ten2->shape[i]) {
      return NULL;
    }
  }

  tensor *res = new_ten(NULL, ten1->shape, ten1->ndim);
  if(res == NULL) return NULL;

  for(unsigned int i = 0; i < ten1->size; ++i) {
    res->values[i] = ten1->values[i] + ten2->values[i];
  }

  return res;
}

tensor *sub_ten(tensor *ten1, tensor *ten2) {
  if(ten1->size != ten2->size) {
    return NULL;
  } else if(ten1->ndim != ten2->ndim) {
    return NULL;
  }

  for(unsigned int i = 0; i < ten1->ndim; ++i) {
    if(ten1->shape[i] != ten2->shape[i]) {
      return NULL;
    }
  }

  tensor *res = new_ten(NULL, ten1->shape, ten1->ndim);
  if(res == NULL) return NULL;

  for(unsigned int i = 0; i < ten1->size; ++i) {
    res->values[i] = ten1->values[i] - ten2->values[i];
  }

  return res;
}

tensor *scale_ten(float p, tensor *ten) {
  tensor *res = new_ten(NULL, ten->shape, ten->ndim);
  if(res == NULL) return NULL;

  for(unsigned int i = 0; i < ten->size; ++i) {
    res->values[i] = p * ten->values[i];
  }

  return res;
}

tensor *mul_ten(tensor *ten1, tensor *ten2) {
  if(ten1->size != ten2->size) {
    return NULL;
  } else if(ten1->ndim != ten2->ndim) {
    return NULL;
  }

  for(unsigned int i = 0; i < ten1->ndim; ++i) {
    if(ten1->shape[i] != ten2->shape[i]) {
      return NULL;
    }
  }

  tensor *res = new_ten(NULL, ten1->shape, ten1->ndim);
  if(res == NULL) return NULL;

  for(unsigned int i = 0; i < ten1->size; ++i) {
    res->values[i] = ten1->values[i] * ten2->values[i];
  }

  return res;
}

tensor *matmul_ten(tensor *ten1, tensor *ten2) {
  if(ten1->ndim != ten2->ndim || ten1->ndim != 2) {
    return NULL;
  } else if(ten1->shape[1] != ten2->shape[0]) {
    return NULL;
  }

  tensor *res = new_ten(NULL, (int[]){ten1->shape[0], ten2->shape[1]}, 2);
  if(res == NULL) return NULL;

  for(unsigned int i = 0; i < ten1->shape[0]; ++i) {
    for(unsigned int j = 0; j < ten2->shape[1]; ++j) {
      for(unsigned int k = 0; k < ten1->shape[1]; ++k) {
        res->values[i * res->shape[1] + j] += ten1->values[i * ten1->shape[1] + k] * ten2->values[k * ten2->shape[1] + j];
      }
    }
  }

  return res;
}

tensor *transpose_ten(tensor *ten) {
  if(ten->ndim != 2) {
    return NULL;
  }

  tensor *res = new_ten(NULL, (int[]){ten->shape[1], ten->shape[0]}, 2);
  if(res == NULL) return NULL;

  for(unsigned int i = 0; i < ten->shape[0]; ++i) {
    for(unsigned int j = 0; j < ten->shape[1]; ++j) {
      res->values[j * res->shape[1] + i] = ten->values[i * ten->shape[1] + j];
    }
  }

  return res;
}

tensor *mean_ten(tensor *ten) {
  tensor *res = new_ten(NULL, (int[]){1}, 1);
  if(res == NULL) return NULL;
  res->values[0] = 0.0f;

  for(unsigned int i = 0; i < ten->size; ++i) {
    res->values[0] += ten->values[i];
  }

  res->values[0] = res->values[0]/ten->size;

  return res;
}

tensor *sum_ten(tensor *ten) {
  tensor *res = new_ten(NULL, (int[]){1}, 1);
  if(res == NULL) return NULL;
  res->values[0] = 0.0f;

  for(unsigned int i = 0; i < ten->size; ++i) {
    res->values[0] += ten->values[i];
  }

  return res;
}

tensor *relu_ten(tensor *ten) {
  tensor *res = new_ten(NULL, ten->shape, ten->ndim);
  if(res == NULL) return NULL;

  for(unsigned int i = 0; i < ten->size; ++i) {
    if(ten->values[i] > 0) {
      res->values[i] = ten->values[i];
    } else {
      res->values[i] = 0;
    }
  }

  return res;
}

tensor *sigmoid_ten(tensor *ten) {
  tensor *res = new_ten(NULL, ten->shape, ten->ndim);
  if(res == NULL) return NULL;

  for(unsigned int i = 0; i < ten->size; ++i) {
    res->values[i] = 1.0f / (1.0f + expf(-ten->values[i]));
  }

  return res;
}

void add_inplace_ten(tensor *des, tensor *src) {
  if(des->ndim != src->ndim || des->size != src->size) return;

  for(unsigned int i = 0; i < des->size; ++i) {
    des->values[i] += src->values[i];
  }
}

void sub_inplace_ten(tensor *des, tensor *src) {
  if(des->ndim != src->ndim || des->size != src->size) return;

  for(unsigned int i = 0; i < des->size; ++i) {
    des->values[i] -= src->values[i];
  }
}

void mul_inplace_ten(tensor *des, tensor *src) {
  if(des->ndim != src->ndim || des->size != src->size) return;

  for(unsigned int i = 0; i < des->size; ++i) {
    des->values[i] = des->values[i] * src->values[i];
  }
}
