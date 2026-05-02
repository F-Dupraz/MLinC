#ifndef NN_H
#define NN_H

#include <stdlib.h>
#include <stdio.h>

#include "./mat.h"

//---------- Defining the neural network ----------//
typedef struct nn_t {
  // Layer 1
  // Weights
  mat *w1;
  // Biases
  mat *b1;

  // Layer 2 (output)
  // Weights
  mat *w2;
  // Biases
  mat *b2;

  // Size of inputs, hidden and outputs
  size_t in_s, hid_s, out_s;

  // Forward pass caches
  mat *a1, *z1, *a2, *z2;
} nn_t;

nn_t *new_nn(size_t in_s, size_t hid_s, size_t out_s);
nn_t *read_nn(FILE *nn_file);
void free_nn(nn_t *nn);
void print_output_nn(nn_t const *nn);
int randomize_nn(nn_t *nn);
int forward_nn(nn_t *nn, mat *x);
int write_nn(nn_t const *nn, FILE *nn_file);

#endif //NN_H
