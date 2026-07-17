#include "./nn.h"

#include <stdlib.h>
#include <time.h>
#include <stdio.h>

layer *new_layer(int in_s, int out_s, act_type act_t) {
  layer *l = malloc(sizeof(layer));
  if(l == NULL) {
    return NULL;
  }

  const int w_size = in_s * out_s;
  float *values = malloc(w_size * sizeof(float));
  if(values == NULL) {
    free(l);
    return NULL;
  }

  for(int i = 0; i < w_size; ++i) {
    values[i] = (float)rand() / RAND_MAX * 2 - 1;
  }

  int shape[] = {out_s, in_s};
  
  node *weights = new_node(values, shape, 2, NULL, 0, OP_LEAF);
  if(weights == NULL) {
    free(l);
    free(values);
    return NULL;
  }

  l->weights = weights;

  free(values);

  const int b_size = out_s;
  float *b_values = malloc(b_size * sizeof(float));
  if(b_values == NULL) {
    free(l);
    free_node(weights);
    return NULL;
  }
    
  for(int j = 0; j < b_size; ++j) {
    b_values[j] = (float)rand() / RAND_MAX * 2 - 1;
  }

  int b_shape[] = {out_s, 1};
  
  node *biases = new_node(b_values, b_shape, 2, NULL, 0, OP_LEAF);
  if(biases == NULL) {
    free(l);
    free(b_values);
    free_node(weights);
    return NULL;
  }

  l->biases = biases;

  free(b_values);

  l->in_s = in_s;
  l->out_s = out_s;
  l->act = act_t;
  
  return l;
}

void free_layer(layer *l) {
  free_node(l->weights);
  free_node(l->biases);
  free(l);
}
