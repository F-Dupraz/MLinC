#ifndef NN_H
#define NN_H

#include <stdlib.h>
#include <stdio.h>

#include "./node.h"

typedef enum { 
  ACT_NONE, 
  ACT_SIGMOID, 
  ACT_RELU 
} act_type;

typedef struct layer_s {
  node *weights;
  node *biases;
  int in_s;
  int out_s;
  act_type act;
} layer;

typedef struct nn_s {
  layer **layers;
  int layers_s;
} nn;

layer *new_layer(int in_s, int out_s, act_type act);
void free_layer(layer *l);
node **call_layer(layer *l, node *x);

nn *new_nn(int in_s, int *outs_s, int layers_s);
void free_nn(nn *nn);
node *call_nn(nn *nn, node *x);

#endif // NN_H
