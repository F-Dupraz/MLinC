#include "./nn.h"
#include "./node.h"
#include "./tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void zero_grad(nn* nn) {
  for(int i = 0; i < nn->layers_s; ++i) {
    for(int j = 0; j < nn->layers[i]->weights->grad->size; ++j) {
      nn->layers[i]->weights->grad->values[j] = 0.0f;
    }
    for(int j = 0; j < nn->layers[i]->biases->grad->size; ++j) {
      nn->layers[i]->biases->grad->values[j] = 0.0f;
    }
  }
}


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

  float limit;
  if (act_t == ACT_RELU) {
    limit = sqrtf(6.0f / in_s);            // He uniforme
  } else {
    limit = sqrtf(6.0f / (in_s + out_s));  // Xavier uniforme
  }

  for(int i = 0; i < w_size; ++i) {
    values[i] = limit * ((float)rand() / RAND_MAX * 2 - 1);
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

  int b_shape[] = {out_s, 1};
  
  node *biases = new_node(NULL, b_shape, 2, NULL, 0, OP_LEAF);
  if(biases == NULL) {
    free(l);
    free_node(weights);
    return NULL;
  }

  l->biases = biases;

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

node *call_layer(layer *l, node *x) {
  node *mul_w_x = matmul_node(l->weights, x);
  if (mul_w_x == NULL) {
    return NULL;
  }

  node *add_bias = add_node(mul_w_x, l->biases);
  if (add_bias == NULL) {
    return NULL;
  }

  switch (l->act) {
  case ACT_NONE:
    return add_bias;

  case ACT_SIGMOID: {
    node *output_s = sigmoid_node(add_bias);
    if (output_s == NULL) return NULL;
    return output_s;
  }
  case ACT_SOFTMAX: {
    node *output_sm = softmax_node(add_bias);
    if (output_sm == NULL) return NULL;
    return output_sm;
  }
  case ACT_RELU: {
    node *output_r = relu_node(add_bias);
    if (output_r == NULL) return NULL;
    return output_r;
  }

  default:
    return NULL;
  }
}

nn *new_nn(int in_s, int *outs_s, int layers_s, act_type *acts) {
  nn *new_nn = malloc(sizeof(nn));
  if(new_nn == NULL) {
    return NULL;
  }

  new_nn->layers_s = layers_s;

  new_nn->layers = malloc(new_nn->layers_s * sizeof(layer*));
  if(new_nn->layers == NULL) {
    free(new_nn);
    return NULL;
  }

  int c_in = in_s;

  for(int i = 0; i < layers_s; ++i) {
    new_nn->layers[i] = new_layer(c_in, outs_s[i], acts[i]);
    if(new_nn->layers[i] == NULL) {
      for(int j = 0; j < i; ++j) {
        free_layer(new_nn->layers[j]);
      }
      free(new_nn->layers);
      free(new_nn);
      return NULL;
    }

    c_in = outs_s[i];
  }

  return new_nn;
}

void free_nn(nn *nn) {
  if(nn == NULL) return;

  for(int i = 0; i < nn->layers_s; ++i) {
    free_layer(nn->layers[i]);
  }

  free(nn->layers);
  free(nn);

  return;
}

node *call_nn(nn *nn, node *x) {
  if(nn == NULL) return NULL;
  if(x == NULL) return NULL;

  node *res = x;
  for(int i = 0; i < nn->layers_s; ++i) {
    res = call_layer(nn->layers[i], res);
    if(res == NULL) return NULL;
  }

  return res;
}

node *mse(node *pred, node *y) {
  node *diff = sub_node(pred, y);
  node *sq = mul_node(diff, diff);
  node *loss = mean_node(sq);
  return loss;
}

node *cross_entropy(node *pred, node *y) {
  node *ce = cross_entropy_loss_node(pred, y);

  return ce;
}

void update(nn *nn, float lr) {
  for(int i = 0; i < nn->layers_s; ++i) {
    tensor *step_w = scale_ten(lr, nn->layers[i]->weights->grad);
    sub_inplace_ten(nn->layers[i]->weights->data, step_w);
    free_ten(step_w);
  
    tensor *step_b = scale_ten(lr, nn->layers[i]->biases->grad);
    sub_inplace_ten(nn->layers[i]->biases->data, step_b);
    free_ten(step_b);
  }

  return;
}

tensor *predict(nn *nn, tensor *x) {
  arena_t tape;
  init_arena(&tape);
  set_arena(&tape);

  node *input = new_node(x->values, x->shape, x->ndim, NULL, 0, OP_LEAF);
  if(input == NULL) {
    clear_arena(&tape);
    set_arena(NULL);
    return NULL;
  }

  node *output = call_nn(nn, input);
  if(output == NULL) {
    reset_arena(&tape);
    clear_arena(&tape);
    set_arena(NULL);
    free_node(input);
    return NULL;
  }

  tensor *result = new_ten(output->data->values, output->data->shape, output->data->ndim);

  reset_arena(&tape);
  clear_arena(&tape);
  set_arena(NULL);
  
  free_node(input);
  
  return result;
}

void train(nn *nn, tensor **xs, tensor **ys, int n, int epochs, float lr, loss_t lt, mnist_set *tr) {
  arena_t tape;
  init_arena(&tape);
  set_arena(&tape);

  for(int epoch = 0; epoch < epochs; epoch++) {
    float epoch_loss = 0.0f;

    for(int i = 0; i < n; ++i) {
      node *n_xs = new_node(xs[i]->values, xs[i]->shape, xs[i]->ndim, NULL, 0, OP_LEAF);
      node *output = call_nn(nn, n_xs);
      node *n_ys = new_node(ys[i]->values, ys[i]->shape, ys[i]->ndim, NULL, 0, OP_LEAF);
      node *loss;
      switch (lt) {
        case MSE:
          loss = mse(output, n_ys);
          break;
        case CROSS_ENTROPY:
          loss = cross_entropy(output, n_ys);
          break;
        default:
          loss = NULL;
          break;
      }
      epoch_loss += loss->data->values[0];

      zero_grad(nn);
      backward(loss);
      update(nn, lr);
      reset_arena(&tape);
      free_node(n_xs);
      free_node(n_ys);
    }

    shuffle_mnist(tr); 
   
    printf("Epoch: %d, loss prom: %f\n", epoch, epoch_loss/n);

    // if(epoch % 10 == 0) {
    // }
  }

  clear_arena(&tape);
  set_arena(NULL);

  return;
}

