#include <stdlib.h>
#include <string.h>

#include "./node.h"
#include "tensor.h"

static arena_t *current_arena = NULL;

void set_arena(arena_t *a) {
  current_arena = a;
}

void init_arena(arena_t *a) {
  a->size = 0;
  a->capacity = 10;
  a->nodes = malloc(10 * sizeof(node*));

  return;
}

void push_arena(arena_t *a, node *n) {
  a->size += 1;

  if(a->size > a->capacity) {
    a->capacity *= 2;
    node **new_nodes = malloc(a->capacity * sizeof(node*));
    memcpy(new_nodes, a->nodes, a->capacity/2 * sizeof(node*));
    node **old_nodes = a->nodes;
    a->nodes = new_nodes;
    free(old_nodes);
  }

  a->nodes[a->size-1] = n;

  return;
}

void reset_arena(arena_t *a) {
  for (int i = 0; i < a->size; ++i)
    free_node(a->nodes[i]);
  a->size = 0;
}

void clear_arena(arena_t *a) {
  for(int i = 0; i < a->size; ++i) {
    free_node(a->nodes[i]);
  }

  free(a->nodes);
  
  return;
}

node *new_node(float *values, int *shape, int ndim, node **children, int n_child, op_type op) {
  tensor *d = new_ten(values, shape, ndim);
  if (d == NULL) return NULL;
  tensor *g = new_ten(NULL, shape, ndim);
  if (g == NULL) {
    free_ten(d);
    return NULL;
  }

  node *n = malloc(sizeof(node));
  if(n == NULL) {
    free_ten(d);
    free_ten(g);
    return NULL;
  }

  n->data = d;
  n->grad = g;
  n->op = op;
  n->scalar = 0.0f;
  n->is_topo = 0;
  n->n_prevs = n_child;
 
  if (n_child > 0) {
    n->prevs = malloc(n_child * sizeof(node*));
    if (n->prevs == NULL) {
      free_ten(d);
      free_ten(g);
      free(n);
      return NULL;
    }
    memcpy(n->prevs, children, n_child * sizeof(node*));
  } else {
    n->prevs = NULL;
  }

  if(current_arena != NULL && op != OP_LEAF) {
    push_arena(current_arena, n);
  }

  return n;
}

void free_node(node *n) {
  if(n == NULL) return;

  free_ten(n->data);
  free_ten(n->grad);

  free(n->prevs);

  free(n);

  return;
}

node *add_node(node *n1, node *n2) {
  node *children[] = {n1, n2};
  tensor *sum = add_ten(n1->data, n2->data);
  if(sum == NULL) return NULL;

  node *new_n = new_node(sum->values, sum->shape, sum->ndim, children, 2, OP_ADD);
  
  free_ten(sum);

  return new_n;
}

void add_node_back(node *n) {
  add_inplace_ten(n->prevs[0]->grad, n->grad);
  add_inplace_ten(n->prevs[1]->grad, n->grad);
  
  return;
}

node *sub_node(node *n1, node *n2) {
  node *children[] = {n1, n2};
  tensor *sub = sub_ten(n1->data, n2->data);
  if(sub == NULL) return NULL;

  node *new_n = new_node(sub->values, sub->shape, sub->ndim, children, 2, OP_SUB);

  free_ten(sub);

  return new_n;
}

void sub_node_back(node *n) {
  add_inplace_ten(n->prevs[0]->grad, n->grad);
  sub_inplace_ten(n->prevs[1]->grad, n->grad);

  return;
}

node *mul_node(node *n1, node *n2) {
  node *children[] = {n1, n2};
  tensor *mul = mul_ten(n1->data, n2->data);
  if(mul == NULL) return NULL;

  node *new_n = new_node(mul->values, mul->shape, mul->ndim, children, 2, OP_MUL);

  free_ten(mul);

  return new_n;
}

void mul_node_back(node *n) {
  tensor *nn1 = mul_ten(n->grad, n->prevs[1]->data);
  tensor *nn2 = mul_ten(n->grad, n->prevs[0]->data);

  add_inplace_ten(n->prevs[0]->grad, nn1);
  add_inplace_ten(n->prevs[1]->grad, nn2);

  free_ten(nn1);
  free_ten(nn2);

  return;
}

node *scale_node(float p, node *n) {
  node *children[] = {n};
  tensor *sc = scale_ten(p, n->data);
  if(sc == NULL) return NULL;

  node *new_n = new_node(sc->values, sc->shape, sc->ndim, children, 1, OP_SCALE);
  if(new_n == NULL) {
    free_ten(sc);
    return NULL;
  }
  new_n->scalar = p;

  free_ten(sc);

  return new_n;
}

void scale_node_back(node *n) {
  for(unsigned int i = 0; i < n->prevs[0]->grad->size; ++i) {
    n->prevs[0]->grad->values[i] += n->grad->values[i] * n->scalar;
  }

  return;
}

node *matmul_node(node *n1, node *n2) {
  node *children[] = {n1, n2};
  tensor *mm = matmul_ten(n1->data, n2->data);
  if(mm == NULL) return NULL;

  node *new_n = new_node(mm->values, mm->shape, mm->ndim, children, 2, OP_MATMUL);

  free_ten(mm);

  return new_n;
}

void matmul_node_back(node *n) {
  tensor *at = transpose_ten(n->prevs[0]->data);
  tensor *bt = transpose_ten(n->prevs[1]->data);

  tensor *dlda = matmul_ten(n->grad, bt);
  tensor *dldb = matmul_ten(at, n->grad);

  for(unsigned int i = 0; i < n->prevs[0]->grad->size; ++i) {
    n->prevs[0]->grad->values[i] += dlda->values[i];
  }

  for(unsigned int j = 0; j < n->prevs[1]->grad->size; ++j) {
    n->prevs[1]->grad->values[j] += dldb->values[j];
  }

  free_ten(at);
  free_ten(bt);
  free_ten(dlda);
  free_ten(dldb);
  
  return;
}

node *transpose_node(node *n) {
  node *children[] = {n};
  tensor *tr = transpose_ten(n->data);
  if(tr == NULL) return NULL;

  node *new_n = new_node(tr->values, tr->shape, tr->ndim, children, 1, OP_TRANSPOSE);

  free_ten(tr);

  return new_n;
}

void transpose_node_back(node *n) {
  tensor *t = transpose_ten(n->grad);

  for(unsigned int i = 0; i < n->prevs[0]->grad->size; ++i) {
    n->prevs[0]->grad->values[i] += t->values[i];
  }

  free_ten(t);

  return;
}

node *mean_node(node *n) {
  node *children[] = {n};
  tensor *m = mean_ten(n->data);
  if(m == NULL) return NULL;

  node *new_n = new_node(m->values, m->shape, m->ndim, children, 1, OP_MEAN);

  free_ten(m);

  return new_n;
}

void mean_node_back(node *n) {
  for(unsigned int i = 0; i < n->prevs[0]->grad->size; ++i) {
    n->prevs[0]->grad->values[i] += n->grad->values[0] / n->prevs[0]->data->size;
  }

  return;
}

node *sum_node(node *n) {
  node *children[] = {n};
  tensor *s = sum_ten(n->data);
  if(s == NULL) return NULL;

  node * new_n = new_node(s->values, s->shape, s->ndim, children, 1, OP_SUM);
  
  free_ten(s);

  return new_n;
}
  
void sum_node_back(node *n) {
  for(unsigned int i = 0; i < n->prevs[0]->grad->size; ++i) {
    n->prevs[0]->grad->values[i] += n->grad->values[0];
  }
}

node *relu_node(node *n) {
  node *children[] = {n};
  tensor *r = relu_ten(n->data);
  if(r == NULL) return NULL;

  node *new_n = new_node(r->values, r->shape, r->ndim, children, 1, OP_RELU);

  free_ten(r);

  return new_n;
}

void relu_node_back(node *n) {
  for(unsigned int i = 0; i < n->prevs[0]->grad->size; ++i) {
    if(n->prevs[0]->data->values[i] > 0) {
      n->prevs[0]->grad->values[i] += n->grad->values[i];
    }
  }

  return;
}

node *sigmoid_node(node *n) {
  node *children[] = {n};
  tensor *s = sigmoid_ten(n->data);
  if(s == NULL) return NULL;

  node *new_n = new_node(s->values, s->shape, s->ndim, children, 1, OP_SIGMOID);

  free_ten(s);

  return new_n;
}

void sigmoid_node_back(node *n) {
  for(unsigned int i = 0; i < n->prevs[0]->grad->size; ++i) {
    n->prevs[0]->grad->values[i] += n->grad->values[i] * n->data->values[i] * (1.0f - n->data->values[i]);
  }

  return;
}

node *softmax_node(node* n) {
  node *children[] = {n};
  tensor *s = softmax_ten(n->data);
  if(s == NULL) return NULL;

  node *new_n = new_node(s->values, s->shape, s->ndim, children, 1, OP_SOFTMAX);

  free_ten(s);

  return new_n;
}

void softmax_node_back(node *n) {
  unsigned int rows = n->data->shape[0];
  unsigned int cols = n->data->shape[1];

  for(unsigned int k = 0; k < cols; ++k) {
    float dot = 0.0f;
    for(unsigned int j = 0; j < rows; ++j) {
      dot += n->grad->values[k + j*cols] * n->data->values[k + j*cols];
    }

    for(unsigned int j = 0; j < rows; ++j) {
      float pred_j = n->data->values[k + j*cols];
      float g_j = n->grad->values[k + j*cols];
      n->prevs[0]->grad->values[k + j*cols] += pred_j * (g_j - dot);
    }
  }

  return;
}

node *cross_entropy_loss_node(node *pred, node *y) {
  node *children[] = {pred, y};
  tensor *ce = cross_entropy_loss_ten(pred->data, y->data);
  if(ce == NULL) return NULL;

  node *new_n = new_node(ce->values, ce->shape, ce->ndim, children, 2, OP_CE);

  free_ten(ce);

  return new_n;
}

void cross_entropy_loss_node_back(node *n) {
  for(unsigned int i = 0; i < n->prevs[0]->grad->size; ++i) {
    n->prevs[0]->grad->values[i] += n->grad->values[0] * (-n->prevs[1]->data->values[i] / (n->prevs[0]->data->values[i] + EPS_CE)); 
  }

  return;
}

void topo(node *n, node ***sorted, int *size, int *capacity) {
  n->is_topo = 1;

  if(n->n_prevs != 0) {
    for(unsigned int i = 0; i < n->n_prevs; ++i) {
      if(n->prevs[i]->is_topo == 0) {
        topo(n->prevs[i], sorted, size, capacity);
      }
    }
  }

  if(*size >= *capacity) {
    *capacity *= 2;
    node **tmp = realloc(*sorted, *capacity * sizeof(node*));
    if (!tmp) {
      return;
    }
    *sorted = tmp;
  }

  (*sorted)[(*size)++] = n;
}

void reset_topo(node *n) {
  if (n->is_topo == 0) return;
  n->is_topo = 0;
  for (unsigned int i = 0; i < n->n_prevs; ++i)
    reset_topo(n->prevs[i]);
}

void backward(node *n) {
  int size = 0;
  int cap = 10;
  node **topo_list = malloc(cap * sizeof(node*));
  if(topo_list == NULL) {
    return;
  }
  topo(n, &topo_list, &size, &cap);

  for(unsigned int i = 0; i < n->grad->size; ++i) {
    n->grad->values[i] = 1.0f;
  }

  for(int j = size-1; j >= 0; --j) {
    switch (topo_list[j]->op) {
      case OP_LEAF:
        break;
      case OP_ADD:
        add_node_back(topo_list[j]);
        break;
      case OP_SUB:
        sub_node_back(topo_list[j]);
        break; 
      case OP_MUL:
        mul_node_back(topo_list[j]);
        break; 
      case OP_SCALE:
        scale_node_back(topo_list[j]);
        break; 
      case OP_MATMUL:
        matmul_node_back(topo_list[j]);
        break; 
      case OP_TRANSPOSE:
        transpose_node_back(topo_list[j]);
        break; 
      case OP_MEAN:
        mean_node_back(topo_list[j]);
        break; 
      case OP_SUM:
        sum_node_back(topo_list[j]);
        break; 
      case OP_RELU:
        relu_node_back(topo_list[j]);
        break; 
      case OP_SIGMOID:
        sigmoid_node_back(topo_list[j]);
        break;
      case OP_SOFTMAX:
        softmax_node_back(topo_list[j]);
        break;
      case OP_CE:
        cross_entropy_loss_node_back(topo_list[j]);
        break;
    }
  }

  reset_topo(n);

  free(topo_list);

  return;
}
