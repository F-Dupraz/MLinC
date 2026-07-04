#include <stdlib.h>
#include <string.h>

#include "./node.h"

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

node *add_node(node *n1, node *n2);
void add_node_back(node *n);
node *sub_node(node *n1, node *n2);
void sub_node_back(node *n);
node *mul_node(node *n1, node *n2);
void mul_node_back(node *n);
node *scale_node(float p, node *n);
void scale_node_back(node *n);
node *matmul_node(node *n1, node *n2);
void matmul_node_back(node *n);
node *transpose_node(node *n);
node *mean_node(node *n);
void mean_node_back(node *n);
node *sum_node(node *n);
void sum_node_back(node *n);
node *relu_node(node *n);
void relu_node_back(node *n);
node *sigmoid_node(node *n);
void sigmoid_node_back(node *n);

void topo(node *n, node ***sorted, int *size, int *capacity);
void backward(node *n);
