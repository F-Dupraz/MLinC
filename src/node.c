#include <stdlib.h>
#include <string.h>

#include "./node.h"

node *new_node(float *values, int *shape, int ndim, node **children, int n_child, char *op) {
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
  n->op_name = op;
 
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

  n->_backward = NULL;

  return n;
}

void free_node(node *node);

node *add_node(node *node1, node *node2);
node *add_node_back(node *node);
node *sub_node(node *node1, node *node2);
node *sub_node_back(node *node);
node *mul_node(node *node1, node *node2);
node *mul_node_back(node *node);
node *scale_node(float p, node *node);
node *scale_node_back(node *node);
node *matmul_node(node *node1, node *node2);
node *matmul_node_back(node *node);
node *transpose_node(node *node);
node *mean_node(node *node);
node *mean_node_back(node *node);
node *sum_node(node *node);
node *sum_node_back(node *node);
node *relu_node(node *node);
node *relu_node_back(node *node);
node *sigmoid_node(node *node);
node *sigmoid_node_back(node *node);

void topo(node *n, node ***sorted, int *size, int *capacity);
void backward(node *node);
