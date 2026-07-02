#include "./tensor.h"

typedef struct node_s {
  tensor *data; 
  tensor *grad;

  char *op_name;
  
  struct node_s **prevs;
  int n_prevs;

  void (*_backward)(struct node_s *t);
} node;

node *new_node(float *values, int *shape, int ndim, node **children, int n_child, char *op);
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

typedef struct neuron_s {
  node **w;
  node *b;
  int in_s;
  int nonlin;
} neuron;

typedef struct layer_s {
  neuron **neurons;
  int in_s;
  int out_s;
} layer;

typedef struct mlp_s {
  layer **layers;
  int layers_s;
} mlp;
