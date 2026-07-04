#include "./tensor.h"

typedef enum {
  OP_LEAF,
  OP_ADD,
  OP_SUB,
  OP_MUL,
  OP_SCALE,
  OP_MATMUL,
  OP_TRANSPOSE,
  OP_MEAN,
  OP_SUM,
  OP_RELU,
  OP_SIGMOID
} op_type;

typedef struct node_s {
  tensor *data; 
  tensor *grad;

  op_type op;
  float scalar;        // solo lo usa OP_SCALE, el resto lo ignora
  
  struct node_s **prevs;
  unsigned int n_prevs;
} node;

node *new_node(float *values, int *shape, int ndim, node **children, int n_child, op_type op);
void free_node(node *n);

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
void transpose_node_back(node *n);
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
