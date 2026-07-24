#include <time.h>

#include "../src/nn.h"
#include "../mnist/mnist.h"

#define NUM_EPOCHS 10
#define LR 0.01

int main() {
  int outs[2] = {64, 10};
  act_type act[2] = {ACT_RELU, ACT_SOFTMAX};
  nn *net = new_nn(784, outs, 2, act);

  srand(time(NULL));
  mnist_set *tr = load_mnist("./mnist/train-images-idx3-ubyte",
                           "./mnist/train-labels-idx1-ubyte", 10000);
  if (tr == NULL) return 1;

  train(net, tr->xs, tr->ys, tr->n, NUM_EPOCHS, LR, CROSS_ENTROPY, tr);

  free_mnist(tr);

  printf("\nTraining finished!\n");

  mnist_set *tt = load_mnist("./mnist/t10k-images-idx3-ubyte",
                           "./mnist/t10k-labels-idx1-ubyte", 10);
  if (tt == NULL) return 1;

  shuffle_mnist(tt);

  for(int k = 0; k < 10; ++k) {
    print_mnist_sample(tt, k);
    printf("\n");
    
    tensor *p = predict(net, tt->xs[k]);

    printf("\n");
    for(int i = 0; i < 10; ++i) {
      printf("p[%d] %f ", i, p->values[i]);
    }
    printf("\n");
    for(int j = 0; j < 10; ++j) {
      printf("y[%d] %f ", j, tt->ys[k]->values[j]);
    }

    printf("\n");
  }

  free_mnist(tt);

  free_nn(net);
  
  return 0;
}
