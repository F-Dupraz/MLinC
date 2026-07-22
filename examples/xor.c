#include "../src/nn.h"

#define NUM_EPOCHS 3000

int main() {
  int outs[2] = {4, 1};
  act_type act[2] = {ACT_SIGMOID, ACT_SIGMOID};
  nn *net = new_nn(2, outs, 2, act);
  tensor *xs[4], *ys[4];
  
  // TODO: crear los 4 leaves (x: {2,1}, y: {1,1})
  int x_shape[2] = {2, 1};
  int y_shape[2] = {1, 1};

  // ejemplo 0: (0,0) → 0
  float x0[2] = {0.0f, 0.0f};
  float y0[1] = {0.0f};
  xs[0] = new_ten(x0, x_shape, 2);
  ys[0] = new_ten(y0, y_shape, 2);

  // ejemplo 1: (0,1) → 1
  float x1[2] = {0.0f, 1.0f};
  float y1[1] = {1.0f};
  xs[1] = new_ten(x1, x_shape, 2);
  ys[1] = new_ten(y1, y_shape, 2); 

  // ejemplo 2: (1,0) → 1
  float x2[2] = {1.0f, 0.0f};
  float y2[1] = {1.0f};
  xs[2] = new_ten(x2, x_shape, 2);
  ys[2] = new_ten(y2, y_shape, 2); 

  // ejemplo 3: (1,1) → 0
  float x3[2] = {1.0f, 1.0f};
  float y3[1] = {0.0f};
  xs[3] = new_ten(x3, x_shape, 2);
  ys[3] = new_ten(y3, y_shape, 2); 

  train(net, xs, ys, 4, NUM_EPOCHS, 2.0f);      // entrena in-place sobre net

  for (int i = 0; i < 4; ++i) {
    tensor *p = predict(net, xs[i]);

    printf("Entrada %f, %f : retorna %f\n", xs[i]->values[0], xs[i]->values[1], p->values[0]);
    
    free_ten(p);                          // caller libera la copia
  }

  for(int j = 0; j < 4; ++j) {
    free_ten(xs[j]);
    free_ten(ys[j]);
  }

  free_nn(net);
  
  return 0;
}
