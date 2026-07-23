#include <stdio.h>
#include <math.h>
#include "../src/node.h"

#define EPS 1e-3f
#define TOL 1e-2f

float rel_error(float numeric, float analytic) {
  float diff = fabsf(analytic - numeric);
  float denom = fabsf(analytic) + fabsf(numeric) + 1e-8f;
  return diff / denom;
}

void zero_grad(node *n) {
  for (unsigned int i = 0; i < n->grad->size; ++i) {
    n->grad->values[i] = 0.0f;
  }
}

void check_add(node *a, node *b) {
  zero_grad(a);
  zero_grad(b);

  node *y = add_node(a, b);
  backward(y);
  
  for(unsigned int i = 0; i < a->data->size; ++i) {
    float analytic_a = a->grad->values[i];
  
    float value_a = a->data->values[i];
    
    a->data->values[i] = value_a + EPS;
    node *y_plus_a = add_node(a, b);
    float f_plus_a = y_plus_a->data->values[i];
    free_node(y_plus_a);
  
    a->data->values[i] = value_a - EPS;
    node *y_minus_a = add_node(a, b);
    float f_minus_a = y_minus_a->data->values[i];
    free_node(y_minus_a);
  
    a->data->values[i] = value_a;
  
    float numeric_a = (f_plus_a - f_minus_a) / (2.0f * EPS);
  
    printf("Analytic value a: %f\n", analytic_a);
    printf("Numeric value a: %f\n", numeric_a);
    if (rel_error(numeric_a, analytic_a) > TOL) {
      printf("FAIL en 'a': analitico=%f numerico=%f err=%f\n",
         analytic_a, numeric_a, rel_error(numeric_a, analytic_a));
    } else {
      printf("PASS en 'a'\n");
    }
  }

  for(unsigned int i = 0; i < b->data->size; ++i) {
    float analytic_b = b->grad->values[i];

    float value_b = b->data->values[i];
 
    b->data->values[i] = value_b + EPS;
    node *y_plus_b = add_node(a, b);
    float f_plus_b = y_plus_b->data->values[i];
    free_node(y_plus_b);

    b->data->values[i] = value_b - EPS;
    node *y_minus_b = add_node(a, b);
    float f_minus_b = y_minus_b->data->values[i];
    free_node(y_minus_b);

    b->data->values[i] = value_b;

    float numeric_b = (f_plus_b - f_minus_b) / (2.0f * EPS);

    printf("Analytic value b: %f\n", analytic_b);
    printf("Numeric value b: %f\n", numeric_b);
    if (rel_error(numeric_b, analytic_b) > TOL) {
      printf("FAIL en 'b': analitico=%f numerico=%f err=%f\n",
         analytic_b, numeric_b, rel_error(numeric_b, analytic_b));
    } else {
      printf("PASS en 'b'\n");
    }
  }

  free_node(y);
}

void check_sub(node *a, node *b) {
  zero_grad(a);
  zero_grad(b);

  node *y = sub_node(a, b);
  backward(y);
  
  for(unsigned int i = 0; i < a->data->size; ++i) {
    float analytic_a = a->grad->values[i];
  
    float value_a = a->data->values[i];
    
    a->data->values[i] = value_a + EPS;
    node *y_plus_a = sub_node(a, b);
    float f_plus_a = y_plus_a->data->values[i];
    free_node(y_plus_a);
  
    a->data->values[i] = value_a - EPS;
    node *y_minus_a = sub_node(a, b);
    float f_minus_a = y_minus_a->data->values[i];
    free_node(y_minus_a);
  
    a->data->values[i] = value_a;
  
    float numeric_a = (f_plus_a - f_minus_a) / (2.0f * EPS);
  
    printf("Analytic value a: %f\n", analytic_a);
    printf("Numeric value a: %f\n", numeric_a);
    if (rel_error(numeric_a, analytic_a) > TOL) {
      printf("FAIL en 'a': analitico=%f numerico=%f err=%f\n",
         analytic_a, numeric_a, rel_error(numeric_a, analytic_a));
    } else {
      printf("PASS en 'a'\n");
    }
  }

  for(unsigned int i = 0; i < b->data->size; ++i) {
    float analytic_b = b->grad->values[i];

    float value_b = b->data->values[i];
 
    b->data->values[i] = value_b + EPS;
    node *y_plus_b = sub_node(a, b);
    float f_plus_b = y_plus_b->data->values[i];
    free_node(y_plus_b);

    b->data->values[i] = value_b - EPS;
    node *y_minus_b = sub_node(a, b);
    float f_minus_b = y_minus_b->data->values[i];
    free_node(y_minus_b);

    b->data->values[i] = value_b;

    float numeric_b = (f_plus_b - f_minus_b) / (2.0f * EPS);

    printf("Analytic value b: %f\n", analytic_b);
    printf("Numeric value b: %f\n", numeric_b);
    if (rel_error(numeric_b, analytic_b) > TOL) {
      printf("FAIL en 'b': analitico=%f numerico=%f err=%f\n",
         analytic_b, numeric_b, rel_error(numeric_b, analytic_b));
    } else {
      printf("PASS en 'b'\n");
    }
  }

  free_node(y);
}

void check_mul(node *a, node *b) {
  zero_grad(a);
  zero_grad(b);

  node *y = mul_node(a, b);
  backward(y);
  
  for(unsigned int i = 0; i < a->data->size; ++i) {
    float analytic_a = a->grad->values[i];
  
    float value_a = a->data->values[i];
    
    a->data->values[i] = value_a + EPS;
    node *y_plus_a = mul_node(a, b);
    float f_plus_a = y_plus_a->data->values[i];
    free_node(y_plus_a);
  
    a->data->values[i] = value_a - EPS;
    node *y_minus_a = mul_node(a, b);
    float f_minus_a = y_minus_a->data->values[i];
    free_node(y_minus_a);
  
    a->data->values[i] = value_a;
  
    float numeric_a = (f_plus_a - f_minus_a) / (2.0f * EPS);
  
    printf("Analytic value a: %f\n", analytic_a);
    printf("Numeric value a: %f\n", numeric_a);
    if (rel_error(numeric_a, analytic_a) > TOL) {
      printf("FAIL en 'a': analitico=%f numerico=%f err=%f\n",
         analytic_a, numeric_a, rel_error(numeric_a, analytic_a));
    } else {
      printf("PASS en 'a'\n");
    }
  }

  for(unsigned int i = 0; i < b->data->size; ++i) {
    float analytic_b = b->grad->values[i];

    float value_b = b->data->values[i];
 
    b->data->values[i] = value_b + EPS;
    node *y_plus_b = mul_node(a, b);
    float f_plus_b = y_plus_b->data->values[i];
    free_node(y_plus_b);

    b->data->values[i] = value_b - EPS;
    node *y_minus_b = mul_node(a, b);
    float f_minus_b = y_minus_b->data->values[i];
    free_node(y_minus_b);

    b->data->values[i] = value_b;

    float numeric_b = (f_plus_b - f_minus_b) / (2.0f * EPS);

    printf("Analytic value b: %f\n", analytic_b);
    printf("Numeric value b: %f\n", numeric_b);
    if (rel_error(numeric_b, analytic_b) > TOL) {
      printf("FAIL en 'b': analitico=%f numerico=%f err=%f\n",
         analytic_b, numeric_b, rel_error(numeric_b, analytic_b));
    } else {
      printf("PASS en 'b'\n");
    }
  }

  free_node(y);
}

// We utilize sum for simplicity
void check_matmul(node *a, node *b) {
  zero_grad(a);
  zero_grad(b);

  node *y = matmul_node(a, b);
  node *loss = sum_node(y);
  backward(loss);

  for(unsigned int i = 0; i < a->data->size; ++i) {
    float analytic_a = a->grad->values[i];
    float value_a = a->data->values[i];

    a->data->values[i] = value_a + EPS;
    node *y_plus_a = matmul_node(a, b);
    node *l_plus_a = sum_node(y_plus_a);
    float f_plus_a = l_plus_a->data->values[0];
    free_node(y_plus_a);
    free_node(l_plus_a);

    a->data->values[i] = value_a - EPS;
    node *y_minus_a = matmul_node(a, b);
    node *l_minus_a = sum_node(y_minus_a);
    float f_minus_a = l_minus_a->data->values[0];
    free_node(y_minus_a);
    free_node(l_minus_a);

    a->data->values[i] = value_a;

    float numeric_a = (f_plus_a - f_minus_a) / (2.0f * EPS);

    printf("Analytic value a: %f\n", analytic_a);
    printf("Numeric value a: %f\n", numeric_a);
    if (rel_error(numeric_a, analytic_a) > TOL) {
      printf("FAIL en 'a': analitico=%f numerico=%f err=%f\n",
         analytic_a, numeric_a, rel_error(numeric_a, analytic_a));
    } else {
      printf("PASS en 'a'\n");
    }
  }

  for(unsigned int i = 0; i < b->data->size; ++i) {
    float analytic_b = b->grad->values[i];
    float value_b = b->data->values[i];

    b->data->values[i] = value_b + EPS;
    node *y_plus_b = matmul_node(a, b);
    node *l_plus_b = sum_node(y_plus_b);
    float f_plus_b = l_plus_b->data->values[0];
    free_node(y_plus_b);
    free_node(l_plus_b);

    b->data->values[i] = value_b - EPS;
    node *y_minus_b = matmul_node(a, b);
    node *l_minus_b = sum_node(y_minus_b);
    float f_minus_b = l_minus_b->data->values[0];
    free_node(y_minus_b);
    free_node(l_minus_b);

    b->data->values[i] = value_b;

    float numeric_b = (f_plus_b - f_minus_b) / (2.0f * EPS);

    printf("Analytic value b: %f\n", analytic_b);
    printf("Numeric value b: %f\n", numeric_b);
    if (rel_error(numeric_b, analytic_b) > TOL) {
      printf("FAIL en 'b': analitico=%f numerico=%f err=%f\n",
         analytic_b, numeric_b, rel_error(numeric_b, analytic_b));
    } else {
      printf("PASS en 'b'\n");
    }
  }

  free_node(loss);
  free_node(y);
}

// We utilize sum for simplicity
void check_transpose(node *a) {
  zero_grad(a);

  node *y = transpose_node(a);
  node *loss = sum_node(y);
  backward(loss);

  for(unsigned int i = 0; i < a->data->size; ++i) {
    float analytic_a = a->grad->values[i];
    float value_a = a->data->values[i];

    a->data->values[i] = value_a + EPS;
    node *y_plus_a = transpose_node(a);
    node *l_plus_a = sum_node(y_plus_a);
    float f_plus_a = l_plus_a->data->values[0];
    free_node(y_plus_a);
    free_node(l_plus_a);

    a->data->values[i] = value_a - EPS;
    node *y_minus_a = transpose_node(a);
    node *l_minus_a = sum_node(y_minus_a);
    float f_minus_a = l_minus_a->data->values[0];
    free_node(y_minus_a);
    free_node(l_minus_a);

    a->data->values[i] = value_a;

    float numeric_a = (f_plus_a - f_minus_a) / (2.0f * EPS);

    printf("Analytic value a: %f\n", analytic_a);
    printf("Numeric value a: %f\n", numeric_a);
    if (rel_error(numeric_a, analytic_a) > TOL) {
      printf("FAIL en 'a': analitico=%f numerico=%f err=%f\n",
         analytic_a, numeric_a, rel_error(numeric_a, analytic_a));
    } else {
      printf("PASS en 'a'\n");
    }
  }

  free_node(loss);
  free_node(y);
}

void check_mean(node *a) {
  zero_grad(a);

  node *y = mean_node(a);

  backward(y);
 
  for(unsigned int i = 0; i < a->data->size; ++i) {
    float analytic_a = a->grad->values[i];
  
    float value_a = a->data->values[i];
    
    a->data->values[i] = value_a + EPS;
    node *y_plus_a = mean_node(a);
    float f_plus_a = y_plus_a->data->values[0];
    free_node(y_plus_a);
  
    a->data->values[i] = value_a - EPS;
    node *y_minus_a = mean_node(a);
    float f_minus_a = y_minus_a->data->values[0];
    free_node(y_minus_a);
  
    a->data->values[i] = value_a;
  
    float numeric_a = (f_plus_a - f_minus_a) / (2.0f * EPS);
  
    printf("Analytic value a: %f\n", analytic_a);
    printf("Numeric value a: %f\n", numeric_a);
    if (rel_error(numeric_a, analytic_a) > TOL) {
      printf("FAIL en 'a': analitico=%f numerico=%f err=%f\n",
         analytic_a, numeric_a, rel_error(numeric_a, analytic_a));
    } else {
      printf("PASS en 'a'\n");
    }
  }

  free_node(y);
}

void check_sum(node *a) {
  zero_grad(a);

  node *y = sum_node(a);

  backward(y);
 
  for(unsigned int i = 0; i < a->data->size; ++i) {
    float analytic_a = a->grad->values[i];
  
    float value_a = a->data->values[i];
    
    a->data->values[i] = value_a + EPS;
    node *y_plus_a = sum_node(a);
    float f_plus_a = y_plus_a->data->values[0];
    free_node(y_plus_a);
  
    a->data->values[i] = value_a - EPS;
    node *y_minus_a = sum_node(a);
    float f_minus_a = y_minus_a->data->values[0];
    free_node(y_minus_a);
  
    a->data->values[i] = value_a;
  
    float numeric_a = (f_plus_a - f_minus_a) / (2.0f * EPS);
  
    printf("Analytic value a: %f\n", analytic_a);
    printf("Numeric value a: %f\n", numeric_a);
    if (rel_error(numeric_a, analytic_a) > TOL) {
      printf("FAIL en 'a': analitico=%f numerico=%f err=%f\n",
         analytic_a, numeric_a, rel_error(numeric_a, analytic_a));
    } else {
      printf("PASS en 'a'\n");
    }
  }

  free_node(y);
}

void check_scale(float p, node *a) {
  zero_grad(a);

  node *y = scale_node(p, a);

  backward(y);
 
  for(unsigned int i = 0; i < a->data->size; ++i) {
    float analytic_a = a->grad->values[i];
  
    float value_a = a->data->values[i];
    
    a->data->values[i] = value_a + EPS;
    node *y_plus_a = scale_node(p, a);
    float f_plus_a = y_plus_a->data->values[i];
    free_node(y_plus_a);
  
    a->data->values[i] = value_a - EPS;
    node *y_minus_a = scale_node(p, a);
    float f_minus_a = y_minus_a->data->values[i];
    free_node(y_minus_a);
  
    a->data->values[i] = value_a;
  
    float numeric_a = (f_plus_a - f_minus_a) / (2.0f * EPS);
  
    printf("Analytic value a: %f\n", analytic_a);
    printf("Numeric value a: %f\n", numeric_a);
    if (rel_error(numeric_a, analytic_a) > TOL) {
      printf("FAIL en 'a': analitico=%f numerico=%f err=%f\n",
         analytic_a, numeric_a, rel_error(numeric_a, analytic_a));
    } else {
      printf("PASS en 'a'\n");
    }
  }

  free_node(y);
}

void check_relu(node *a) {
  zero_grad(a);

  node *y = relu_node(a);

  backward(y);
 
  for(unsigned int i = 0; i < a->data->size; ++i) {
    float analytic_a = a->grad->values[i];
  
    float value_a = a->data->values[i];
    
    a->data->values[i] = value_a + EPS;
    node *y_plus_a = relu_node(a);
    float f_plus_a = y_plus_a->data->values[i];
    free_node(y_plus_a);
  
    a->data->values[i] = value_a - EPS;
    node *y_minus_a = relu_node(a);
    float f_minus_a = y_minus_a->data->values[i];
    free_node(y_minus_a);
  
    a->data->values[i] = value_a;
  
    float numeric_a = (f_plus_a - f_minus_a) / (2.0f * EPS);
  
    printf("Analytic value a: %f\n", analytic_a);
    printf("Numeric value a: %f\n", numeric_a);
    if (rel_error(numeric_a, analytic_a) > TOL) {
      printf("FAIL en 'a': analitico=%f numerico=%f err=%f\n",
         analytic_a, numeric_a, rel_error(numeric_a, analytic_a));
    } else {
      printf("PASS en 'a'\n");
    }
  }

  free_node(y);
}

void check_sigmoid(node *a) {
  zero_grad(a);

  node *y = sigmoid_node(a);

  backward(y);
 
  for(unsigned int i = 0; i < a->data->size; ++i) {
    float analytic_a = a->grad->values[i];
  
    float value_a = a->data->values[i];
    
    a->data->values[i] = value_a + EPS;
    node *y_plus_a = sigmoid_node(a);
    float f_plus_a = y_plus_a->data->values[i];
    free_node(y_plus_a);
  
    a->data->values[i] = value_a - EPS;
    node *y_minus_a = sigmoid_node(a);
    float f_minus_a = y_minus_a->data->values[i];
    free_node(y_minus_a);
  
    a->data->values[i] = value_a;
  
    float numeric_a = (f_plus_a - f_minus_a) / (2.0f * EPS);
  
    printf("Analytic value a: %f\n", analytic_a);
    printf("Numeric value a: %f\n", numeric_a);
    if (rel_error(numeric_a, analytic_a) > TOL) {
      printf("FAIL en 'a': analitico=%f numerico=%f err=%f\n",
         analytic_a, numeric_a, rel_error(numeric_a, analytic_a));
    } else {
      printf("PASS en 'a'\n");
    }
  }

  free_node(y);
}

// We utilize sum for simplicity
void check_softmax(node *a) {
  zero_grad(a);
  float w_vals[] = {0.3f, 1.7f, -0.5f, 2.1f};
  node *w = new_node(w_vals, a->data->shape, a->data->ndim, NULL, 0, OP_LEAF);
  node *y = softmax_node(a);
  node *weighted = mul_node(y, w);
  node *loss = sum_node(weighted);
  backward(loss);

  for(unsigned int i = 0; i < a->data->size; ++i) {
    float analytic_a = a->grad->values[i];
    float value_a = a->data->values[i];

    a->data->values[i] = value_a + EPS;
    node *y_plus_a = softmax_node(a);
    node *w_plus_a = mul_node(y_plus_a, w);
    node *l_plus_a = sum_node(w_plus_a);
    float f_plus_a = l_plus_a->data->values[0];
    free_node(y_plus_a);
    free_node(w_plus_a);
    free_node(l_plus_a);

    a->data->values[i] = value_a - EPS;
    node *y_minus_a = softmax_node(a);
    node *w_minus_a = mul_node(y_minus_a, w);
    node *l_minus_a = sum_node(w_minus_a);
    float f_minus_a = l_minus_a->data->values[0];
    free_node(y_minus_a);
    free_node(w_minus_a);
    free_node(l_minus_a);

    a->data->values[i] = value_a;

    float numeric_a = (f_plus_a - f_minus_a) / (2.0f * EPS);

    printf("Analytic value a: %f\n", analytic_a);
    printf("Numeric value a: %f\n", numeric_a);
    if (rel_error(numeric_a, analytic_a) > TOL) {
      printf("FAIL en 'a': analitico=%f numerico=%f err=%f\n",
         analytic_a, numeric_a, rel_error(numeric_a, analytic_a));
    } else {
      printf("PASS en 'a'\n");
    }
  }

  free_node(loss);
  free_node(w);
  free_node(weighted);
  free_node(y);
}

void check_ce(node *pred, node *y) {
  zero_grad(pred);
  zero_grad(y);

  node *loss = cross_entropy_loss_node(pred, y);
  backward(loss);

  for(unsigned int i = 0; i < pred->data->size; ++i) {
    float analytic_pred = pred->grad->values[i];
    float value_pred = pred->data->values[i];

    pred->data->values[i] = value_pred + EPS;
    node *l_plus = cross_entropy_loss_node(pred, y);
    float f_plus = l_plus->data->values[0];
    free_node(l_plus);

    pred->data->values[i] = value_pred - EPS;
    node *l_minus = cross_entropy_loss_node(pred, y);
    float f_minus = l_minus->data->values[0];
    free_node(l_minus);

    pred->data->values[i] = value_pred;

    float numeric_pred = (f_plus - f_minus) / (2.0f * EPS);

    printf("Analytic value pred: %f\n", analytic_pred);
    printf("Numeric value pred: %f\n", numeric_pred);
    if (rel_error(numeric_pred, analytic_pred) > TOL) {
      printf("FAIL en 'pred': analitico=%f numerico=%f err=%f\n",
         analytic_pred, numeric_pred, rel_error(numeric_pred, analytic_pred));
    } else {
      printf("PASS en 'pred'\n");
    }
  }

  free_node(loss);
}
int main(void) {
  int shape[] = {2, 2};
  int shape_loss[] = {4, 1};
  node *a = new_node((float[]){2.0f, 2.0f, 2.0f, 2.0f}, shape, 2, NULL, 0, OP_LEAF);
  node *b = new_node((float[]){1.0f, 2.0f, 2.0f, 1.0f}, shape, 2, NULL, 0, OP_LEAF);
  node *pred = new_node((float[]){0.7f, 0.1f, 0.1f, 0.1f}, shape_loss, 2, NULL, 0, OP_LEAF);
  node *y = new_node((float[]){1.0f, 0.0f, 0.0f, 0.0f}, shape_loss, 2, NULL, 0, OP_LEAF);

  printf("=== check_add ===\n");
  check_add(a, b);

  printf("\n\n");
  printf("=== check_sub ===\n");
  check_sub(a, b);

  printf("\n\n");
  printf("=== check_mul ===\n");
  check_mul(a, b);

  printf("\n\n");
  printf("=== check_matmul ===\n");
  check_matmul(a, b);

  printf("\n\n");
  printf("=== check_transpose ===\n");
  check_transpose(a);

  printf("\n\n");
  printf("=== check_mean ===\n");
  check_mean(a);

  printf("\n\n");
  printf("=== check_sum ===\n");
  check_sum(a);

  printf("\n\n");
  printf("=== check_scale ===\n");
  check_scale(7.5f, a);

  printf("\n\n");
  printf("=== check_relu ===\n");
  check_relu(a);

  printf("\n\n");
  printf("=== check_sigmoid ===\n");
  check_sigmoid(a);

  printf("\n\n");
  printf("=== check_softmax ===\n");
  check_softmax(a);

  printf("\n\n");
  printf("=== check_ce ===\n");
  check_ce(pred, y);

  free_node(a);
  free_node(b);
  free_node(pred);
  free_node(y);

  return 0;
}
