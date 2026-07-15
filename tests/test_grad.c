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

int main(void) {
  int shape[] = {2, 2};
  node *a = new_node((float[]){2.0f, 2.0f, 2.0f, 2.0f}, shape, 2, NULL, 0, OP_LEAF);
  node *b = new_node((float[]){1.0f, 2.0f, 2.0f, 1.0f}, shape, 2, NULL, 0, OP_LEAF);

  printf("=== check_add ===\n");
  check_add(a, b);

  printf("\n\n");
  printf("=== check_sub ===\n");
  check_sub(a, b);

  printf("\n\n");
  printf("=== check_mul ===\n");
  check_mul(a, b);

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

  free_node(a);
  free_node(b);

  return 0;
}
