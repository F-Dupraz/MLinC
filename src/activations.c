#include "activations.h"

float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

float sigmoid_deriv(float x) {
  return x * (1.0f - x);
}
