// Loader para el formato IDX de MNIST (http://yann.lecun.com/exdb/mnist/).
// Los archivos vienen .gz: descomprimir antes con `gzip -d`.

#ifndef MNIST_H
#define MNIST_H

#include "../src/tensor.h"

typedef struct mnist_set_s {
  tensor **xs;   // n tensores {rows*cols, 1}, valores normalizados a [0,1]
  tensor **ys;   // n tensores {10, 1}, one-hot
  int n;
  int rows;
  int cols;
} mnist_set;

// max_n <= 0 carga el dataset completo; si no, corta en max_n ejemplos.
// Devuelve NULL ante cualquier error (ya imprime el motivo en stderr).
mnist_set *load_mnist(const char *img_path, const char *lbl_path, int max_n);
void free_mnist(mnist_set *s);

// Fisher-Yates sobre xs e ys en lockstep. Usa rand(), sembrá con srand().
void shuffle_mnist(mnist_set *s);

// Sanity check visual: dibuja el digito i en ASCII y muestra su label.
void print_mnist_sample(const mnist_set *s, int i);

#endif // MNIST_H
