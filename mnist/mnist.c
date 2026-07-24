#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "./mnist.h"

#define MNIST_MAGIC_IMAGES 2051u
#define MNIST_MAGIC_LABELS 2049u
#define MNIST_CLASSES      10

// Los enteros del header son big-endian (MSB first). Leerlos con un
// fread directo a un uint32_t los rompe en cualquier maquina little-endian,
// asi que los reconstruimos byte a byte.
// Separamos "salio bien" del valor leido para no confundir un 0 legitimo
// con un error de lectura.
static int read_u32_be(FILE *f, uint32_t *out) {
  unsigned char b[4];
  if (fread(b, 1, 4, f) != 4) { return 0; }

  *out = ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16)
       | ((uint32_t)b[2] <<  8) |  (uint32_t)b[3];

  return 1;
}

// Devuelve el bloque crudo de pixeles (n*rows*cols bytes). El caller lo libera.
static unsigned char *read_images_raw(const char *path, uint32_t *n_out,
                                      uint32_t *rows_out, uint32_t *cols_out) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) {
    fprintf(stderr, "mnist: no pude abrir '%s'\n", path);
    return NULL;
  }

  uint32_t magic = 0, n = 0, rows = 0, cols = 0;
  if (!read_u32_be(f, &magic) || !read_u32_be(f, &n)
   || !read_u32_be(f, &rows)  || !read_u32_be(f, &cols)) {
    fprintf(stderr, "mnist: header truncado en '%s'\n", path);
    fclose(f);
    return NULL;
  }

  if (magic != MNIST_MAGIC_IMAGES) {
    fprintf(stderr, "mnist: magic %u en '%s', esperaba %u "
                    "(archivo equivocado o todavia comprimido?)\n",
            magic, path, MNIST_MAGIC_IMAGES);
    fclose(f);
    return NULL;
  }

  if (n == 0 || rows == 0 || cols == 0) {
    fprintf(stderr, "mnist: dimensiones invalidas en '%s'\n", path);
    fclose(f);
    return NULL;
  }

  size_t total = (size_t)n * (size_t)rows * (size_t)cols;
  unsigned char *raw = malloc(total);
  if (raw == NULL) {
    fprintf(stderr, "mnist: no entran %zu bytes de pixeles\n", total);
    fclose(f);
    return NULL;
  }

  // Un solo fread grande: 60k freads de 784 bytes serian 60k syscalls.
  if (fread(raw, 1, total, f) != total) {
    fprintf(stderr, "mnist: datos truncados en '%s'\n", path);
    free(raw);
    fclose(f);
    return NULL;
  }

  fclose(f);

  *n_out = n;
  *rows_out = rows;
  *cols_out = cols;

  return raw;
}

// Devuelve el bloque crudo de labels (n bytes). El caller lo libera.
static unsigned char *read_labels_raw(const char *path, uint32_t *n_out) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) {
    fprintf(stderr, "mnist: no pude abrir '%s'\n", path);
    return NULL;
  }

  uint32_t magic = 0, n = 0;
  if (!read_u32_be(f, &magic) || !read_u32_be(f, &n)) {
    fprintf(stderr, "mnist: header truncado en '%s'\n", path);
    fclose(f);
    return NULL;
  }

  if (magic != MNIST_MAGIC_LABELS) {
    fprintf(stderr, "mnist: magic %u en '%s', esperaba %u\n",
            magic, path, MNIST_MAGIC_LABELS);
    fclose(f);
    return NULL;
  }

  if (n == 0) {
    fprintf(stderr, "mnist: 0 labels en '%s'\n", path);
    fclose(f);
    return NULL;
  }

  unsigned char *raw = malloc(n);
  if (raw == NULL) {
    fclose(f);
    return NULL;
  }

  if (fread(raw, 1, n, f) != n) {
    fprintf(stderr, "mnist: datos truncados en '%s'\n", path);
    free(raw);
    fclose(f);
    return NULL;
  }

  fclose(f);

  *n_out = n;

  return raw;
}

mnist_set *load_mnist(const char *img_path, const char *lbl_path, int max_n) {
  uint32_t n_img = 0, rows = 0, cols = 0, n_lbl = 0;

  unsigned char *raw_px = read_images_raw(img_path, &n_img, &rows, &cols);
  if (raw_px == NULL) { return NULL; }

  unsigned char *raw_lbl = read_labels_raw(lbl_path, &n_lbl);
  if (raw_lbl == NULL) {
    free(raw_px);
    return NULL;
  }

  if (n_img != n_lbl) {
    fprintf(stderr, "mnist: %u imagenes vs %u labels, no coinciden\n",
            n_img, n_lbl);
    free(raw_px);
    free(raw_lbl);
    return NULL;
  }

  int n = (int)n_img;
  if (max_n > 0 && max_n < n) { n = max_n; }

  mnist_set *s = malloc(sizeof(mnist_set));
  if (s == NULL) {
    free(raw_px);
    free(raw_lbl);
    return NULL;
  }

  s->n = n;
  s->rows = (int)rows;
  s->cols = (int)cols;

  // calloc y no malloc: si fallamos a mitad de camino, free_mnist recorre
  // el array entero y free_ten(NULL) es no-op. Sin esto leeriamos punteros basura.
  s->xs = calloc((size_t)n, sizeof(tensor*));
  s->ys = calloc((size_t)n, sizeof(tensor*));
  if (s->xs == NULL || s->ys == NULL) {
    free(raw_px);
    free(raw_lbl);
    free_mnist(s);
    return NULL;
  }

  size_t px = (size_t)rows * (size_t)cols;

  float *buf = malloc(px * sizeof(float));
  if (buf == NULL) {
    free(raw_px);
    free(raw_lbl);
    free_mnist(s);
    return NULL;
  }

  // Column vectors, consistente con la convencion W*x del resto del codigo.
  int x_shape[2] = { (int)px, 1 };
  int y_shape[2] = { MNIST_CLASSES, 1 };

  for (int i = 0; i < n; ++i) {
    for (size_t j = 0; j < px; ++j) {
      buf[j] = (float)raw_px[(size_t)i * px + j] / 255.0f;
    }
    s->xs[i] = new_ten(buf, x_shape, 2);

    unsigned char lbl = raw_lbl[i];
    if (lbl >= MNIST_CLASSES) {
      fprintf(stderr, "mnist: label %u fuera de rango en el ejemplo %d\n",
              (unsigned)lbl, i);
      free(buf);
      free(raw_px);
      free(raw_lbl);
      free_mnist(s);
      return NULL;
    }

    float onehot[MNIST_CLASSES] = { 0.0f };  // se re-inicializa cada vuelta
    onehot[lbl] = 1.0f;
    s->ys[i] = new_ten(onehot, y_shape, 2);

    if (s->xs[i] == NULL || s->ys[i] == NULL) {
      fprintf(stderr, "mnist: sin memoria en el ejemplo %d\n", i);
      free(buf);
      free(raw_px);
      free(raw_lbl);
      free_mnist(s);
      return NULL;
    }
  }

  free(buf);
  free(raw_px);
  free(raw_lbl);

  return s;
}

void free_mnist(mnist_set *s) {
  if (s == NULL) { return; }

  if (s->xs != NULL) {
    for (int i = 0; i < s->n; ++i) { free_ten(s->xs[i]); }
    free(s->xs);
  }

  if (s->ys != NULL) {
    for (int i = 0; i < s->n; ++i) { free_ten(s->ys[i]); }
    free(s->ys);
  }

  free(s);

  return;
}

void shuffle_mnist(mnist_set *s) {
  if (s == NULL || s->n < 2) { return; }

  for (int i = s->n - 1; i > 0; --i) {
    int j = rand() % (i + 1);

    tensor *tx = s->xs[i]; s->xs[i] = s->xs[j]; s->xs[j] = tx;
    tensor *ty = s->ys[i]; s->ys[i] = s->ys[j]; s->ys[j] = ty;
  }

  return;
}

void print_mnist_sample(const mnist_set *s, int i) {
  if (s == NULL || i < 0 || i >= s->n) { return; }

  const char *ramp = " .:-=+*#%@";
  tensor *x = s->xs[i];

  for (int r = 0; r < s->rows; ++r) {
    for (int c = 0; c < s->cols; ++c) {
      float v = x->values[r * s->cols + c];
      int idx = (int)(v * 9.0f + 0.5f);
      if (idx < 0) { idx = 0; }
      if (idx > 9) { idx = 9; }
      // dos chars por pixel, si no el digito sale aplastado en la terminal
      putchar(ramp[idx]);
      putchar(ramp[idx]);
    }
    putchar('\n');
  }

  int label = -1;
  for (int k = 0; k < MNIST_CLASSES; ++k) {
    if (s->ys[i]->values[k] == 1.0f) { label = k; }
  }
  printf("label = %d\n", label);

  return;
}
