#include "nn.h"
#include "activations.h"
#include "mat.h"

nn_t *new_nn(size_t in_s, size_t hid_s, size_t out_s) {
  if(in_s <= 0 || hid_s <= 0 || out_s <= 0) { return NULL; }
  nn_t *nn = malloc(sizeof(nn_t));
  if(nn == NULL) { return NULL; }
  
  nn->in_s = in_s;
  nn->hid_s = hid_s;
  nn->out_s = out_s;

  nn->z1 = new_mat(hid_s, 1);
  nn->a1 = new_mat(hid_s, 1);
  nn->z2 = new_mat(out_s, 1);
  nn->a2 = new_mat(out_s, 1);

  nn->w1 = new_mat(hid_s, in_s);
  nn->b1 = new_mat(hid_s, 1);
  nn->w2 = new_mat(out_s, hid_s);
  nn->b2 = new_mat(out_s, 1);
  if(nn->w1 == NULL || nn->b1 == NULL || nn->w2 == NULL || nn->b2 == NULL) {
    free_nn(nn);
    return NULL;
  }

  return nn;
}

//
int forward_nn(nn_t *nn, mat *x) {
  for(size_t i = 0; i < nn->hid_s; ++i) {
    float sum = 0.0f;
    for(size_t j = 0; j < nn->in_s; ++j) {
      sum += getat_mat(x, j, 0) * getat_mat(nn->w1, i, j);
    }
    setat_mat(nn->z1, i, 0, sum + getat_mat(nn->b1, i, 0));
    setat_mat(nn->a1, i, 0, sigmoid(getat_mat(nn->z1, i, 0)));
  }

  for(size_t i = 0; i < nn->out_s; ++i) {
    float sum = 0.0f;
    for(size_t j = 0; j < nn->hid_s; ++j) {
      sum += getat_mat(nn->a1, j, 0) * getat_mat(nn->w2, i, j);
    }
    setat_mat(nn->z2, i, 0, sum + getat_mat(nn->b2, i, 0));
    setat_mat(nn->a2, i, 0, sigmoid(getat_mat(nn->z2, i, 0)));
  }

  return 1;
}

// Here I chose to implement the Tinn version because of its simplicity
nn_t *read_nn(FILE *nn_file) {
  if(nn_file == NULL) { return NULL; }
  
  size_t nins = 0;
  size_t nhid = 0;
  size_t nout = 0;
  fscanf(nn_file, "%zu %zu %zu\n", &nins, &nhid, &nout);
  
  nn_t *nn = new_nn(nins, nhid, nout);
  
  float temp = 0;

  for(size_t i = 0; i < nn->w1->rows; ++i) {
    for(size_t j = 0; j < nn->w1->cols; ++j) {
      fscanf(nn_file, "%f ", &temp);
      setat_mat(nn->w1, i, j, temp);
    }
    fscanf(nn_file, "\n");
  }

  for(size_t i = 0; i < nn->b1->rows; ++i) {
    fscanf(nn_file, "%f ", &temp);
    setat_mat(nn->b1, i, 0, temp);
  }
  fscanf(nn_file, "\n");

  for(size_t i = 0; i < nn->w2->rows; ++i) {
    for(size_t j = 0; j < nn->w2->cols; ++j) {
      fscanf(nn_file, "%f ", &temp);
      setat_mat(nn->w2, i, j, temp);
    }
    fscanf(nn_file, "\n");
  }

  for(size_t i = 0; i < nn->b2->rows; ++i) {
    fscanf(nn_file, "%f ", &temp);
    setat_mat(nn->b2, i, 0, temp);
  }

  fclose(nn_file);

  return nn;
}

void free_nn(nn_t *nn) {
  free_mat(nn->w1);
  free_mat(nn->b1);
  free_mat(nn->w2);
  free_mat(nn->b2);
  free_mat(nn->a1);
  free_mat(nn->z1);
  free_mat(nn->a2);
  free_mat(nn->z2);
  free(nn);

  return;
}

void randomize_nn(nn_t *nn) {
  for(size_t i = 0; i < nn->w1->rows*nn->w1->cols; ++i) {
    float rw = ((float)rand() / RAND_MAX) - 0.5;
    nn->w1->data[i] = rw;
  }

  for(size_t i = 0; i < nn->w2->rows*nn->w2->cols; ++i) {
    float rw = ((float)rand() / RAND_MAX) - 0.5;
    nn->w2->data[i] = rw;
  }

  return;
}

int write_nn(nn_t const *nn, FILE *nn_file) {
  if(nn_file == NULL) { return 0; }
  
  fprintf(nn_file, "%zu %zu %zu\n", nn->in_s, nn->hid_s, nn->out_s);

  for(size_t i = 0; i < nn->w1->rows; ++i) {
    for(size_t j = 0; j < nn->w1->cols; ++j) {
      fprintf(nn_file, "%f ", getat_mat(nn->w1, i, j));
    }
    fprintf(nn_file, "\n");
  }

  for(size_t i = 0; i < nn->b1->rows; ++i) {
    fprintf(nn_file, "%f ", getat_mat(nn->b1, i, 0));
  }
  fprintf(nn_file, "\n");

  for(size_t i = 0; i < nn->w2->rows; ++i) {
    for(size_t j = 0; j < nn->w2->cols; ++j) {
      fprintf(nn_file, "%f ", getat_mat(nn->w2, i, j));
    }
    fprintf(nn_file, "\n");
  }

  for(size_t i = 0; i < nn->b2->rows; ++i) {
    fprintf(nn_file, "%f ", getat_mat(nn->b2, i, 0));
  }

  fclose(nn_file);

  return 1;
}
