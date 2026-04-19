# Plan: Building ML from Scratch in C (Adjusted for Reality)

**Timeline:** 10–12 weeks, 3–4 hours/week  
**Total commitment:** ~36–48 hours  
**Language:** Pure C (ANSI C99)

## Why this plan differs from the original

The original 8-week plan underestimated:
- **Debugging time in C** — segfaults, indexing bugs, memory corruption are invisible until they explode. Budget 30%–50% of coding time for debugging.
- **Concept density** — autodiff and backprop are heavy; rushing them yields buggy code that wastes weeks later.
- **Academic load** — you're carrying Análisis II, Electrónica Digital I, and other courses. A bad week makes compressed schedules impossible.
- **Relevance** — GEMM optimization is valuable for HPC/infrastructure, but tensor architecture and memory design matter more for your AI/QC/fintech goals.

**Changes:**
- **Weeks:** 8 → 10–12
- **Phases:** Keep 4, but redistribute
- **Phase 1:** 4 days → 7 days (matrix library, NN, training loop)
- **Phase 2:** 4 days → 6–7 days (autodiff with N-dimensional tensors, not just scalars)
- **Phase 3:** Replace "GEMM optimization" with "Tensor design & ggml architecture" (4 days)
- **Phase 4:** Keep polish, maybe 3 days

The result is a repo that's **deeper, debugged, and portfolio-ready**.

---

## Repository structure

```
ml-from-scratch-c/
├── docs/
│   ├── roadmap.md          # learning path (this document)
│   └── README.md           # project overview
├── src/
│   ├── mat.h / mat.c       # matrix operations (2D, foundational)
│   ├── tensor.h / tensor.c # N-dimensional tensor library (Phase 2)
│   ├── nn.h / nn.c         # neural network (2-layer MLP)
│   ├── autograd.h / autograd.c   # autodiff engine (scalar → tensor)
│   └── optim.h / optim.c   # optimizers (SGD, Adam sketch)
├── examples/
│   ├── xor.c               # XOR with explicit backprop
│   ├── xor_auto.c          # XOR with autograd
│   └── mnist_toy.c         # toy MNIST (optional, Phase 2.5)
├── tests/
│   ├── test_mat.c          # matrix operation tests
│   ├── test_tensor.c       # tensor operation tests
│   ├── test_grad.c         # gradient checking (numerical vs autodiff)
│   └── run_tests.sh        # test runner
├── perf/
│   └── bench_matmul.c      # simple matmul benchmark
├── third_party/
│   └── README_REFERENCES.md # links to inspiration repos
└── Makefile (or CMakeLists.txt)
```

---

## Phase 1: Matrix library, neural networks, training loop (7 days)

**Objective:**
- Implement a clean, simple matrix library in C.
- Build a 2-layer feedforward MLP (fully connected).
- Train it on XOR using explicit backpropagation (chain rule by hand).
- Debug until loss decreases and XOR converges.

### Semana 1

#### Día 1 (1–2 h)

**Material:**
- Read **`glouw/tinn`** (~200 lines). Focus on:
  - How are weights stored? (Single array or matrix structs?)
  - Forward pass: matmul + sigmoid.
  - Backward pass: gradient calculation and weight update.
- Skim **`codeplea/genann`** README — get a sense of the API style.

**Code:**
- Initialize repo:
  ```bash
  mkdir ml-from-scratch-c && cd ml-from-scratch-c
  git init
  mkdir -p src examples tests docs
  touch README.md
  touch src/{mat.h,mat.c,nn.h,nn.c}
  touch examples/{xor.c,xor_auto.c}
  touch tests/{test_mat.c,run_tests.sh}
  ```
- Write `src/mat.h`: Define `matrix_t` struct and function declarations (no implementations yet):
  ```c
  typedef struct {
      int rows, cols;
      double *data;  // row-major layout
  } matrix_t;

  matrix_t* mat_new(int rows, int cols);
  void mat_free(matrix_t *m);
  void mat_print(matrix_t *m);
  void mat_copy(matrix_t *dst, matrix_t *src);
  void mat_mul(matrix_t *a, matrix_t *b, matrix_t *out);
  void mat_add(matrix_t *a, matrix_t *b, matrix_t *out);
  void mat_sub(matrix_t *a, matrix_t *b, matrix_t *out);
  void mat_scale(matrix_t *m, double alpha);
  double sigmoid(double x);
  double sigmoid_deriv(double x);
  ```
- Start `README.md`:
  ```markdown
  # ML from Scratch in C

  Building machine learning fundamentals in pure C, from matrix ops through autodiff to tensor libraries.

  **Inspiration:** tinn, genann, micrograd.c, smdaa/teeny-autograd-c, ggml, llama2.c.
  ```

---

#### Día 2 (1–2 h)

**Material:**
- Read **`nomemory/neat-matrix-library`** — focus on `mat_mul`, memory layout, and error handling.
- Watch or read a **short matrix algebra refresher** (Khan Academy or similar) — make sure you know what row-major indexing means.

**Code:**
- Implement in `src/mat.c`:
  - `mat_new(rows, cols)` — allocate `rows * cols * sizeof(double)`, initialize to 0.
  - `mat_free(m)` — deallocate.
  - `mat_print(m)` — printf loop, nice formatting.
  - `mat_copy(dst, src)` — `memcpy`.
- Test with a simple `examples/test_basic.c`:
  ```c
  int main() {
      matrix_t *m = mat_new(3, 4);
      mat_print(m);
      mat_free(m);
      return 0;
  }
  ```
- Compile and run: `gcc -O2 -Wall src/mat.c examples/test_basic.c -o test_basic && ./test_basic`

---

#### Día 3 (1.5–2 h)

**Material:**
- Re-read the **forward pass** section of tinn or genann code.
- Understand: weight initialization (random small values), matrix indexing in C, and sigmoid computation.

**Code:**
- Implement in `src/mat.c`:
  - `mat_mul(a, b, out)` — standard triple-loop: `out[i][j] += a[i][k] * b[k][j]`.
  - `sigmoid(x)` and `sigmoid_deriv(x)`.
- Implement in `src/mat.c`:
  - `mat_add(a, b, out)` — element-wise: `out[i][j] = a[i][j] + b[i][j]`.
  - `mat_scale(m, alpha)` — scale all elements by alpha (useful for learning rate).
- In `src/nn.h`, define the network:
  ```c
  typedef struct {
      matrix_t *w1, *b1;  // layer 1: weights, biases
      matrix_t *w2, *b2;  // layer 2
      int in_size, hid_size, out_size;
      // forward pass caches
      matrix_t *z1, *a1, *z2, *a2;
  } nn_t;
  ```
- In `src/nn.h`, declare:
  ```c
  nn_t* nn_new(int in_size, int hid_size, int out_size);
  void nn_free(nn_t *net);
  void nn_forward(nn_t *net, matrix_t *x);  // caches z1, a1, z2, a2
  void nn_print_output(nn_t *net);
  ```

---

#### Día 4 (2–2.5 h)

**Material:**
- Carefully read the **backpropagation** section of tinn or genann.
- Draw out the computational graph on paper:
  - Input x → w1, x → z1=w1*x+b1 → a1=sigmoid(z1) → w2, a1 → z2=w2*a1+b2 → a2=sigmoid(z2) → loss.
  - Backward: d(loss)/d(a2) → d(a2)/d(z2) → chain back.

**Code:**
- Implement in `src/nn.c`:
  ```c
  nn_t* nn_new(int in_size, int hid_size, int out_size) {
      nn_t *net = malloc(sizeof(nn_t));
      net->w1 = mat_new(hid_size, in_size);
      net->b1 = mat_new(hid_size, 1);
      net->w2 = mat_new(out_size, hid_size);
      net->b2 = mat_new(out_size, 1);
      // ... init weights randomly (small values) ...
      // ... allocate caches z1, a1, z2, a2 ...
      return net;
  }
  ```
- Implement `nn_forward(net, x)`:
  - Compute z1 = w1 * x + b1.
  - Compute a1 = sigmoid(z1) (element-wise).
  - Compute z2 = w2 * a1 + b2.
  - Compute a2 = sigmoid(z2).
  - **Cache** z1, a1, z2, a2 for backward pass.
- Create `examples/xor.c` with the XOR dataset:
  ```c
  double X[4][2] = {{0,0},{0,1},{1,0},{1,1}};
  double Y[4][1] = {{0},{1},{1},{0}};
  ```

---

### Semana 2

#### Día 5 (2–2.5 h)

**Material:**
- Study the **chain rule** carefully. Draw the computation graph again.
- For the 2-layer network:
  - Loss = MSE = 0.5 * sum((a2 - y)^2)
  - d(Loss)/d(a2) = (a2 - y)
  - d(a2)/d(z2) = sigmoid_deriv(z2)
  - d(z2)/d(w2) = a1^T (transpose!)
  - d(z2)/d(b2) = 1
  - d(z2)/d(a1) = w2^T
  - Then chain through layer 1 similarly.

**Code:**
- In `src/nn.c`, implement `nn_backward(net, x, y, learning_rate)`:
  ```c
  void nn_backward(nn_t *net, matrix_t *x, matrix_t *y, double lr) {
      // Compute dz2 = (a2 - y) * sigmoid_deriv(z2)
      matrix_t *dz2 = mat_new(net->out_size, 1);  // error at output layer
      // ... copy a2, subtract y, element-wise mult with sigmoid_deriv(z2) ...

      // Compute dw2 = dz2 * a1^T
      // Compute db2 = dz2 (sum over batch, but single example here)
      
      // Compute da1 = w2^T * dz2
      // Compute dz1 = da1 * sigmoid_deriv(z1)
      
      // Compute dw1 = dz1 * x^T
      // Compute db1 = dz1
      
      // Update weights: w -= lr * dw
      // Update biases:  b -= lr * db
      
      // Free temporaries
  }
  ```
- This is the **hardest part** — take your time. Index bugs here break everything downstream.
- Test on paper first: manually compute gradients for a single XOR example, then verify the code produces the same numbers.

---

#### Día 6 (2–3 h)

**Material:**
- Revisit backprop one more time if you got stuck.
- Simple test: compile your backward pass and run it on a single XOR example. Print the gradients. Do they look reasonable (not huge, not zero)?

**Code:**
- **Debug** the backward pass. This will consume 1–2 hours.
- Write a training loop in `examples/xor.c`:
  ```c
  int main() {
      nn_t *net = nn_new(2, 4, 1);  // 2 inputs, 4 hidden, 1 output
      
      matrix_t *X = mat_from_array(4, 2, (double[]){...});
      matrix_t *Y = mat_from_array(4, 1, (double[]){...});
      
      for (int epoch = 0; epoch < 1000; epoch++) {
          double total_loss = 0;
          for (int i = 0; i < 4; i++) {
              // Forward
              nn_forward(net, X_row[i]);
              // Compute loss for this example
              // Backward
              nn_backward(net, X_row[i], Y_row[i], 0.5);  // lr = 0.5
              total_loss += loss;
          }
          if (epoch % 100 == 0) printf("Epoch %d, Loss: %.4f\n", epoch, total_loss / 4);
      }
      
      nn_free(net);
      return 0;
  }
  ```
- **Goal:** See loss decrease. If it doesn't, debug:
  - Learning rate too high? Try 0.1.
  - Weight initialization? Try small random values in [-0.5, 0.5].
  - Backward pass bug? Print intermediate gradients.

---

#### Día 7 (1.5–2 h)

**Material:**
- None — just **refinement and validation**.

**Code:**
- Get the training loop working. Loss should visibly decrease.
- Add a **prediction function**: given a 2D input, what does the network output?
  ```c
  void nn_predict(nn_t *net, double x1, double x2) {
      // forward on [x1, x2]
      printf("Input: (%.1f, %.1f) -> Output: %.4f\n", x1, x2, net->a2->data[0]);
  }
  ```
- At the end of training, call `nn_predict(net, 0, 0)`, `nn_predict(net, 0, 1)`, etc.
- Expected: ~0 for (0,0) and (1,1), ~1 for (0,1) and (1,0).
- **Commit everything to git.** Write a brief `docs/phase1.md`:
  ```markdown
  # Phase 1: Matrix Library & Basic Neural Network

  - Implemented a matrix library with basic ops (mul, add, transpose).
  - Built a 2-layer MLP trained with explicit backpropagation.
  - Trained on XOR dataset; network converges to correct outputs.
  - Lessons: Row-major indexing, chain rule, gradient accumulation.
  ```

---

## Phase 2: Autodiff with N-dimensional tensors (6–7 days)

**Objective:**
- Replace explicit backprop with an **automatic differentiation engine**.
- Generalize from 2D matrices to **N-dimensional tensors** (foundational for Phase 3).
- Implement autograd inspired by micrograd.c and smdaa/teeny-autograd-c, but with tensor support.
- Train XOR again using autograd instead of hand-written gradients.

### Semana 3

#### Día 8 (1.5–2 h)

**Material:**
- Read **`Jaykef/micrograd.c`** carefully. Understand:
  - `Value` struct: data, grad, operation, children pointers.
  - `backward()` function pointer stored in each node.
  - Recursive backward traversal via topological sort.
- Read the **blog post** from smdaa's teeny-autograd-c. Note how they add:
  - Broadcasting (automatic shape alignment).
  - Multiple children per node (for ops like add, mul).

**Code:**
- In `src/tensor.h`, define a **scalar tensor** first (warm-up):
  ```c
  typedef struct tensor_s {
      double data;
      double grad;
      
      char *op_name;  // "add", "mul", "tanh", etc.
      struct tensor_s *children[2];  // at most 2 operands
      int n_children;
      
      void (*backward)(struct tensor_s *);  // function to compute gradients
  } tensor_t;
  ```
- Declare:
  ```c
  tensor_t* tensor_new(double data);
  void tensor_free(tensor_t *t);
  tensor_t* tensor_add(tensor_t *a, tensor_t *b);
  tensor_t* tensor_mul(tensor_t *a, tensor_t *b);
  tensor_t* tensor_tanh(tensor_t *x);
  void tensor_backward(tensor_t *root);
  void tensor_zero_grad(tensor_t *root);
  ```

---

#### Día 9 (2–2.5 h)

**Material:**
- Review the **chain rule** one more time. For a computational graph:
  - Each node stores `data` (value) and `grad` (gradient w.r.t. loss).
  - During backward, compute `grad` by chaining: if node is child of parent, parent's grad flows back through the operation.

**Code:**
- In `src/tensor.c`, implement:
  ```c
  tensor_t* tensor_new(double data) {
      tensor_t *t = malloc(sizeof(tensor_t));
      t->data = data;
      t->grad = 0;
      t->op_name = NULL;
      t->n_children = 0;
      t->backward = NULL;
      return t;
  }
  ```
- Implement `tensor_add(a, b)`:
  - Create result node: `result->data = a->data + b->data`.
  - Set result's `backward` to a function that does:
    ```c
    void add_backward(tensor_t *node) {
        // node's children are a and b
        // grad(a) += node->grad * 1
        // grad(b) += node->grad * 1
        node->children[0]->grad += node->grad;
        node->children[1]->grad += node->grad;
    }
    ```
- Similarly, implement `tensor_mul(a, b)` with:
  - `result->data = a->data * b->data`.
  - `mul_backward`: `grad(a) += node->grad * b->data; grad(b) += node->grad * a->data`.
- Implement `tensor_tanh(x)`:
  - `result->data = tanh(x->data)`.
  - `tanh_backward`: `grad(x) += node->grad * (1 - tanh^2(x->data))`.

---

#### Día 10 (1.5–2 h)

**Material:**
- Understand **topological sort and backward traversal**. Review how micrograd does this.

**Code:**
- Implement `tensor_backward(root)`:
  - Build a **topological order** of all nodes reachable from root.
  - Set `root->grad = 1.0` (loss = output).
  - Iterate through nodes in **reverse** topological order, calling each node's `backward()` function.
  ```c
  void tensor_backward(tensor_t *root) {
      // (1) Topological sort: DFS to collect all nodes
      // (2) root->grad = 1.0
      // (3) Iterate nodes in reverse, call each node->backward()
  }
  ```
- **Test** on a simple computation: 
  - `a = tensor_new(2.0), b = tensor_new(3.0), c = tensor_add(a, b), d = tensor_mul(c, b)`
  - `tensor_backward(d)`.
  - Expected: `d->grad = 1.0, c->grad = 3.0, b->grad = 3 + 6 = 9, a->grad = 3`.
  - Print and verify by hand.

---

### Semana 4

#### Día 11 (2–2.5 h)

**Material:**
- Think about **how to generalize scalars to N-dim tensors**. Read snippets from smdaa/teeny-autograd-c where they handle shape broadcasting.

**Code:**
- Extend `tensor.h` to handle **shapes**:
  ```c
  typedef struct {
      int ndim;
      int *shape;  // [2, 3, 4] for a 2x3x4 tensor
  } shape_t;

  typedef struct tensor_s {
      double *data;  // flattened 1D array
      double *grad;
      shape_t shape;
      
      char *op_name;
      struct tensor_s *children[2];
      int n_children;
      void (*backward)(struct tensor_s *);
  } tensor_t;
  ```
- Implement `shape_numel(shape)` — total number of elements.
- Implement `shape_broadcast(shape_a, shape_b)` — align shapes (simplified version OK for Phase 2).
- Implement `tensor_new_shaped(shape)`, `tensor_new_zeros(shape)`, `tensor_new_randn(shape)`.

**This day is about design.** Don't implement all operations; just get the shape system working.

---

#### Día 12 (2–3 h)

**Material:**
- None — just implementation and debugging.

**Code:**
- Implement element-wise operations for N-dim tensors:
  - `tensor_add_shaped`, `tensor_mul_shaped`, `tensor_tanh_shaped` — handle broadcasting.
- Implement **matrix multiplication** for 2D tensors:
  ```c
  tensor_t* tensor_matmul(tensor_t *a, tensor_t *b);  // a: (m, n), b: (n, p) -> (m, p)
  ```
- Write a **numerical gradient checker** test:
  ```c
  // For each element of the output, compute gradient two ways:
  // (1) Autodiff.
  // (2) Finite differences: (f(x+eps) - f(x-eps)) / (2*eps).
  // Compare. If they match (~1e-4 relative error), autodiff is correct.
  ```
- **Test extensively.** This is where most bugs hide.

---

#### Día 13 (2 h)

**Material:**
- None.

**Code:**
- Build a simple neural network using `tensor_t`:
  ```c
  // Each weight/bias is a tensor_t
  // Forward: a = sigmoid(tensor_matmul(w1, x) + b1)
  // Loss: 0.5 * sum((a - y)^2)
  // Backward: loss_tensor->backward()
  ```
- Retrain XOR using the new autograd system. In `examples/xor_auto.c`:
  ```c
  tensor_t *x = tensor_new_shaped([2, 1]);
  tensor_t *y = tensor_new_shaped([1, 1]);
  tensor_t *w1 = tensor_new_randn([4, 2]);
  tensor_t *b1 = tensor_new_zeros([4, 1]);
  // ... forward, compute loss, backward, update weights ...
  ```
- **Goal:** Same convergence as Phase 1, but using autodiff instead of hand-written backprop.
- **Commit.** Write `docs/phase2.md`.

---

#### Día 14 (1.5–2 h, optional buffer or start Phase 3)

**Material:**
- If autodiff is working and clean, you can start Phase 3 early.
- If there are bugs, use this day to debug.

**Code:**
- Optional: Implement a simple **Adam optimizer sketch** in `src/optim.h`:
  ```c
  typedef struct {
      double lr;
      double beta1, beta2;
      double eps;
      // m (first moment), v (second moment) for each parameter
  } adam_t;
  ```
- Update `examples/xor_auto.c` to use Adam instead of vanilla SGD.
- Retrain, compare speed.

---

## Phase 3: Tensor architecture and ggml study (4 days)

**Objective:**
- Understand **how production ML systems organize tensors and memory** by studying ggml and llama2.c.
- **Refactor your autograd** to use a design closer to ggml's (graph-based, memory pooling).
- Document the connection between your implementation and real systems.

### Semana 5

#### Día 15 (1.5–2 h)

**Material:**
- Read **ggml.h** (the header file from the ggml repo). Focus on:
  - `ggml_tensor` struct: data pointer, grad pointer, operation enum, operand list, metadata.
  - `ggml_cgraph` (computation graph): stores a list of all tensors in topological order.
  - `ggml_compute_forward` and `ggml_compute_backward` — how ops are dispatched.
- Skim the **Hugging Face blog post on ggml**. Get the big picture.

**Code:**
- In `docs/tensor_design.md`, write notes:
  ```markdown
  # Tensor Architecture Comparison

  ## Your implementation (Phase 2)
  - Each tensor is a node with children pointers.
  - Backward is recursive via topological sort.
  - Pros: Simple, explicit.
  - Cons: Fragmented memory, slower dispatch.

  ## ggml design
  - Tensors live in a global context with a memory pool.
  - Operations add nodes to a computation graph.
  - Forward/backward dispatch via operation enum.
  - Pros: Cache-friendly, batch-friendly, quantization-ready.
  - Cons: More complex bookkeeping.

  ## Lessons for next iteration
  - Use a global context for memory management.
  - Store operations as enums, not function pointers.
  - Build a graph structure (list of nodes) separate from tensor tree.
  ```

---

#### Día 16 (2–2.5 h)

**Material:**
- Read **karpathy/llama2.c** (the single-file inference code). Focus on:
  - How weights are stored (interleaved in a single file or contiguous block?).
  - How forward pass iterates through layers (loop over layer count, dispatch per operation).
  - KV cache structure (what is it, how does it reduce computation?).

**Code:**
- In `src/tensor_v2.h`, sketch a **graph-based design** (don't fully implement, just outline):
  ```c
  typedef enum {
      OP_ADD, OP_MUL, OP_MATMUL, OP_TANH, OP_RELU, ...
  } op_type_t;

  typedef struct tensor_s {
      double *data, *grad;
      shape_t shape;
      op_type_t op;
      struct tensor_s *operands[2];  // input tensors
      // ... other fields ...
  } tensor_t;

  typedef struct {
      tensor_t **nodes;
      int n_nodes;
      // ... memory pool ...
  } graph_t;
  ```
- Write a brief **refactoring plan** in `docs/phase3.md`:
  ```markdown
  # Phase 3: ggml-inspired design

  ## What I learned from ggml/llama2.c
  - Operation dispatch via enum is faster than function pointers.
  - Global memory pools reduce fragmentation.
  - Computation graphs separate from tensor DAG enable easier optimization.

  ## Next steps (not implemented in Phase 3, but for future work)
  1. Migrate from function pointers to op enums.
  2. Add a graph_t context that manages memory.
  3. Implement batch operations (multiple examples at once).
  ```

---

#### Día 17 (2 h)

**Material:**
- No new material — polish and document.

**Code:**
- Update `README.md` with the full project story:
  ```markdown
  # ML from Scratch in C

  A journey through machine learning fundamentals in pure C.

  ## What this repo contains
  - **Phase 1:** Matrix library + 2-layer MLP with explicit backprop (trains on XOR).
  - **Phase 2:** Automatic differentiation engine with N-dimensional tensors.
  - **Phase 3:** Study of tensor architecture (ggml, llama2.c) and refactoring ideas.

  ## Key insights
  - Backpropagation is just chain rule applied systematically.
  - Autodiff separates model definition from gradient computation.
  - Production systems (ggml) use operation enums + global memory pools for efficiency.

  ## Inspiration
  - tinn, genann (minimal MLPs)
  - micrograd.c, smdaa/teeny-autograd-c (autodiff)
  - ggml, llama2.c (production design)

  See `docs/roadmap.md` for the full learning path.
  ```

---

#### Día 18 (1.5–2 h)

**Material:**
- Final polish — no new reading.

**Code:**
- Write `docs/roadmap.md`:
  ```markdown
  # Learning Roadmap

  ## 1. Foundations: matrix library + backprop (Days 1–7)
  Inspired by tinn (~200 lines) and genann, you'll implement:
  - Matrix operations (mul, add, element-wise).
  - A 2-layer feedforward network.
  - Backpropagation by hand, using the chain rule.

  The goal: understand what a neural network *computes* and how gradients flow.

  ## 2. Autodiff: N-dimensional tensors (Days 8–14)
  Inspired by micrograd.c and teeny-autograd-c, you'll build:
  - A scalar autodiff engine (like micrograd).
  - Generalize to N-dimensional tensors with broadcasting.
  - Retrain XOR using autograd instead of explicit backprop.

  The goal: learn that gradients can be computed automatically, and understand the structure of a computation graph.

  ## 3. Production design: ggml + llama2.c (Days 15–18)
  Study (don't reimplement) how real systems work:
  - ggml's tensor library: operation enums, global contexts, memory pooling.
  - llama2.c's inference: weight loading, forward pass, KV cache.

  Sketch ideas for a v2 of your autograd (enum-based dispatch, global memory).

  The goal: see how ideas scale from toy code to production systems.

  ## 4. Portfolio + refinement (3 final days)
  Polish the repo, write clear docs, and prepare for interviews.

  ---

  ## How this helps your career
  You now have:
  - A **clear implementation** of core ML concepts from first principles.
  - **Portfolio evidence** that you understand low-level C, memory management, and numerical computing.
  - **Concrete insights** about how production ML systems are built.

  This is valuable for:
  - Infrastructure/systems roles (ML platforms, CUDA optimization).
  - Quantitative finance (numerical computing, performance).
  - Quantum computing (tensor operations, linear algebra).
  - Any team doing ML research or deployment in languages like C/C++/Rust.
  ```

- Commit everything. Write a final summary in `docs/phase3.md`.

---

## Phase 4: Portfolio and refinement (3 days)

**Objective:**
- Make the repo **presentation-ready** for GitHub, interviews, and your CV.
- Write clean examples and documentation.
- Ensure everything compiles and runs cleanly.

### Semana 6

#### Día 19 (1.5 h)

**Material:**
- None.

**Code:**
- Verify everything compiles:
  ```bash
  gcc -O2 -Wall -std=c99 src/mat.c src/tensor.c src/nn.c examples/xor.c -lm -o xor
  ./xor
  ```
- Add **comments in English** to `examples/xor.c` and `examples/xor_auto.c`:
  ```c
  // Forward pass: compute network output
  nn_forward(net, x);
  
  // Backward pass: compute gradients via chain rule
  nn_backward(net, x, y, learning_rate);
  
  // Update weights: gradient descent step
  // (incorporated in nn_backward)
  ```
- Write a `Makefile` or `CMakeLists.txt` so compilation is one command.
- Add a `.gitignore`:
  ```
  *.o
  *.a
  xor
  xor_auto
  test_*
  ```

---

#### Día 20 (1.5 h)

**Material:**
- None.

**Code:**
- Polish `README.md`:
  ```markdown
  # ML from Scratch in C

  Implementing machine learning fundamentals in pure C, from matrix operations through automatic differentiation to tensor libraries inspired by production systems.

  ## Quick start

  ```bash
  make
  ./xor
  ./xor_auto
  ```

  The first trains XOR with explicit backprop. The second uses autodiff. Both should converge.

  ## What's inside

  - **src/mat.h, mat.c**: 2D matrix library (mul, add, transpose).
  - **src/tensor.h, tensor.c**: N-dimensional autodiff engine (scalar, then shaped tensors).
  - **src/nn.h, nn.c**: 2-layer feedforward MLP.
  - **examples/**: XOR training examples (explicit backprop and autodiff).
  - **tests/**: Numerical gradient checking, validation.
  - **docs/roadmap.md**: Detailed learning path.

  ## Learning path

  1. **Matrices and backprop** (Days 1–7): Implement a simple NN, train on XOR by hand-writing gradients.
  2. **Autodiff** (Days 8–14): Build an automatic differentiation engine. Retrain XOR with autodiff.
  3. **Production design** (Days 15–18): Study ggml and llama2.c. Understand tensor architecture at scale.
  4. **Polish** (Days 19–21): Documentation, tests, portfolio-ready code.

  See `docs/roadmap.md` for the full story and references.

  ## References

  **Minimal implementations:**
  - [tinn](https://github.com/glouw/tinn) — 200-line neural network in C.
  - [genann](https://github.com/codeplea/genann) — Simple NN library with arbitrary depth.

  **Autodiff:**
  - [micrograd.c](https://github.com/Jaykef/micrograd.c) — Karpathy's micrograd ported to C.
  - [teeny-autograd-c](https://github.com/smdaa/teeny-autograd-c) — Tensor autodiff with excellent blog.

  **Production:**
  - [ggml](https://github.com/ggml-org/ggml) — Tensor library for inference, quantization, and training.
  - [llama2.c](https://github.com/karpathy/llama2.c) — Llama 2 inference in one file of C.

  ## Lessons

  - **Backpropagation is just the chain rule applied systematically.**
  - **Automatic differentiation separates model definition from gradient computation.**
  - **Production systems use operation enums and memory pooling for efficiency and scale.**

  ---

  Author: [Your name]  
  License: MIT (or your choice)
  ```

- Update `docs/roadmap.md` with the full, final version (as shown above).

---

#### Día 21 (1 h)

**Material:**
- None.

**Code:**
- Final **git commits** and cleanup:
  ```bash
  git add -A
  git commit -m "Phase 4: Portfolio-ready code, clean docs"
  git log --oneline  # verify commits are clear
  ```
- (Optional) Push to GitHub:
  ```bash
  git remote add origin https://github.com/yourusername/ml-from-scratch-c.git
  git branch -M main
  git push -u origin main
  ```

---

## Timeline summary

| Week | Phase | Focus | Days | Goals |
|------|-------|-------|------|-------|
| 1 | 1 | Matrix lib + NN basics | 1–4 | mat_mul, mat_add, nn_forward, first tests |
| 2 | 1 | Backprop + training | 5–7 | nn_backward, XOR convergence, git commit |
| 3 | 2 | Scalar autodiff | 8–10 | Micrograd-style engine, gradient checking |
| 4 | 2 | Tensor generalization | 11–14 | N-dim tensors, retrain XOR with autograd, polish |
| 5 | 3 | ggml + architecture study | 15–18 | Read ggml/llama2.c, sketch v2 design, document |
| 6 | 4 | Portfolio + final polish | 19–21 | README, Makefile, clean code, GitHub ready |

**Total:** 21 days ≈ 10–12 weeks at 3–4 hours/week.

---

## Grading your own progress

### Phase 1: Done when
- XOR trains and loss decreases visibly.
- Matrix ops are correct (test on known examples).
- Code compiles without warnings.

### Phase 2: Done when
- Autograd produces correct gradients (numerical gradient checker passes).
- XOR retrains with autodiff and converges.
- Tensor shapes work (matmul with different dimensions, broadcasting).

### Phase 3: Done when
- You understand the difference between your scalar graph and ggml's design.
- You've sketched a v2 refactoring (even if not coded).
- You can explain why operation enums + global memory > function pointers + scattered allocation.

### Phase 4: Done when
- README is clear and complete.
- Everything compiles (`make && ./xor`).
- Code is commented and formatted nicely.
- Git history tells a story (small, logical commits).

---

## FAQ

**Q: What if I get stuck on backprop?**  
A: It's the hardest part. Spend extra time on Día 5 and 6 if needed. Draw the computation graph on paper. Test with a single example by hand. Use print statements to debug gradient values.

**Q: Is 3–4 hours/week realistic?**  
A: Yes, if you're disciplined. One 2–3h session on the weekend, maybe a 1–1.5h session midweek. Breaks help — don't try to push 4h straight.

**Q: Should I implement GEMM optimization?**  
A: Not in this plan. BLIS/GEMM is HPC; you're learning ML + C fundamentals. Understanding tensor memory layout (Phase 3) is more valuable for you.

**Q: Can I skip Phase 3?**  
A: You can, but don't. Phase 3 is where you see how your toy code scales to production. It's the "aha!" moment. Takes 4 days and gives huge returns for interviews.

**Q: What if I finish early?**  
A: Nice! Extend Phase 2 with Conv2D layers, or implement a simple MNIST classifier. Or read the full ggml source.

---

## Commit messages template

```
Día 1: Init repo, matrix_t struct
Día 2: Implement mat_new, mat_free, mat_print
Día 3: Implement mat_mul, mat_add, sigmoid
Día 4: Begin nn_t struct and nn_forward
Día 5: Implement nn_backward (explicit backprop)
Día 6: Debug training loop, XOR convergence
Día 7: Polish Phase 1, write docs
Día 8: Start tensor_t struct, scalar autodiff
Día 9: Implement tensor_add, tensor_mul, backpropagation
Día 10: Test numerical gradient checker
...
```

---

Good luck. You've got this.
