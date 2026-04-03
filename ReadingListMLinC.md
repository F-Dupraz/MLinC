# A reading list for building ML from scratch in C

**Twenty-nine freely available papers, surveys, and technical blog posts** cover every concept you need to implement a neural network library in C — from matrix math to automatic differentiation to production tensor systems like ggml. This list is organized by your three project phases, prioritizes pedagogical clarity for a 2nd-year CS student, and includes only resources that are free to access right now. Every link points to arXiv, an author's site, or an open-access journal. The Dongarra/Gustavson/Karp paper is assessed at the end: skip it — a better, free alternative exists.

---

## Phase 1: Matrix operations, neural networks, and backpropagation

This phase covers the mathematical and algorithmic foundations: how matrices work in numerical computing, what a neural network actually computes, and how backpropagation trains one. Start with the beginner resources, then read the seminal papers for depth.

**Michael Nielsen — *Neural Networks and Deep Learning* (2015)**
Free online book: [neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com/)
The best starting point for Phase 1. Chapter 2 derives backpropagation from first principles with careful notation, and the entire book builds a working digit-recognition MLP. Written for exactly your level — undergraduate math, no ML prerequisites. **Difficulty: Beginner–Intermediate.**

**Andrej Karpathy — "Hacker's Guide to Neural Networks" (2014)**
Blog post: [karpathy.github.io/neuralnets](http://karpathy.github.io/neuralnets/)
A code-first walkthrough of neural networks as "real-valued circuits." Covers forward passes, numerical vs. analytic gradients, and backpropagation using simple code examples — the perfect mental model for someone writing C, not Python. Karpathy himself recommends supplementing with CS231n notes (next entry). **Difficulty: Beginner.**

**Stanford CS231n Course Notes — "Backpropagation, Intuitions" (2015+)**
Course notes: [cs231n.github.io/optimization-2](https://cs231n.github.io/optimization-2/)
Supplementary handout on matrix backprop: [cs231n.stanford.edu/handouts/linear-backprop.pdf](https://cs231n.stanford.edu/handouts/linear-backprop.pdf)
Explains the chain rule visually with circuit diagrams and covers staged gradient computation (add gates, multiply gates, max gates). The matrix backprop handout is directly relevant to implementing layer-wise backpropagation over matrix operations in C. **Difficulty: Beginner–Intermediate.**

**David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams — "Learning representations by back-propagating errors" (1986)**
Free PDF: [cs.toronto.edu/~hinton/absps/naturebp.pdf](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)
The paper that launched modern neural network training. A short, dense 3-page *Nature* article showing that gradient descent through hidden layers can learn internal representations. Read this after Nielsen's Chapter 2 — you'll appreciate both the elegance and the historical weight. **Difficulty: Intermediate.**

**Yann LeCun, Léon Bottou, Genevieve B. Orr, Klaus-Robert Müller — "Efficient BackProp" (1998)**
Free PDF: [cseweb.ucsd.edu/classes/wi08/cse253/Handouts/lecun-98b.pdf](https://cseweb.ucsd.edu/classes/wi08/cse253/Handouts/lecun-98b.pdf)
A practical masterclass in making backpropagation work: **input normalization, weight initialization, learning rate schedules**, and second-order methods. Directly applicable when your C MLP trains but converges poorly — this paper tells you why and what to fix. **Difficulty: Intermediate.**

**Frank Rosenblatt — "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" (1958)**
Free PDF: [homepages.math.uic.edu/~lreyzin/papers/rosenblatt58.pdf](https://homepages.math.uic.edu/~lreyzin/papers/rosenblatt58.pdf)
The original perceptron paper. The simplest neural network model — a weighted sum plus threshold — maps directly to your first C implementation. Historical notation takes some effort, but the core idea is straightforward. **Difficulty: Intermediate.**

**Warren S. McCulloch, Walter Pitts — "A Logical Calculus of the Ideas Immanent in Nervous Activity" (1943)**
Free PDF: [cs.cmu.edu/~epxing/Class/10715/reading/McCulloch.and.Pitts.pdf](https://www.cs.cmu.edu/~epxing/Class/10715/reading/McCulloch.and.Pitts.pdf)
The very first mathematical model of an artificial neuron. Heavy on mathematical logic — skim it for historical context and the core idea that simple threshold units can compute logical functions. **Difficulty: Advanced.**

**William Layton, Myron Sussman — *Numerical Linear Algebra* (free textbook)**
Free PDF: [people.sc.fsu.edu/~jburkardt/classes/nla_2015/numerical_linear_algebra.pdf](https://people.sc.fsu.edu/~jburkardt/classes/nla_2015/numerical_linear_algebra.pdf)
A free alternative to Golub & Van Loan's *Matrix Computations* (which has no free chapters). Covers Gaussian elimination, LU/Cholesky factorization, matrix norms, condition numbers, and floating-point error analysis — the numerical grounding you need to understand why naive matrix operations in C can go wrong. **Difficulty: Intermediate.**

---

## Phase 2: Automatic differentiation, computation graphs, and tensors

This phase is where your project transitions from a hand-coded MLP to a general-purpose system. You need to understand how autodiff engines decompose functions into elementary operations, build computation graphs, and propagate gradients automatically.

**Christopher Olah — "Calculus on Computational Graphs: Backpropagation" (2015)**
Blog post: [colah.github.io/posts/2015-08-Backprop](https://colah.github.io/posts/2015-08-Backprop/)
Start here. Beautifully illustrated explanation of how derivatives flow through computation graphs via the chain rule. Explains both forward-mode and reverse-mode differentiation as **path-summing algorithms on directed acyclic graphs**. The single best visual introduction before you write any autodiff code. **Difficulty: Beginner.**

**Robert E. Wengert — "A Simple Automatic Derivative Evaluation Program" (1964)**
Free PDF: [cs.princeton.edu/courses/archive/fall19/cos597C/files/wengert1964.pdf](https://www.cs.princeton.edu/courses/archive/fall19/cos597C/files/wengert1964.pdf)
A remarkable 2-page paper that invented forward-mode AD. Introduces the core idea of decomposing a function into elementary operations ("Wengert list") and propagating derivatives through subroutines — exactly the approach you'll implement in C. Worth reading for its brevity alone. **Difficulty: Beginner.**

**Yu-Hsueh Fang, He-Zhe Lin, Jie-Jyun Liu, Chih-Jen Lin — "A Step-by-step Introduction to the Implementation of Automatic Differentiation" (2024)**
arXiv: [arxiv.org/abs/2402.16020](https://arxiv.org/abs/2402.16020)
The most directly hands-on resource for Phase 2. Bridges the gap between mathematical concepts and code, showing how to build and traverse a computation graph, implement forward-mode AD, and extend to reverse mode. Written specifically for teaching students to *implement* an AD system. **Difficulty: Beginner–Intermediate.**

**Atılım Güneş Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind — "Automatic Differentiation in Machine Learning: A Survey" (2018)**
arXiv: [arxiv.org/abs/1502.05767](https://arxiv.org/abs/1502.05767)
JMLR PDF: [jmlr.org/papers/volume18/17-468/17-468.pdf](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf)
**The definitive survey on AD for ML.** Covers forward mode, reverse mode, operator overloading vs. source transformation, and the precise relationship between backpropagation and reverse-mode AD. Read sections 1–4 carefully; the rest is reference material. This paper alone could serve as the theoretical backbone of Phase 2. **Difficulty: Intermediate.**

**Charles C. Margossian — "A Review of Automatic Differentiation and Its Efficient Implementation" (2019)**
arXiv: [arxiv.org/abs/1811.05031](https://arxiv.org/abs/1811.05031)
Implementation-focused where Baydin is theory-focused. Covers operator overloading, expression templates, **memory management (taping, checkpointing)**, and efficiency considerations. Directly relevant to the engineering decisions you'll face building an autodiff engine in C. **Difficulty: Intermediate.**

**Andreas Griewank — "On Automatic Differentiation" (1989)**
Free PDF: [softlib.rice.edu/pub/CRPC-TRs/reports/CRPC-TR89003.pdf](https://softlib.rice.edu/pub/CRPC-TRs/reports/CRPC-TR89003.pdf)
A foundational technical report by the leading authority on AD. Covers the mathematical basis of forward and reverse mode, complexity analysis, and the chain rule applied to evaluation traces. This is the freely available entry point to Griewank's work (his book *Evaluating Derivatives* is the definitive reference but not free). **Difficulty: Intermediate–Advanced.**

**Michael Collins — "Computational Graphs, and Backpropagation" (course notes, Columbia University, ~2020)**
Free PDF: [cs.columbia.edu/~mcollins/ff2.pdf](http://www.cs.columbia.edu/~mcollins/ff2.pdf)
Rigorous course notes that formalize computation graphs with precise definitions (nodes, edges, leaf variables), then derive both forward and backward algorithms with full proofs using Jacobians. Excellent for a CS student who wants mathematical rigor alongside clear pseudocode. **Difficulty: Beginner–Intermediate.**

**Martín Abadi et al. — "TensorFlow: A System for Large-Scale Machine Learning" (2016)**
arXiv: [arxiv.org/abs/1605.08695](https://arxiv.org/abs/1605.08695)
Describes TensorFlow's static computation graph architecture: how tensor operations become graph nodes and how autodiff extends the graph with gradient nodes. Read this as a reference architecture for how computation graphs, tensor operations, and AD work together at production scale. **Difficulty: Intermediate.**

**Adam Paszke et al. — "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (2019)**
arXiv: [arxiv.org/abs/1912.01703](https://arxiv.org/abs/1912.01703)
Describes PyTorch's *dynamic* computation graph (define-by-run, tape-based) approach, contrasting with TensorFlow's static graphs. Read sections on the autograd engine and how tensors track computation history. Understanding **both paradigms** — static and dynamic — informs your own design decisions. **Difficulty: Intermediate.**

**Tamara G. Kolda, Brett W. Bader — "Tensor Decompositions and Applications" (2009)**
Free PDF: [kolda.net/publication/TensorReview.pdf](https://www.kolda.net/publication/TensorReview.pdf)
The definitive survey on tensors as N-dimensional arrays. Section 2 provides rigorous notation for tensor operations (fibers, slices, unfoldings, n-mode products) that ground your understanding of what an N-dimensional tensor data structure must support. Heavy on multilinear algebra — read Section 2 and skim the rest. **Difficulty: Intermediate–Advanced.**

---

## Phase 3: Tensor library architecture, memory layout, and production ML systems

This phase is about systems engineering: how real ML libraries organize memory, optimize computation, and handle quantization. The resources here range from the NumPy design paper to ggml internals.

**Charles R. Harris et al. — "Array programming with NumPy" (2020)**
Open access: [nature.com/articles/s41586-020-2649-2](https://www.nature.com/articles/s41586-020-2649-2)
The definitive reference for tensor library design. Covers **memory layout, strides, broadcasting, views vs. copies, and vectorization**. NumPy's internal C implementation uses contiguous memory with stride-based indexing and shape metadata — exactly the architecture you'll replicate. **Difficulty: Beginner–Intermediate.**

**Kazushige Goto, Robert A. van de Geijn — "Anatomy of High-Performance Matrix Multiplication" (2008)**
Free PDF: [cs.utexas.edu/~flame/pubs/GotoTOMS.pdf](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS.pdf)
The essential paper for writing efficient GEMM in C. Explains **cache hierarchy exploitation, loop tiling, memory blocking**, and TLB management with remarkable clarity. GEMM is the computational workhorse of neural networks — this paper shows you why naive triple-nested loops are 10–100× slower than necessary and how to fix it. **Difficulty: Intermediate–Advanced.**

**Omri Mallis — "Understanding How LLM Inference Works with llama.cpp" (2023)**
Blog post: [omrimallis.com/posts/understanding-how-llm-inference-works-with-llama-cpp](https://www.omrimallis.com/posts/understanding-how-llm-inference-works-with-llama-cpp/)
An engineering-focused walkthrough of llama.cpp's internals: **ggml tensor data structures (shape, strides, memory layout), the transformer computation graph, and the KV cache**. Written for engineers, not ML researchers, with direct references to source code. The best single resource for understanding how ggml works in practice. **Difficulty: Beginner–Intermediate.**

**GGML Tensor Library Architecture Documentation (DeepWiki, community-maintained, 2023–2025)**
Documentation: [deepwiki.com/ggml-org/llama.cpp/4-ggml-tensor-library](https://deepwiki.com/ggml-org/llama.cpp/4-ggml-tensor-library)
The most comprehensive technical documentation of ggml's architecture: computation graph execution, custom memory management via contiguous arenas (`ggml_context`), backend abstraction, quantization type system, and the `vec_dot` operation. This is the reference implementation of a production tensor library in pure C. **Difficulty: Intermediate.**

**Georgi Gerganov — "Bringing Whisper and LLaMA to the Masses" (Changelog Podcast, 2023)**
Transcript/audio: [changelog.com/podcast/532](https://changelog.com/podcast/532)
First-person account from ggml's creator explaining why he built a tensor library in C, the design goals (strict memory management, no runtime allocation, multithreading), and how the architecture enabled rapid implementation of whisper.cpp and llama.cpp. Crucial context for understanding the *philosophy* behind production ML systems in C. **Difficulty: Beginner.**

**Vijay Janapa Reddi et al. — *Machine Learning Systems* (Harvard CS249r, open-access textbook, 2023–2025)**
Free online: [mlsysbook.ai](https://mlsysbook.ai/)
A comprehensive open-access textbook covering the full ML systems stack — hardware, memory hierarchy, model optimization, quantization, and deployment. Pitched at the undergraduate level and covers the systems-level concerns (memory management, compute efficiency, model compression) that define Phase 3. **Difficulty: Beginner–Intermediate.**

**Tianqi Chen et al. — "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning" (2018)**
arXiv: [arxiv.org/abs/1802.04799](https://arxiv.org/abs/1802.04799)
Shows how a production ML compiler handles **operator fusion, memory layout optimization, loop tiling**, and hardware-specific code generation. Illustrates the design space between high-level ML frameworks and low-level C — exactly where your tensor library sits. **Difficulty: Advanced.**

**Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer — "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (2022)**
arXiv: [arxiv.org/abs/2208.07339](https://arxiv.org/abs/2208.07339)
Foundational paper on quantized inference. Explains vector-wise quantization, mixed-precision decomposition, and how outlier features affect INT8 matrix multiplication — the core techniques behind ggml's block-based quantization formats (Q4_0, Q4_K_M, etc.). **Difficulty: Intermediate–Advanced.**

**Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh — "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2022)**
arXiv: [arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
Demonstrates compressing LLM weights to **3–4 bits per weight** with minimal accuracy loss using second-order information. This is the quantization method used by the GGUF ecosystem. Understanding GPTQ explains why ggml's quantized data types are structured the way they are. **Difficulty: Advanced.**

**Apoorv Nandan — tensor.h: "Creating a Tiny Tensor Library in Raw C" (2024)**
GitHub: [github.com/apoorvnandan/tensor.h](https://github.com/apoorvnandan/tensor.h)
A pedagogical implementation of a tensor library from scratch in pure C. Covers N-dimensional array representation (shape, strides, values), tensor operations, autodiff via computation graphs, and building neural networks. Clean, readable code — an excellent companion to your own implementation. **Difficulty: Beginner–Intermediate.**

---

## The Dongarra/Gustavson/Karp paper: skip it

The user asked about **"Implementing Linear Algebra Algorithms for Dense Matrices on a Vector Pipeline Machine"** (J.J. Dongarra, F.G. Gustavson, A. Karp, 1984, *SIAM Review*). After investigation, this paper is **not recommended** for three reasons:

1. **It's paywalled.** The paper sits behind SIAM's paywall at `epubs.siam.org`. No freely accessible PDF was found on institutional repositories or the authors' pages, which immediately disqualifies it from this list.

2. **The hardware model is obsolete.** The paper optimizes for Cray-era vector pipeline machines — a hardware paradigm that no longer exists. The insights about vectorization and pipeline utilization don't transfer to modern CPU cache hierarchies.

3. **A superior free alternative exists.** Goto & Van de Geijn's "Anatomy of High-Performance Matrix Multiplication" (2008) covers the same fundamental idea — restructuring linear algebra to exploit hardware — but targets **modern cache-based architectures**, is freely available, and is far more accessible for a student. Read that instead.

The Dongarra paper matters historically: it helped establish the principles that led to BLAS, LAPACK, and eventually modern libraries like OpenBLAS and ggml. But for a 2nd-year student building ML in C, the Goto paper delivers the same conceptual payoff with directly applicable, modern content.

---

## A suggested reading order across all three phases

Rather than reading every paper sequentially, here is a practical path through the material that builds understanding efficiently:

- **Week 1–2 (Foundations):** Nielsen's free book (Chapters 1–2), Karpathy's blog post, CS231n backprop notes
- **Week 3 (Historical depth):** Rumelhart et al. (1986), then LeCun et al. "Efficient BackProp"
- **Week 4 (Autodiff foundations):** Olah's blog post, Wengert (1964), then Fang et al. (2024) step-by-step tutorial
- **Week 5 (Autodiff theory):** Baydin et al. survey (Sections 1–4), Margossian review for implementation concerns
- **Week 6 (Systems design):** NumPy paper, Goto & Van de Geijn, tensor.h GitHub repo
- **Week 7 (Production systems):** Omri Mallis llama.cpp blog, GGML DeepWiki docs, Gerganov podcast
- **Week 8 (Advanced):** TVM paper, LLM.int8(), GPTQ, Harvard ML Systems textbook (selected chapters)

The TensorFlow and PyTorch papers serve as reference architectures — read them when designing your computation graph system to understand both static and dynamic approaches. Kolda & Bader is reference material for tensor algebra notation when you need it.