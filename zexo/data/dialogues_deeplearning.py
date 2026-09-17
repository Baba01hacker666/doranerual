"""Zexo Conversational Dataset: Deep Learning & LLMs (Dialogues 25-37)."""

DIALOGUES_DEEPLEARNING = [
    # Dialogue 25: Self-Attention Mechanism
    [
        ("Can you explain the mathematical formulation of scaled dot-product attention in transformers?",
         "Scaled dot-product attention maps a set of query vectors (Q), key vectors (K), and value vectors (V) to an output representation:\n\n$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V$$\n\nHere is what happens step by step:\n1. $Q K^T$: The dot product measures pairwise compatibility (similarity) between every token's query and all other tokens' keys, yielding an $(N \\times N)$ raw score matrix.\n2. Division by $\\sqrt{d_k}$: Scales down the raw scores.\n3. $\\text{softmax}(\\cdot)$: Normalizes the scores across rows into positive attention weights that sum to 1.0.\n4. Multiplication by $V$: Computes a weighted sum of the value vectors, blending relevant token information into each representation."),
        ("Why do we divide by $\\sqrt{d_k}$? What happens if we omit it?",
         "Suppose the components of $q$ and $k$ are independent random variables with zero mean and unit variance ($E[q_i] = E[k_i] = 0$, $\\text{Var}(q_i) = \\text{Var}(k_i) = 1$). The dot product $q \\cdot k = \\sum_{i=1}^{d_k} q_i k_i$ has a mean of 0 and a variance equal to $d_k$.\n\nAs the projection dimension $d_k$ grows large (e.g., $d_k = 64$ or $128$), the magnitude of the dot products grows proportionally to $\\sqrt{d_k}$. When large numbers enter the softmax function, the largest value dominates, driving softmax probabilities toward 0 or 1. In this saturated region, the softmax gradient $\\frac{\\partial \\text{softmax}}{\\partial z}$ vanishes to nearly zero, stalling training. Dividing by $\\sqrt{d_k}$ normalizes the variance back to 1.0, preserving healthy gradient flow throughout backpropagation."),
        ("That makes total sense mathematically! It prevents softmax saturation.",
         "Precisely. It is a deceptively simple scaling factor that makes training deep attention networks stable across arbitrary head dimensions.")
    ],

    # Dialogue 26: Multi-Head Attention Subspaces
    [
        ("Why do transformer architectures use Multi-Head Attention rather than a single large attention head?",
         "Multi-Head Attention (MHA) allows the model to jointly attend to information from different representation subspaces at different positions. A single attention head can only produce one distribution of attention weights per token, effectively forcing it to average out competing relationships.\n\nWith multiple heads (e.g., 8, 16, or 32 heads), different heads specialize:\n- One head might track syntactic dependencies (e.g., matching a verb to its subject).\n- Another head tracks coreference resolution (matching pronouns like 'it' or 'they' to their antecedents).\n- Another head attends to local adjacent tokens (n-gram context).\n- Yet another head captures long-range document-level semantic themes."),
        ("How is this computed without multiplying the computational cost by the number of heads?",
         "By partitioning the model dimension $d_{\\text{model}}$ into $h$ heads, each with dimension $d_k = d_{\\text{model}} / h$. For example, with $d_{\\text{model}} = 4096$ and $h = 32$, each head has $d_k = 128$. Linear projection matrices $W_Q, W_K, W_V$ project the input into these subspaces in parallel. After independent scaled dot-product attention in each head, their outputs are concatenated along the feature dimension back to $(N \\times d_{\\text{model}})$ and projected through $W_O$. The total FLOP count is essentially identical to running a single attention head with dimension $d_{\\text{model}}$.")
    ],

    # Dialogue 27: Rotary Position Embedding (RoPE)
    [
        ("What is Rotary Position Embedding (RoPE), and why has it largely replaced learned absolute embeddings?",
         "Rotary Position Embedding (RoPE), introduced in RoFormer and adopted by LLaMA and Zexo, encodes positional information by rotating query and key vectors in the 2D complex planes of feature pairs according to their sequence position $m$.\n\nInstead of adding an embedding vector to token embeddings ($x_m + p_m$), RoPE multiplies the query $q_m$ and key $k_n$ by orthogonal block-diagonal rotation matrices $R_m$ and $R_n$. Crucially, the inner product becomes:\n$$\\langle R_m q, R_n k \\rangle = q^T R_m^T R_n k = q^T R_{n - m} k$$\nThe inner product depends solely on the relative distance $(n - m)$ between tokens rather than their absolute positions $m$ and $n$."),
        ("Why does this give RoPE better sequence length extrapolation properties?",
         "Because the relative rotation angle naturally decays with distance as frequencies increase across feature dimensions. The network learns relative syntactic distances directly. When extending context windows during inference, techniques like RoPE frequency scaling (e.g., YaRN or Linear RoPE Scaling) can interpolate unseen positions smoothly without re-initializing or re-training base positional tables.")
    ],

    # Dialogue 28: SwiGLU Activation Function
    [
        ("What is SwiGLU, and why is it preferred over ReLU or standard GELU in modern LLMs?",
         "SwiGLU is a Gated Linear Unit (GLU) variant that combines the SiLU (Swish) activation function with multiplicative gating. In a standard feed-forward layer (MLP), an intermediate linear transformation passes through an activation function before output projection: $\\text{FFN}(x) = \\text{act}(x W_1) W_2$.\n\nIn SwiGLU, the input is projected twice—once through a gate matrix $W_1$ and once through a value matrix $W_3$. The gate is activated with SiLU and multiplied element-wise by the value projection:\n$$\\text{SwiGLU}(x) = \\left(\\text{SiLU}(x W_1) \\otimes (x W_3)\\right) W_2$$\nThis multiplicative gating enables the network to dynamically filter, suppress, or pass specific features, providing higher expressivity and faster empirical convergence."),
        ("Since SwiGLU adds a third weight matrix $W_3$, does it increase the parameter count?",
         "If the intermediate dimension remained $4 \\times d_{\\text{model}}$, adding $W_3$ would increase MLP parameters by 50%. To maintain exact parameter parity with standard $4 \\times d_{\\text{model}}$ FFNs, architectures typically scale the hidden dimension down to approximately $\\frac{8}{3} d_{\\text{model}}$ (or rounded to the nearest multiple of 256/64 for GPU tensor core alignment). Even with identical parameter budgets, SwiGLU consistently outperforms ReLU and GELU across perplexity benchmarks.")
    ],

    # Dialogue 29: RMSNorm vs LayerNorm
    [
        ("What is RMSNorm, and how does it differ from Layer Normalization?",
         "Layer Normalization (LayerNorm) normalizes activations across the feature dimension by computing both the mean $\\mu$ and variance $\\sigma^2$:\n$$\\text{LayerNorm}(x) = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\odot \\gamma + \\beta$$\nRMSNorm (Root Mean Square Normalization) hypothesizes that the re-scaling property of LayerNorm is what stabilizes training, while mean-centering ($\\,x - \\mu\\,$) is computationally redundant. RMSNorm simplifies the normalization by scaling solely by the root mean square:\n$$\\text{RMS}(x) = \\sqrt{\\frac{1}{d} \\sum_{i=1}^d x_i^2 + \\epsilon}, \\quad \\text{RMSNorm}(x) = \\frac{x}{\\text{RMS}(x)} \\odot \\gamma$$\nRMSNorm eliminates two passes over data (computing mean and subtracting it), saving approximately 20-30% of normalization compute while maintaining identical training stability."),
        ("Where is RMSNorm positioned in modern transformer layers?",
         "Modern models use 'Pre-RMSNorm', placing RMSNorm before the multi-head attention and feed-forward blocks ($x + \\text{Attention}(\\text{RMSNorm}(x))$) rather than after ('Post-Norm'). Pre-Norm preserves a clean, unmodulated residual identity stream from the bottom layer to the top layer, completely preventing vanishing or exploding gradients during backward passes.")
    ],

    # Dialogue 30: Backpropagation & Computational Graphs
    [
        ("How does reverse-mode automatic differentiation compute gradients in deep neural networks?",
         "Reverse-mode automatic differentiation operates on a Directed Acyclic Graph (DAG) of mathematical operations. It occurs in two phases:\n1. Forward Pass: Computes the output of each operation from inputs to loss $L$, caching intermediate tensors required for derivative calculation.\n2. Backward Pass: Traverses the graph in topological reverse order, applying the chain rule to accumulate adjoints (gradients $\\bar{v} = \\frac{\\partial L}{\\partial v}$) from the loss backward to the leaf parameters."),
        ("Why is reverse-mode autodiff much faster than forward-mode autodiff for deep learning?",
         "Consider a function with $N$ inputs (parameters) and $M$ outputs. Forward-mode autodiff propagates directional derivatives alongside the forward pass; finding the full gradient vector requires $N$ forward passes—one for each parameter. In a model with 1 billion parameters ($N = 10^9$) and 1 scalar loss ($M = 1$), forward-mode would take 1 billion passes!\n\nReverse-mode autodiff, conversely, requires only $M$ backward passes. Since the loss is a single scalar ($M = 1$), the exact gradient with respect to all 1 billion parameters is computed in a single reverse sweep, matching the asymptotic complexity of the forward pass.")
    ],

    # Dialogue 31: Vanishing & Exploding Gradients
    [
        ("What causes vanishing and exploding gradients in deep neural networks?",
         "In deep feed-forward networks, computing the gradient of the loss with respect to early layer weights involves multiplying Jacobian matrices across all intervening layers: $\\prod_{l=1}^L J_l$. If the singular values of these Jacobians are systematically less than 1 (e.g., when sigmoid or tanh activations saturate near 0), gradients shrink exponentially with depth ($0.9^{50} \\approx 0.005$), vanishing before reaching early layers.\n\nConversely, if singular values exceed 1 (or weights are initialized too large), gradients grow exponentially, causing numerical overflow (NaNs) and weight divergence."),
        ("How do residual connections mathematically solve the vanishing gradient problem?",
         "In a residual block, the activation at layer $l+1$ is defined as $x_{l+1} = x_l + F(x_l, W_l)$. When taking the derivative of the loss with respect to $x_l$:\n$$\\frac{\\partial L}{\\partial x_l} = \\frac{\\partial L}{\\partial x_{l+1}} \\frac{\\partial x_{l+1}}{\\partial x_l} = \\frac{\\partial L}{\\partial x_{l+1}} \\left(I + \\frac{\\partial F(x_l, W_l)}{\\partial x_l}\\right)$$\nBecause of the identity matrix $I$, the term $\\frac{\\partial L}{\\partial x_{l+1}}$ propagates directly backward without being attenuated by $\\frac{\\partial F}{\\partial x_l}$. Even if the sub-layer gradient $\\frac{\\partial F}{\\partial x_l}$ vanishes completely, the identity gradient highway guarantees that gradients reach early layers unhindered.")
    ],

    # Dialogue 32: Optimizers - SGD vs Adam vs AdamW
    [
        ("What is the fundamental difference between Adam and AdamW?",
         "The fundamental difference lies in how weight decay is implemented relative to adaptive gradient updates.\n\nIn standard Adam with L2 regularization, the weight decay gradient term $\\lambda \\theta$ is added directly to the loss gradient: $g_t = \\nabla L(\\theta_t) + \\lambda \\theta_t$. This combined vector is then tracked by both the first moment ($m_t$, exponential moving average of gradients) and the second moment ($v_t$, exponential moving average of squared gradients). As a result, parameters with large historical gradients receive smaller relative weight decay, while parameters with small gradients receive disproportionately large weight decay.\n\nAdamW (decoupled weight decay) isolates weight decay from the gradient moments completely, applying decay directly to the weights after the adaptive step:\n$$\\theta_{t+1} = \\theta_t - \\eta_t \\left( \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon} + \\lambda \\theta_t \\right)$$\nThis restores the true intended regularization behavior and dramatically improves generalization in transformer training."),
        ("Why are bias corrections $\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}$ necessary in Adam?",
         "Because $m_0$ and $v_0$ are initialized to zero vectors. In early iterations ($t=1, 2$), the moving averages are heavily biased toward zero because $\\beta_1$ is typically 0.9 and $\\beta_2$ is 0.999. Dividing by $(1 - \\beta^t)$ mathematically compensates for this initialization discrepancy, ensuring unbiased gradient estimators from the very first step.")
    ],

    # Dialogue 33: Learning Rate Scheduling & Warmup
    [
        ("Why do transformer models require a learning rate warmup phase at the start of training?",
         "At the beginning of training, weights are randomly initialized, and internal statistics (such as LayerNorm running estimates and attention patterns) have not stabilized. If a high learning rate is used immediately, early gradients can be noisy and disproportionately large, causing parameters to take massive steps into chaotic regions of the loss landscape from which recovery is difficult or impossible.\n\nA warmup schedule starts the learning rate at 0 and linearly increases it over the first several hundred or thousand steps. This allows the model to find stable gradient directions before the maximum learning rate is applied."),
        ("Why is Cosine Annealing decay preferred over standard step decay?",
         "Step decay abruptly drops the learning rate at predefined epochs, which can cause sudden shifts and requires manual tuning of step milestones. Cosine Annealing smoothly reduces the learning rate following a cosine curve down to a small minimum $\\eta_{\\min}$. This gentle deceleration allows the optimizer to continuously explore the loss surface before settling smoothly into wide, flat, generalized minima.")
    ],

    # Dialogue 34: Tokenization & Byte-Pair Encoding
    [
        ("How does Byte-Pair Encoding (BPE) tokenize raw text into subwords?",
         "BPE is a data-driven compression algorithm adapted for tokenization:\n1. Start with a base vocabulary containing all single characters (or raw bytes 0-255 in byte-level BPE).\n2. Count the frequencies of all adjacent token pairs across the training corpus.\n3. Identify the most frequent adjacent pair (e.g., 't' followed by 'h') and merge it into a single new token ('th').\n4. Add this merged token to the vocabulary.\n5. Repeat steps 2-4 iteratively until reaching the target vocabulary size (e.g., 32,000 or 128,000 tokens).\n\nFrequent words become single tokens (e.g., 'learning'), while rare or misspelled words are broken down into subword pieces (e.g., 'neuro' + 'morphic'), avoiding out-of-vocabulary errors."),
        ("What are the trade-offs of choosing a smaller vs larger vocabulary size?",
         "A small vocabulary (e.g., 8,000 tokens) keeps the embedding matrix compact, saving VRAM, but splits words into many pieces, increasing sequence lengths and quadratic attention cost. A large vocabulary (e.g., 128,000 tokens) achieves high compression ratio (fewer tokens per sentence, faster inference), but significantly increases embedding table memory and parameter count.")
    ],

    # Dialogue 35: KV Caching in Autoregressive Inference
    [
        ("Why does autoregressive text generation require a KV Cache, and what problem does it solve?",
         "During autoregressive generation, the model predicts one token at a time: token $T+1$ depends on tokens $1 \\dots T$. Without caching, at step $T+1$, the model would have to recompute Key and Value projections for all preceding $T$ tokens from scratch. Across a generation of length $N$, the computational complexity would be $O(N^2)$ in matrix multiplications.\n\nWith a KV Cache, the Key and Value tensor projections for prior tokens are stored in memory. At step $T+1$, the model computes $Q, K, V$ for the single new token only, appends the new $K$ and $V$ to the cache, and computes attention against the cached history. This reduces the per-step computation to $O(N)$, making generation memory-bandwidth bound rather than compute bound."),
        ("How does Grouped-Query Attention (GQA) reduce the memory footprint of the KV Cache?",
         "In standard Multi-Head Attention, each query head has a corresponding key and value head. For large models with long context windows, storing separate KV caches for all 32 or 64 heads consumes gigabytes of VRAM per request. Grouped-Query Attention (GQA) partitions query heads into groups (e.g., 8 query heads per group) that share a single key and value head. This slashes KV cache memory by $8\\times$ with virtually no degradation in model accuracy.")
    ],

    # Dialogue 36: Decoding & Sampling Strategies
    [
        ("How do Temperature, Top-K, and Top-P (Nucleus) sampling interact during generation?",
         "They work together in a three-stage pipeline to shape the next-token probability distribution:\n\n1. Temperature ($T$): Divides logits by $T$ before softmax ($z_i / T$). $T < 1.0$ sharpens probabilities toward the top candidate (deterministic, conservative); $T > 1.0$ flattens the distribution (creative, diverse).\n2. Top-K: Retains only the $K$ highest-probability tokens and sets all others to $-\\infty$, preventing bizarre tail tokens.\n3. Top-P (Nucleus): Sorts the remaining candidates and accumulates probabilities until their sum reaches threshold $P$ (e.g., 0.90). If the model is confident, the nucleus contains only 1 or 2 tokens; if uncertain, the nucleus dynamically widens.\n\nFinally, the model samples a token randomly from this filtered, normalized distribution."),
        ("Why does pure greedy decoding often lead to repetitive loops in open-ended generation?",
         "Greedy decoding always picks $\\arg\\max$. Natural human language does not consistently use the most probable statistical token at every step; doing so drives the model into localized attractor states—repetitive loops where the highest-probability token after a phrase is the exact same phrase. Nucleus sampling injects the natural linguistic entropy required for coherent, varied prose.")
    ],

    # Dialogue 37: Mixture of Experts (MoE) Architecture
    [
        ("What is a Mixture of Experts (MoE) transformer, and how does sparse routing work?",
         "In a dense transformer, every token passes through every feed-forward network (FFN) parameter in every layer. In an MoE transformer, the single FFN block is replaced by $E$ independent expert networks (e.g., 8 or 64 experts), alongside a lightweight gating/routing router.\n\nFor each token $x$, the router computes gating scores $H(x) = \\text{Softmax}(\\text{TopK}(x W_g, k))$. Only the top-$k$ experts (typically $k=2$) are evaluated, and their outputs are weighted by the router probabilities. Because unselected experts perform zero computation, a model can possess 50 billion total parameters while activating only 10 billion parameters per token, drastically cutting compute cost."),
        ("What is the purpose of an auxiliary load-balancing loss in MoE models?",
         "Without a load-balancing loss, MoE training suffers from 'router collapse': the router quickly develops a preference for 1 or 2 experts early in training. Those favored experts receive all gradients and get better, while the remaining experts starve and never learn. An auxiliary load balancing loss penalizes the router when token assignment across experts is non-uniform, ensuring all experts are utilized equally across training batches.")
    ]
]
