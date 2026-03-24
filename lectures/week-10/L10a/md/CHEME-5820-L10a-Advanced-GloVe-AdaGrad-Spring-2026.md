# Advanced: GloVe Gradient Derivation and AdaGrad Optimization

This notebook derives the gradients of the GloVe weighted least-squares objective from first principles and examines why AdaGrad, the optimizer used in the original GloVe paper, is well-suited to the structure of this problem. We also compare AdaGrad to Adam and discuss how GloVe's iteration over nonzero co-occurrence pairs differs from the sliding-window training used in CBOW and Skip-Gram.

> __Learning Objectives:__
>
> By the end of this notebook, you should be able to:
>
> * __Derive GloVe gradients:__ Apply the chain rule to the weighted least-squares objective to obtain gradients for word vectors, context vectors, and bias terms. Explain why each gradient is proportional to the residual and the weighting function.
> * __Explain AdaGrad for GloVe:__ Describe how AdaGrad accumulates squared gradients to adapt per-parameter learning rates, and explain why this is a natural fit for GloVe's sparse co-occurrence structure where entry magnitudes vary over several orders of magnitude.
> * __Compare AdaGrad and Adam:__ Identify AdaGrad's monotonic decay limitation and describe how Adam's exponential moving averages address it. Explain when each optimizer is preferred for GloVe training.

Let's get started!

___

## Gradient Derivation for GloVe
We derive the gradients of the GloVe objective with respect to all four parameter groups. Recall the objective from the lecture:
$$J = \sum_{i,j:\, X_{ij}>0} f(X_{ij})\left(\mathbf{w}_{i}^{\top}\tilde{\mathbf{w}}_{j} + b_{i} + \tilde{b}_{j} - \log X_{ij}\right)^{2}$$
where $\mathbf{w}_{i}\in\mathbb{R}^{d}$ is the word vector, $\tilde{\mathbf{w}}_{j}\in\mathbb{R}^{d}$ is the context vector, $b_{i},\tilde{b}_{j}\in\mathbb{R}$ are scalar biases, and $f(X_{ij}) = \min(1, (X_{ij}/x_{\max})^{\alpha})$ is the weighting function. Define the residual for each pair:
$$e_{ij} = \mathbf{w}_{i}^{\top}\tilde{\mathbf{w}}_{j} + b_{i} + \tilde{b}_{j} - \log X_{ij}$$
so that $J = \sum_{i,j} f(X_{ij})\,e_{ij}^{2}$. We now differentiate $J$ with respect to each parameter group.

### Gradient with respect to $\mathbf{w}_{i}$
The word vector $\mathbf{w}_{i}$ appears in the residual through the dot product $\mathbf{w}_{i}^{\top}\tilde{\mathbf{w}}_{j}$. Applying the chain rule to a single term in the sum:
$$\begin{align*}
\frac{\partial}{\partial \mathbf{w}_{i}}\left[f(X_{ij})\,e_{ij}^{2}\right]
  &= f(X_{ij})\cdot 2\,e_{ij}\cdot\frac{\partial e_{ij}}{\partial \mathbf{w}_{i}}
  && \text{chain rule on }e_{ij}^{2}\\
  &= f(X_{ij})\cdot 2\,e_{ij}\cdot\frac{\partial}{\partial \mathbf{w}_{i}}\left(\mathbf{w}_{i}^{\top}\tilde{\mathbf{w}}_{j} + b_{i} + \tilde{b}_{j} - \log X_{ij}\right)
  && \text{expand }e_{ij}\\
  &= f(X_{ij})\cdot 2\,e_{ij}\cdot\tilde{\mathbf{w}}_{j}
  && \frac{\partial}{\partial \mathbf{w}_{i}}\mathbf{w}_{i}^{\top}\tilde{\mathbf{w}}_{j} = \tilde{\mathbf{w}}_{j}
\end{align*}$$
The full gradient sums over all context words $j$ that co-occur with word $i$, but during training we process one pair $(i,j)$ at a time, so the per-pair gradient is:
$$\frac{\partial J}{\partial \mathbf{w}_{i}} = 2\,f(X_{ij})\,e_{ij}\,\tilde{\mathbf{w}}_{j} \in \mathbb{R}^{d}$$

### Gradient with respect to $\tilde{\mathbf{w}}_{j}$
By the same logic, $\tilde{\mathbf{w}}_{j}$ enters the residual through the dot product $\mathbf{w}_{i}^{\top}\tilde{\mathbf{w}}_{j}$, so:
$$\begin{align*}
\frac{\partial}{\partial \tilde{\mathbf{w}}_{j}}\left[f(X_{ij})\,e_{ij}^{2}\right]
  &= f(X_{ij})\cdot 2\,e_{ij}\cdot\frac{\partial}{\partial \tilde{\mathbf{w}}_{j}}\left(\mathbf{w}_{i}^{\top}\tilde{\mathbf{w}}_{j}\right)
  = 2\,f(X_{ij})\,e_{ij}\,\mathbf{w}_{i}
\end{align*}$$
The per-pair gradient is:
$$\frac{\partial J}{\partial \tilde{\mathbf{w}}_{j}} = 2\,f(X_{ij})\,e_{ij}\,\mathbf{w}_{i} \in \mathbb{R}^{d}$$
Note the symmetry: the gradient for the word vector depends on the context vector and vice versa. This symmetry reflects the fact that the co-occurrence matrix can be made symmetric ($X_{ij} = X_{ji}$), so the roles of word and context are interchangeable.

### Gradients with respect to bias terms
The biases $b_{i}$ and $\tilde{b}_{j}$ enter the residual linearly, so their gradients are the simplest:
$$\begin{align*}
\frac{\partial J}{\partial b_{i}} &= 2\,f(X_{ij})\,e_{ij}\cdot\frac{\partial e_{ij}}{\partial b_{i}} = 2\,f(X_{ij})\,e_{ij} \in \mathbb{R}\\
\frac{\partial J}{\partial \tilde{b}_{j}} &= 2\,f(X_{ij})\,e_{ij}\cdot\frac{\partial e_{ij}}{\partial \tilde{b}_{j}} = 2\,f(X_{ij})\,e_{ij} \in \mathbb{R}
\end{align*}$$
Both bias gradients equal $2\,f(X_{ij})\,e_{ij}$ because $\partial e_{ij}/\partial b_{i} = \partial e_{ij}/\partial \tilde{b}_{j} = 1$.

### Summary of GloVe Gradients
All four gradients share the same structure: the weighting function times the residual times a direction vector.

> __Definition (GloVe per-pair gradients):__
>
> For each pair $(i,j)$ with $X_{ij} > 0$, let $e_{ij} = \mathbf{w}_{i}^{\top}\tilde{\mathbf{w}}_{j} + b_{i} + \tilde{b}_{j} - \log X_{ij}$. The gradients are:
> $$\begin{align*}
> \nabla_{\mathbf{w}_{i}} J &= 2\,f(X_{ij})\,e_{ij}\,\tilde{\mathbf{w}}_{j} \in \mathbb{R}^{d} \\
> \nabla_{\tilde{\mathbf{w}}_{j}} J &= 2\,f(X_{ij})\,e_{ij}\,\mathbf{w}_{i} \in \mathbb{R}^{d} \\
> \nabla_{b_{i}} J &= 2\,f(X_{ij})\,e_{ij} \in \mathbb{R} \\
> \nabla_{\tilde{b}_{j}} J &= 2\,f(X_{ij})\,e_{ij} \in \mathbb{R}
> \end{align*}$$
> Each gradient is proportional to the residual $e_{ij}$ (how far the model is from the target $\log X_{ij}$) and scaled by $f(X_{ij})$ (how much this pair should influence training). When $e_{ij} = 0$, the model perfectly predicts $\log X_{ij}$ and all gradients vanish.

Compared to CBOW and Skip-Gram, the GloVe gradients are simpler: there is no softmax, no cross-entropy, and no Jacobian to unwind. The objective is a weighted regression, so the gradients reduce to scaled residuals.

___

## AdaGrad for GloVe
The original GloVe paper uses AdaGrad to optimize the weighted least-squares objective. AdaGrad maintains a per-parameter accumulator of squared gradients and uses it to scale the learning rate, giving each parameter its own effective step size. This section explains why AdaGrad is a natural fit for GloVe and describes the update rules.

__Why Adaptive Learning Rates?__ GloVe iterates over all nonzero entries of $\mathbf{X}$, and each gradient is proportional to $f(X_{ij})\,e_{ij}$. Consider word $i$ = "the", which participates in hundreds of nonzero pairs, versus word $i$ = "aardvark", which may appear in only a handful. A single global learning rate $\eta$ faces a dilemma:

* __Too large__: parameters for frequent words like "the" accumulate large net updates per epoch, causing their vectors to oscillate around the optimum rather than converging.
* __Too small__: parameters for rare words like "aardvark" receive too few updates per epoch, and each update is too small to meaningfully move the vector away from its initialization.

This is not just a matter of word frequency — even among frequent words, individual embedding components may receive gradients that vary by orders of magnitude depending on which context vectors they interact with.

AdaGrad resolves this by tracking how much gradient each parameter has received and automatically reducing the learning rate for parameters with large cumulative gradients (frequent pairs) while preserving it for parameters with small cumulative gradients (rare pairs).

### AdaGrad Update Rules
AdaGrad maintains an accumulator for each parameter that tracks the sum of squared gradients seen so far.

> __Definition (AdaGrad for GloVe):__
>
> For each parameter $\theta$ (which can be $\mathbf{w}_{i}$, $\tilde{\mathbf{w}}_{j}$, $b_{i}$, or $\tilde{b}_{j}$), initialize an accumulator $\mathbf{G}_{\theta} = \mathbf{0}$ with the same shape as $\theta$. Given learning rate $\eta > 0$ and stability constant $\epsilon > 0$ (typically $10^{-8}$), the update for each co-occurrence pair $(i,j)$ proceeds in three steps:
>
> 1. Compute the gradient $\mathbf{g}_{\theta} = \nabla_{\theta} J$ for the current pair.
> 2. Accumulate squared gradients: $\mathbf{G}_{\theta} \leftarrow \mathbf{G}_{\theta} + \mathbf{g}_{\theta} \odot \mathbf{g}_{\theta}$
> 3. Update the parameter: $\theta \leftarrow \theta - \dfrac{\eta}{\sqrt{\mathbf{G}_{\theta}} + \epsilon} \odot \mathbf{g}_{\theta}$
>
> where $\odot$ denotes element-wise multiplication and all operations (square root, division, addition of $\epsilon$) are element-wise.

For the word vector $\mathbf{w}_{i}\in\mathbb{R}^{d}$, each of the $d$ components has its own accumulator entry. If component $k$ of $\mathbf{w}_{i}$ frequently receives large gradients, its effective learning rate $\eta / (\sqrt{G_{\mathbf{w}_{i},k}} + \epsilon)$ shrinks. Components with small cumulative gradients retain rates close to $\eta$.

### Concrete AdaGrad Updates for GloVe
Expanding the AdaGrad rule for each of the four GloVe parameter groups gives the complete update equations.

> __Definition (GloVe-AdaGrad update equations):__
>
> For each pair $(i,j)$ with $X_{ij} > 0$, compute the residual $e_{ij} = \mathbf{w}_{i}^{\top}\tilde{\mathbf{w}}_{j} + b_{i} + \tilde{b}_{j} - \log X_{ij}$ and the common scalar $s_{ij} = 2\,f(X_{ij})\,e_{ij}$. Then:
>
> $$\begin{align*}
> \mathbf{g}_{\mathbf{w}_{i}} &= s_{ij}\,\tilde{\mathbf{w}}_{j}, &\quad \mathbf{G}_{\mathbf{w}_{i}} &\leftarrow \mathbf{G}_{\mathbf{w}_{i}} + \mathbf{g}_{\mathbf{w}_{i}} \odot \mathbf{g}_{\mathbf{w}_{i}}, &\quad \mathbf{w}_{i} &\leftarrow \mathbf{w}_{i} - \frac{\eta}{\sqrt{\mathbf{G}_{\mathbf{w}_{i}}} + \epsilon} \odot \mathbf{g}_{\mathbf{w}_{i}} \\[6pt]
> \mathbf{g}_{\tilde{\mathbf{w}}_{j}} &= s_{ij}\,\mathbf{w}_{i}, &\quad \mathbf{G}_{\tilde{\mathbf{w}}_{j}} &\leftarrow \mathbf{G}_{\tilde{\mathbf{w}}_{j}} + \mathbf{g}_{\tilde{\mathbf{w}}_{j}} \odot \mathbf{g}_{\tilde{\mathbf{w}}_{j}}, &\quad \tilde{\mathbf{w}}_{j} &\leftarrow \tilde{\mathbf{w}}_{j} - \frac{\eta}{\sqrt{\mathbf{G}_{\tilde{\mathbf{w}}_{j}}} + \epsilon} \odot \mathbf{g}_{\tilde{\mathbf{w}}_{j}} \\[6pt]
> g_{b_{i}} &= s_{ij}, &\quad G_{b_{i}} &\leftarrow G_{b_{i}} + g_{b_{i}}^{2}, &\quad b_{i} &\leftarrow b_{i} - \frac{\eta}{\sqrt{G_{b_{i}}} + \epsilon}\,g_{b_{i}} \\[6pt]
> g_{\tilde{b}_{j}} &= s_{ij}, &\quad G_{\tilde{b}_{j}} &\leftarrow G_{\tilde{b}_{j}} + g_{\tilde{b}_{j}}^{2}, &\quad \tilde{b}_{j} &\leftarrow \tilde{b}_{j} - \frac{\eta}{\sqrt{G_{\tilde{b}_{j}}} + \epsilon}\,g_{\tilde{b}_{j}}
> \end{align*}$$
>
> Note that $\mathbf{w}_{i}$ must be read before it is updated, since the gradient $\mathbf{g}_{\tilde{\mathbf{w}}_{j}} = s_{ij}\,\mathbf{w}_{i}$ depends on the current value of $\mathbf{w}_{i}$.

The memory overhead is one accumulator per parameter: $\mathbf{G}_{\mathbf{w}_{i}}\in\mathbb{R}^{d}$ for each of $N_{\mathcal{V}}$ word vectors, $\mathbf{G}_{\tilde{\mathbf{w}}_{j}}\in\mathbb{R}^{d}$ for each context vector, and one scalar each for $G_{b_{i}}$ and $G_{\tilde{b}_{j}}$. The total memory is $1\times$ the parameter count.

___

## Why AdaGrad Fits GloVe
AdaGrad was designed for sparse problems where different parameters receive gradients at different frequencies. GloVe's training loop has exactly this structure.

> __Sparse iteration over nonzero pairs__
>
> GloVe iterates over all pairs $(i,j)$ with $X_{ij} > 0$. In a typical corpus, the co-occurrence matrix is sparse: most word pairs never co-occur within the context window. Frequent words like "the" participate in many nonzero pairs and receive gradient updates at nearly every step, while rare words participate in few pairs and receive updates infrequently. AdaGrad naturally handles this asymmetry:
>
> * **Frequent words** accumulate large $\mathbf{G}_{\mathbf{w}_{i}}$ values, shrinking their effective learning rate and preventing oscillation.
> * **Rare words** accumulate small $\mathbf{G}_{\mathbf{w}_{i}}$ values, preserving a larger effective learning rate so that each update has a meaningful impact.

This property is shared with Skip-Gram (see the L9c Advanced notebook), but GloVe's iteration pattern is different.

### GloVe vs. Skip-Gram Iteration
In Skip-Gram, training iterates over the corpus sequentially: for each word in each sentence, the model generates (target, context) pairs from a sliding window. The same word pair may appear multiple times across different sentences, but the training order follows the corpus structure.

GloVe decouples training from the corpus order. The co-occurrence matrix is precomputed, and training iterates over its nonzero entries in shuffled order. Each entry $(i,j)$ appears exactly once per epoch regardless of how many times the pair occurred in the corpus — the count information is encoded in $X_{ij}$ and $f(X_{ij})$ rather than in repeated sampling. This means:

* The number of training steps per epoch equals the number of nonzero entries in $\mathbf{X}$, not the corpus length.
* The weighting function $f(X_{ij})$ controls the influence of each pair, replacing the implicit frequency weighting of repeated sampling in Skip-Gram.
* Shuffling the nonzero entries is straightforward since they are stored as a list of $(i, j, X_{ij})$ triples.

___

## Comparison with Adam
AdaGrad has a well-known limitation: the accumulator $\mathbf{G}_{\theta}$ grows monotonically, so the effective learning rate $\eta / (\sqrt{\mathbf{G}_{\theta}} + \epsilon)$ decays to zero over time. For long training runs, this can prevent the optimizer from making further progress. Adam addresses this by replacing the cumulative sum with exponential moving averages.

> __Definition (Adam for GloVe):__
>
> Given decay rates $\beta_1 \in [0,1)$ (typically $0.9$) and $\beta_2 \in [0,1)$ (typically $0.999$), initialize first moment $\mathbf{m}_{\theta} = \mathbf{0}$ and second moment $\mathbf{v}_{\theta} = \mathbf{0}$. At step $t$, after computing gradient $\mathbf{g}_{\theta}$:
>
> $$\begin{align*}
> \mathbf{m}_{\theta} &\leftarrow \beta_1\,\mathbf{m}_{\theta} + (1-\beta_1)\,\mathbf{g}_{\theta} \\
> \mathbf{v}_{\theta} &\leftarrow \beta_2\,\mathbf{v}_{\theta} + (1-\beta_2)\,\mathbf{g}_{\theta} \odot \mathbf{g}_{\theta} \\
> \hat{\mathbf{m}}_{\theta} &= \mathbf{m}_{\theta}\,/\,(1 - \beta_1^{t}) \\
> \hat{\mathbf{v}}_{\theta} &= \mathbf{v}_{\theta}\,/\,(1 - \beta_2^{t}) \\
> \theta &\leftarrow \theta - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_{\theta}} + \epsilon} \odot \hat{\mathbf{m}}_{\theta}
> \end{align*}$$
>
> The exponential decay means old gradients are gradually forgotten, so the effective learning rate can increase or decrease based on recent gradient history. The memory overhead is $2\times$ the parameter count (storing both $\mathbf{m}$ and $\mathbf{v}$), compared to $1\times$ for AdaGrad.

The table below compares the two optimizers in the context of GloVe training.

| Property | AdaGrad | Adam |
|----------|---------|------|
| Accumulator | Cumulative sum: $\mathbf{G} \leftarrow \mathbf{G} + \mathbf{g}^2$ | Exponential average: $\mathbf{v} \leftarrow \beta_2\mathbf{v} + (1-\beta_2)\mathbf{g}^2$ |
| Learning rate decay | Monotonic: $\eta/\sqrt{G}$ only decreases | Bounded: can increase if recent gradients are small |
| Momentum | None | First moment $\mathbf{m}$ smooths gradient direction |
| Memory per parameter | $1\times$ (accumulator only) | $2\times$ (first and second moments) |
| Hyperparameters | $\eta$ only | $\eta$, $\beta_1$, $\beta_2$ |
| GloVe paper default | $\eta = 0.05$ | Not used in original paper |

For GloVe on moderately sized corpora (up to a few billion tokens), AdaGrad converges within 50–100 epochs and the monotonic decay is not a practical issue. For very large corpora or when training for many epochs, Adam's bounded decay can help maintain learning progress.

___

## Summary

This notebook derived the GloVe gradients from the weighted least-squares objective and examined how AdaGrad adapts learning rates to the sparse structure of co-occurrence-based training.

> __Key Takeaways:__
>
> * __Gradients reduce to scaled residuals:__ Each GloVe gradient equals the weighting function $f(X_{ij})$ times the residual $e_{ij}$ times a direction vector (the partner embedding for vectors, or 1 for biases). There is no softmax or cross-entropy to differentiate, making GloVe gradients simpler than those of CBOW or Skip-Gram.
> * __AdaGrad matches GloVe's sparsity:__ GloVe iterates over nonzero co-occurrence entries, so frequent words receive many gradient updates and rare words receive few. AdaGrad's per-parameter accumulator automatically scales learning rates to account for this imbalance, shrinking rates for frequent words and preserving them for rare words.
> * __Adam prevents indefinite decay:__ AdaGrad's monotonically growing accumulator can prematurely stop learning on long training runs. Adam replaces this with exponential moving averages that allow the effective learning rate to recover, at the cost of additional memory and hyperparameters.

___

## References
* [Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 1532-1543).](https://nlp.stanford.edu/pubs/glove.pdf)
* [Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. JMLR, 12, 2121-2159.](https://jmlr.org/papers/v12/duchi11a.html)
* [Kingma, D.P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR 2015.](https://arxiv.org/abs/1412.6980)

___
