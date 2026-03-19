# Advanced: Gradient Descent Algorithms for Skip-Gram

This notebook derives the gradients for Skip-Gram with Negative Sampling and examines adaptive gradient methods (AdaGrad and Adam) that address the sparse, high-dimensional optimization problem arising in word embedding training.

> __Learning Objectives:__
>
> By the end of this notebook, you should be able to:
>
> * __Derive Skip-Gram gradients:__ Compute the gradients of the Negative Sampling loss with respect to the input and output weight matrices, and explain why only a small subset of parameters receives non-zero gradients per training example.
> * __Compare AdaGrad and Adam:__ Describe how AdaGrad accumulates squared gradients to adapt per-parameter learning rates, and how Adam improves on this with exponential moving averages that prevent indefinite learning rate decay.
> * __Connect optimizers to Skip-Gram structure:__ Explain why adaptive methods are well-suited to Skip-Gram's sparse gradient structure, where frequent words need smaller learning rates and rare words need larger ones.

Let's get started!

___

## Gradient Computation for Skip-Gram
We first derive the gradients needed for optimization. For a single training example with target word $w_t$ and context position $c$, the Negative Sampling loss is:
$$\mathcal{L}_c = -\log \sigma((\mathbf{w}_c^{(2)})^{\top} \mathbf{h}) - \sum_{i=1}^{k} \log \sigma(-(\mathbf{w}_{n_i}^{(2)})^{\top} \mathbf{h})$$
where $\mathbf{h} = \mathbf{W}_1 \mathbf{v}_{w_t} \in \mathbb{R}^{d_h}$ is the target word embedding (column of $\mathbf{W}_1$), $\mathbf{w}_c^{(2)}$ is the context word's output embedding (row of $\mathbf{W}_2$), $\mathbf{w}_{n_i}^{(2)}$ are negative sample embeddings, and $\sigma(x) = 1/(1+e^{-x})$. The total loss sums over all $|\mathcal{C}| = 2m$ context positions: $\mathcal{L} = \sum_{c \in \mathcal{C}} \mathcal{L}_c$.

> **Gradients with respect to output embeddings.** For the positive context word and each negative sample:
> $$\frac{\partial \mathcal{L}_c}{\partial \mathbf{w}_c^{(2)}} = (\sigma((\mathbf{w}_c^{(2)})^{\top} \mathbf{h}) - 1) \mathbf{h} \in \mathbb{R}^{d_h}$$
> $$\frac{\partial \mathcal{L}_c}{\partial \mathbf{w}_{n_i}^{(2)}} = \sigma((\mathbf{w}_{n_i}^{(2)})^{\top} \mathbf{h}) \mathbf{h} \in \mathbb{R}^{d_h} \quad \text{for } i = 1, \ldots, k$$
> Only the positive context word and $k$ negative samples receive non-zero gradients. The remaining $N_{\mathcal{V}} - k - 1$ rows of $\mathbf{W}_2$ have zero gradients.

The gradient with respect to the target embedding collects contributions from both the positive and negative terms.

> **Gradient with respect to the target embedding.** Applying the chain rule:
> $$\frac{\partial \mathcal{L}_c}{\partial \mathbf{h}} = (\sigma((\mathbf{w}_c^{(2)})^{\top} \mathbf{h}) - 1) \mathbf{w}_c^{(2)} + \sum_{i=1}^{k} \sigma((\mathbf{w}_{n_i}^{(2)})^{\top} \mathbf{h}) \mathbf{w}_{n_i}^{(2)} \in \mathbb{R}^{d_h}$$
> Since $\mathbf{h} = \mathbf{W}_1 \mathbf{v}_{w_t}$ and $\mathbf{v}_{w_t}$ is one-hot, the gradient $\frac{\partial \mathcal{L}_c}{\partial \mathbf{W}_1} = \frac{\partial \mathcal{L}_c}{\partial \mathbf{h}} \mathbf{v}_{w_t}^{\top}$ updates only the column of $\mathbf{W}_1$ corresponding to target word $w_t$.

This sparsity is the key structural property of Skip-Gram optimization: each training example touches one column of $\mathbf{W}_1$ and $k+1$ rows of $\mathbf{W}_2$, leaving everything else unchanged. Frequent words accumulate many gradient updates while rare words receive few, motivating adaptive learning rate methods.

___

## Adaptive Gradient Methods
Standard gradient descent uses a single learning rate $\alpha$ for all parameters: $\theta_{k+1} = \theta_k - \alpha \nabla_{\theta} \mathcal{L}(\theta_k)$. For Skip-Gram, this is problematic because frequent words need smaller learning rates (to avoid oscillation) while rare words need larger ones (to learn from limited updates). AdaGrad and Adam address this by maintaining per-parameter learning rates.

> **AdaGrad.** Given initial parameters $\theta_0$, learning rate $\alpha > 0$, and $\epsilon > 0$ (typically $10^{-8}$), initialize $\mathbf{G}_0 = \mathbf{0}$. At each iteration $k$, compute the gradient $\mathbf{g}_k = \nabla_{\theta} \mathcal{L}(\theta_k)$, accumulate squared gradients $\mathbf{G}_{k+1} = \mathbf{G}_k + \mathbf{g}_k \odot \mathbf{g}_k$, and update:
> $$\theta_{k+1} = \theta_k - \frac{\alpha}{\sqrt{\mathbf{G}_{k+1}} + \epsilon} \odot \mathbf{g}_k$$
> where all operations are element-wise. The effective learning rate for parameter $i$ is $\alpha/(\sqrt{G_{k+1,i}} + \epsilon)$. Parameters with large cumulative gradients (frequent words) automatically receive smaller learning rates, while parameters with small cumulative gradients (rare words) retain rates close to $\alpha$.

AdaGrad is a natural fit for Skip-Gram's sparse gradients, but it has a critical limitation: $\mathbf{G}_k$ grows monotonically, so the effective learning rate decays to zero as $k \to \infty$. For long training runs over large corpora, this aggressive decay can stop learning before the embeddings converge.

> **Adam.** Adam replaces AdaGrad's cumulative sum with exponential moving averages. Given decay rates $\beta_1 \in [0,1)$ (typically $0.9$) and $\beta_2 \in [0,1)$ (typically $0.999$), maintain first moment $\mathbf{m}_k$ and second moment $\mathbf{v}_k$:
> $$\mathbf{m}_k = \beta_1 \mathbf{m}_{k-1} + (1 - \beta_1) \mathbf{g}_k, \qquad \mathbf{v}_k = \beta_2 \mathbf{v}_{k-1} + (1 - \beta_2) \mathbf{g}_k \odot \mathbf{g}_k$$
> Apply bias correction $\hat{\mathbf{m}}_k = \mathbf{m}_k/(1 - \beta_1^k)$ and $\hat{\mathbf{v}}_k = \mathbf{v}_k/(1 - \beta_2^k)$, then update:
> $$\theta_k = \theta_{k-1} - \frac{\alpha}{\sqrt{\hat{\mathbf{v}}_k} + \epsilon} \odot \hat{\mathbf{m}}_k$$
> The first moment $\mathbf{m}_k$ smooths noisy gradients (momentum), while the second moment $\mathbf{v}_k$ provides per-parameter scaling (adaptive rates). With $\beta_2 = 0.999$, gradients from $\sim 1000$ iterations ago contribute negligibly ($0.999^{1000} \approx 0.37$), preventing the indefinite decay that limits AdaGrad.

The effective learning rate $\alpha/(\sqrt{\hat{v}_{k,i}} + \epsilon)$ can increase or decrease based on recent gradient history, allowing Adam to adapt to changing loss landscapes during training. The memory overhead is $2 \times \text{sizeof}(\theta)$ to store $\mathbf{m}$ and $\mathbf{v}$.

___

## Application to Mini-Batch Training
In practice, Skip-Gram processes mini-batches of $B$ training examples. For each mini-batch, the gradients are averaged and then passed to the optimizer.

> **Mini-batch gradient accumulation.** For a mini-batch of $B$ training examples, accumulate:
> $$\mathbf{g}_{\mathbf{W}_1,k} = \frac{1}{B} \sum_{b=1}^{B} \sum_{c \in \mathcal{C}_b} \frac{\partial \mathcal{L}_{b,c}}{\partial \mathbf{W}_1}, \qquad \mathbf{g}_{\mathbf{W}_2,k} = \frac{1}{B} \sum_{b=1}^{B} \sum_{c \in \mathcal{C}_b} \left( \frac{\partial \mathcal{L}_{b,c}}{\partial \mathbf{w}_{c}^{(2)}} + \sum_{i=1}^{k} \frac{\partial \mathcal{L}_{b,c}}{\partial \mathbf{w}_{n_i}^{(2)}} \right)$$
> Adam updates are applied separately to $\mathbf{W}_1$ and $\mathbf{W}_2$ using these accumulated gradients. In each mini-batch, only embeddings for observed words (targets and their contexts/negative samples) receive non-zero gradients. Adam's moment estimates for unobserved words remain unchanged, automatically handling sparsity without special logic.

Training typically continues for a fixed number of epochs rather than until convergence, as word embeddings continue improving even with small gradient norms. For Skip-Gram with Negative Sampling and Adam, default hyperparameters ($\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$) require minimal tuning.

___

## Summary

This notebook derived the gradients for Skip-Gram with Negative Sampling and examined two adaptive gradient methods suited to the sparse optimization problem.

> __Key Takeaways:__
>
> * **Sparse gradients drive the design:** Each Skip-Gram training example updates only the target word's input embedding and the output embeddings for the context word and negative samples, leaving all other parameters unchanged. This sparsity means frequent words accumulate many updates while rare words receive few, making a single learning rate suboptimal.
> * **AdaGrad adapts but decays:** AdaGrad maintains per-parameter accumulated squared gradients, automatically reducing learning rates for frequent words and preserving them for rare words. However, the monotonic growth of the accumulator causes learning rates to decay to zero over long training runs.
> * **Adam bounds the decay:** Adam replaces cumulative sums with exponential moving averages of both gradients and squared gradients, providing adaptive per-parameter learning rates that can increase or decrease based on recent gradient history. This prevents the indefinite decay of AdaGrad while retaining its sparse-friendly properties.

___

## References
* [Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. NeurIPS 2013.](https://arxiv.org/abs/1310.4546)
* [Rong, X. (2014). word2vec Parameter Learning Explained. ArXiv, abs/1411.2738.](https://arxiv.org/abs/1411.2738)
* [Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. JMLR, 12, 2121-2159.](https://jmlr.org/papers/v12/duchi11a.html)
* [Kingma, D.P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR 2015.](https://arxiv.org/abs/1412.6980)

___
