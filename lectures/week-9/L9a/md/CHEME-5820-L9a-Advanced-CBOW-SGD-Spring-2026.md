# Advanced: Stochastic Gradient Descent and CBOW Gradient Derivation

This notebook derives the weight update rules for the Continuous Bag of Words (CBOW) model from first principles. We motivate stochastic gradient descent (SGD) as an efficient approximation to full-batch gradient descent, then derive the analytical gradients $\nabla_{\mathbf{W}_1}\mathcal{L}$ and $\nabla_{\mathbf{W}_2}\mathcal{L}$ via the chain rule.

> __Learning Objectives:__
>
> By the end of this notebook, you should be able to:
>
> * __Motivate stochastic gradient descent:__ Explain how SGD approximates the full-batch gradient with a single training example, and prove that the stochastic gradient is an unbiased estimator of the true gradient.
> * __Derive the CBOW gradients via the chain rule:__ Apply the softmax-cross-entropy gradient identity and the chain rule to derive closed-form gradient expressions for both weight matrices.
> * __Interpret the outer-product structure:__ Explain why both CBOW update rules reduce to rank-1 outer products and what this means for the geometry of the weight updates.

Let's get started!

___

## Stochastic Gradient Descent

Full-batch gradient descent minimizes the average loss over all $M$ training pairs $\{(\mathbf{x}^{(m)}, \mathbf{y}^{(m)})\}_{m=1}^{M}$:
$$\mathcal{L}(\mathbf{W}_1, \mathbf{W}_2) = \frac{1}{M}\sum_{m=1}^{M} \mathcal{L}^{(m)}(\mathbf{W}_1, \mathbf{W}_2)$$
Each update requires computing $\nabla_{\mathbf{W}} \mathcal{L}^{(m)}$ for every training pair before taking a single step. For a corpus with millions of sentences and a vocabulary of size $N_{\mathcal{V}} \approx 10^{6}$, this is prohibitive. Stochastic gradient descent replaces the full sum with a single randomly selected term.

> __Definition (Stochastic Gradient Descent):__
>
> Let $\mathcal{L}(\theta) = \frac{1}{M}\sum_{m=1}^{M}\mathcal{L}^{(m)}(\theta)$ be the average loss over $M$ training examples, where $\theta$ collects all parameters. At each iteration $t$, SGD draws one index $m_t$ uniformly at random from $\{1,\dots,M\}$ and updates:
>
> $$\theta^{(t+1)} = \theta^{(t)} - \eta\,\nabla_\theta\mathcal{L}^{(m_t)}\!\left(\theta^{(t)}\right)$$
>
> where $\eta > 0$ is the learning rate. The key property is that the stochastic gradient is an __unbiased estimator__ of the full gradient:
>
> $$\mathbb{E}_{m_t}\!\left[\nabla_\theta\mathcal{L}^{(m_t)}(\theta)\right] = \nabla_\theta\mathcal{L}(\theta)$$
>
> so each update moves in the correct direction on average, even though individual steps are noisy.

The variance in each gradient estimate is not purely a drawback. While individual steps are noisier than full-batch updates, the noise helps the optimizer escape shallow local minima and saddle points that trap a deterministic trajectory — an effect that becomes more pronounced as the loss surface grows more complex. Mini-batch SGD generalizes this idea by drawing a random subset $\mathcal{B}_t \subset \{1,\dots,M\}$ of size $B$ at each step:
$$\theta^{(t+1)} = \theta^{(t)} - \frac{\eta}{B}\sum_{m\in\mathcal{B}_t}\nabla_\theta\mathcal{L}^{(m)}\!\left(\theta^{(t)}\right)$$
The variance of the gradient estimate scales as $1/B$, so larger batches give smoother updates at the cost of more computation per step. Pure SGD ($B=1$) and full-batch gradient descent ($B=M$) are the two extremes; in practice, batch sizes of 32–512 are common.

The table below places SGD in context with full-batch gradient descent and momentum:

| Method | Gradient estimate | Update rule |
|--------|-------------------|-------------|
| Full-batch GD | Exact: $\frac{1}{M}\sum_m \nabla\mathcal{L}^{(m)}$ | $\theta \leftarrow \theta - \eta\,\nabla\mathcal{L}$ |
| Momentum | Exact gradient + history | $v \leftarrow \beta v - \eta\,\nabla\mathcal{L}$; $\;\theta \leftarrow \theta + v$ |
| SGD | Stochastic: $\nabla\mathcal{L}^{(m_t)}$ | $\theta \leftarrow \theta - \eta\,\nabla\mathcal{L}^{(m_t)}$ |
| Mini-batch SGD | $\frac{1}{B}\sum_{m\in\mathcal{B}}\nabla\mathcal{L}^{(m)}$ | $\theta \leftarrow \theta - \frac{\eta}{B}\sum_{m\in\mathcal{B}}\nabla\mathcal{L}^{(m)}$ |

For CBOW, each training pair $(\mathbf{x}^{(m)}, \mathbf{y}^{(m)})$ is one (context sum, target one-hot) pair. The standard training loop iterates over all pairs in random order, which is equivalent to SGD with $B=1$ and one full pass over the data per epoch.

___

## Deriving the CBOW Gradients

We derive $\nabla_{\mathbf{W}_1}\mathcal{L}$ and $\nabla_{\mathbf{W}_2}\mathcal{L}$ for a single training pair $(\mathbf{x}, \mathbf{y})$. Recall the CBOW forward pass:
$$\mathbf{h} = \mathbf{W}_1\,\mathbf{x}, \qquad \mathbf{u} = \mathbf{W}_2\,\mathbf{h}, \qquad \hat{y}_i = \frac{e^{u_i}}{\sum_{j=1}^{N_{\mathcal{V}}} e^{u_j}}, \qquad \mathcal{L} = -\sum_{i=1}^{N_{\mathcal{V}}} y_i\,\log\hat{y}_i$$
The gradient is computed by applying the chain rule backwards through the network, starting at the loss and working back through the softmax, $\mathbf{W}_2$, the hidden layer $\mathbf{h}$, and finally $\mathbf{W}_1$.

Begin with the gradient of $\mathcal{L}$ with respect to the pre-softmax logits $\mathbf{u}$. Expanding the chain rule through the cross-entropy:
$$\frac{\partial\mathcal{L}}{\partial u_i} = -\sum_{j=1}^{N_{\mathcal{V}}} y_j \frac{\partial \log \hat{y}_j}{\partial u_i} = -\sum_{j=1}^{N_{\mathcal{V}}} \frac{y_j}{\hat{y}_j}\,\frac{\partial \hat{y}_j}{\partial u_i}$$
The softmax Jacobian splits into two cases: for $j = i$, $\;\partial\hat{y}_i/\partial u_i = \hat{y}_i(1-\hat{y}_i)$; for $j \neq i$, $\;\partial\hat{y}_j/\partial u_i = -\hat{y}_j\hat{y}_i$. Substituting both cases and simplifying:
$$\begin{align*}
\frac{\partial\mathcal{L}}{\partial u_i}
  &= -y_i(1-\hat{y}_i) + \hat{y}_i\sum_{j \neq i} y_j
  && \text{two-case Jacobian}\\
  &= -y_i + \hat{y}_i\!\left(y_i + \sum_{j \neq i} y_j\right)
  = -y_i + \hat{y}_i\underbrace{\sum_{j=1}^{N_{\mathcal{V}}} y_j}_{=\,1}
  && \text{rearrange, one-hot property}\\
  &= \hat{y}_i - y_i
  && \text{since }\sum_j y_j = 1
\end{align*}$$
This is the __softmax-cross-entropy identity__: the gradient at the output equals the prediction error. Defining the error vector $\boldsymbol{\delta} = \hat{\mathbf{y}} - \mathbf{y} \in \mathbb{R}^{N_{\mathcal{V}}}$, we have $\nabla_{\mathbf{u}}\mathcal{L} = \boldsymbol{\delta}$. With $\boldsymbol{\delta}$ established, the gradient with respect to $\mathbf{W}_2$ follows from the linear relation $u_i = \sum_k W_{2,ik}\,h_k$, which gives $\partial u_i/\partial W_{2,ij} = h_j$:
$$\begin{align*}
\frac{\partial\mathcal{L}}{\partial W_{2,ij}}
  &= \frac{\partial\mathcal{L}}{\partial u_i}\,\frac{\partial u_i}{\partial W_{2,ij}} = \delta_i\,h_j
  && \text{chain rule, one term survives per }(i,j)
\end{align*}$$
Assembling over all $(i,j)$ gives the outer product $\nabla_{\mathbf{W}_2}\mathcal{L} = \boldsymbol{\delta}\,\mathbf{h}^{\top} \in \mathbb{R}^{N_{\mathcal{V}} \times d_h}$. The error $\boldsymbol{\delta}$ also propagates back through $\mathbf{W}_2$ to the hidden layer. Each element $h_k$ contributes to every output $u_i$ via $\partial u_i/\partial h_k = W_{2,ik}$, so summing the chain rule over all output units:
$$\begin{align*}
\frac{\partial\mathcal{L}}{\partial h_k}
  &= \sum_{i=1}^{N_{\mathcal{V}}} \frac{\partial\mathcal{L}}{\partial u_i}\,\frac{\partial u_i}{\partial h_k}
   = \sum_{i=1}^{N_{\mathcal{V}}} \delta_i\,W_{2,ik}
  && \partial u_i/\partial h_k = W_{2,ik}
\end{align*}$$
In matrix form, $\partial\mathcal{L}/\partial\mathbf{h} = \mathbf{W}_2^{\top}\boldsymbol{\delta} \in \mathbb{R}^{d_h}$. The chain rule now reaches $\mathbf{W}_1$ via $h_k = \sum_l W_{1,kl}\,x_l$. Because $W_{1,kl}$ affects only $h_k$ and not $h_{k'}$ for $k' \neq k$:
$$\begin{align*}
\frac{\partial\mathcal{L}}{\partial W_{1,kl}}
  &= \frac{\partial\mathcal{L}}{\partial h_k}\,\frac{\partial h_k}{\partial W_{1,kl}}
   = \left(\mathbf{W}_2^{\top}\boldsymbol{\delta}\right)_k x_l
  && \partial h_k/\partial W_{1,kl} = x_l
\end{align*}$$
Assembling over all $(k,l)$ gives the outer product $\nabla_{\mathbf{W}_1}\mathcal{L} = \left(\mathbf{W}_2^{\top}\boldsymbol{\delta}\right)\mathbf{x}^{\top} \in \mathbb{R}^{d_h \times N_{\mathcal{V}}}$. Both gradients are rank-1 outer products — a consequence of the single hidden layer and linear operations on either side of it.

### Summary of the Update Rules

The backward pass yields closed-form update rules that depend only on quantities already computed during the forward pass.

> __Definition (CBOW SGD Update Rules):__
>
> Let $\boldsymbol{\delta} = \hat{\mathbf{y}} - \mathbf{y} \in \mathbb{R}^{N_{\mathcal{V}}}$ be the prediction error for a single training pair $(\mathbf{x}, \mathbf{y})$, and let $\mathbf{h} = \mathbf{W}_1\mathbf{x}$. The SGD updates are:
>
> $$\mathbf{W}_2 \leftarrow \mathbf{W}_2 - \eta\,\boldsymbol{\delta}\,\mathbf{h}^{\top}$$
>
> $$\mathbf{W}_1 \leftarrow \mathbf{W}_1 - \eta\,\left(\mathbf{W}_2^{\top}\boldsymbol{\delta}\right)\mathbf{x}^{\top}$$
>
> Both updates depend only on the prediction error $\boldsymbol{\delta}$, the hidden state $\mathbf{h}$, and the input $\mathbf{x}$ — all quantities already computed during the forward pass. Note that $\mathbf{W}_2$ must be evaluated at its current value (before the update) when computing $\nabla_{\mathbf{W}_1}\mathcal{L}$.

___

## Summary

This notebook derived the weight update rules for the CBOW model from first principles, working from the SGD approximation through the chain rule to closed-form gradient expressions.

> __Key Takeaways:__
>
> * **SGD is an unbiased gradient estimator:** Replacing the full-sum gradient with a single randomly selected training example gives an unbiased estimate of the true gradient, so each update moves in the correct direction on average, even though individual steps are noisy.
> * **The softmax-cross-entropy gradient simplifies to the prediction error:** Differentiating the cross-entropy loss through the softmax reduces to the prediction error at each output unit, a result that follows from the one-hot target property and the two-case softmax Jacobian.
> * **Both update rules reduce to outer products:** The gradients of both weight matrices are rank-1 outer products, so each SGD step modifies each weight matrix along a single direction determined by the current prediction error.

___
