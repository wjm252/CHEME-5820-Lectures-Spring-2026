# Training
Suppose we have a training dataset $\mathcal{D} = \{(\mathbf{x}_{1},\mathbf{y}_{1}),\dotsc,(\mathbf{x}_{n},\mathbf{y}_{n})\}$ with $n$ examples, where 
$\mathbf{x}_{i}\in\mathbb{R}^{d_{in}}$ is the $i$-th feature vector and $\mathbf{y}_{i}$ is the corresponding output. The output can be a discrete label for classification tasks (e.g., $y_{i}\in\{0,1,\dots,K-1\}$ for $K$ classes) or a continuous value for regression tasks (e.g., $y_{i}\in\mathbb{R}$).

Feedforward neural networks are trained using [the _backpropagation_ algorithm](https://en.wikipedia.org/wiki/Backpropagation), a _supervised learning_ method based on gradient descent. Backpropagation computes the gradient of a loss function with respect to network weights and biases using the chain rule.

> __Alternative Optimization Methods__: Other optimization algorithms such as genetic algorithms, particle swarm optimization, and simulated annealing can theoretically be used. However, gradient descent and its variants remain the most common due to computational efficiency and well-understood convergence properties.

The algorithm involves two steps:   
1. **Forward Pass**: Compute the network output for an input by passing it through each layer and applying activation functions, yielding a predicted output $\hat{\mathbf{y}}$.  
2. **Backward Pass**: Compute the gradient of the loss function with respect to each parameter by propagating the error backward through the network using the chain rule.

### Forward Pass
__Initialize__: Set the weights and biases of the network randomly or using an initialization heuristic. Let $\mathbf{z}^{(0)} = (x_{1},x_{2},\dots,x_{d_{in}}, 1)^{\top}$ be the augmented input vector.

For each layer $i=1,2,\dots,L$ __do__:
1. For each node $j=1,2,\dots,m_{i}$ in layer $i$ __do__:
      1. Compute the pre-activation: $a^{(i)}_{j} = (\mathbf{z}^{(i-1)})^{\top}\mathbf{w}^{(i)}_{j}$
      2. Compute the activation: $z^{(i)}_{j} = \sigma_{i}(a^{(i)}_{j})$
2. Store the layer output: $\mathbf{z}^{(i)} = (z^{(i)}_{1}, z^{(i)}_{2},\dots,z^{(i)}_{m_{i}})^{\top}$
3. The output of the final layer is the predicted output: $\hat{\mathbf{y}} = \mathbf{z}^{(L)}$

### Backward Pass (Gradient Descent)
The backward pass computes the gradient of the loss function with respect to each parameter by propagating the error backward through the network using the chain rule.

__Loss Function__: The loss function $\mathcal{L}(\mathbf{y},\hat{\mathbf{y}})$ measures the difference between the actual output $\mathbf{y}$ and predicted output $\hat{\mathbf{y}} = f_{\theta}(\mathbf{x})$. The loss is large when the prediction is far from the actual value and small when close. For regression tasks, the mean squared error (MSE) loss $\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}\|\mathbf{y}_i - \hat{\mathbf{y}}_i\|_{2}^{2}$ is commonly used, while [cross-entropy loss](https://en.wikipedia.org/wiki/Cross-entropy) is typical for classification.

We assume $\mathcal{L}$ is differentiable with respect to the parameters, allowing us to compute the gradient $\nabla_{\theta}{\mathcal{L}}$. The gradient points in the direction of steepest increase. We iteratively update parameters to minimize the loss:
$$
\begin{equation*}
\theta_{k+1} = \theta_{k} - \alpha\cdot\nabla_{\theta}\mathcal{L}(\theta_{k})\quad\text{where }k = 0,1,2,\dots
\end{equation*}
$$
where $k$ is the iteration index, $\nabla_{\theta}\mathcal{L}$ is the gradient of the loss with respect to $\theta$, and $\alpha > 0$ is the _learning rate_.

__Stopping Criteria__: Gradient descent continues until a stopping criterion is met, such as parameter convergence $\|\theta_{k+1} - \theta_{k}\|_{2} \leq \epsilon$, small gradient magnitude $\|\nabla_{\theta}\mathcal{L}(\theta_{k})\|_{2} \leq \epsilon$, or reaching maximum iterations.

## Stochastic Gradient Descent
Computing the full gradient over all training examples can be expensive. Stochastic Gradient Descent (SGD) is a less expensive approximation. Let $\mathcal{L}(\theta)$ denote the average loss over all $n$ training examples:
$$
\begin{equation*}
\mathcal{L}(\theta) = \frac{1}{n}\sum_{i=1}^{n}\mathcal{L}_{i}(\theta)
\end{equation*}
$$
where $\mathcal{L}_{i}(\theta)$ is the loss on example $i$. The full gradient descent update is:
$$
\begin{equation*}
\theta_{k+1} = \theta_{k} - \frac{\alpha}{n}\sum_{i=1}^{n}\nabla_{\theta}\mathcal{L}_{i}(\theta_{k})
\end{equation*}
$$

In _stochastic gradient descent_, we approximate the full gradient using a single randomly sampled training example:
$$
\begin{equation*}
\theta \gets \theta - \alpha\cdot\nabla_{\theta}\mathcal{L}_{i}(\theta)
\end{equation*}
$$
where $i$ is randomly selected from $\mathcal{D}$.

__Initialize__: Choose initial parameters $\theta$ and learning rate $\alpha$.

While not converged __do__:
1. Store current parameters: $\theta_{\text{old}} \gets \theta$
2. Randomly shuffle the training data
3. For $i = 1,2,\dots,n$ __do__:
    1. Compute the update: $\theta \gets \theta - \alpha\cdot\nabla_{\theta}\mathcal{L}_{i}(\theta)$
4. Check convergence: if $\|\theta - \theta_{\text{old}}\|_{2} \leq \epsilon$, then converged
5. Optionally update learning rate $\alpha$

### Mini-Batch Gradient Descent
Mini-batch gradient descent randomly samples a batch of $b$ training examples at each iteration:
$$
\begin{equation*}
\theta \gets \theta - \frac{\alpha}{b}\sum_{i=1}^{b}\nabla_{\theta}\mathcal{L}_{i}(\theta)
\end{equation*}
$$
where $b$ is the mini-batch size.

> __Mini-Batch Size__: The mini-batch size is a hyperparameter. Small mini-batches can lead to faster convergence but with more noise. Larger mini-batches provide more stable convergence but may be slower per epoch. The choice depends on the problem and available computational resources.

The computation of $\nabla_{\theta}\mathcal{L}_{i}(\theta)$ is simplified using [automatic differentiation](https://arxiv.org/abs/1502.05767), which efficiently computes derivatives of compositions of elementary functions.
