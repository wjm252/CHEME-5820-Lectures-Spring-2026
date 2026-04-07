# Advanced: The Adam Optimizer

The autoencoder example (L10c) uses Adam to train a nonlinear autoencoder on MNIST. In the L9c and L10a Advanced notebooks, we introduced Adam briefly as a comparison to AdaGrad, but did not examine the algorithm in detail. This notebook provides a self-contained treatment of Adam: the motivation behind its design, the full update rule with bias correction, and practical guidance for hyperparameter selection.

> __Learning Objectives:__
>
> By the end of this notebook, you should be able to:
>
> * __Describe the Adam update rule:__ Write out the first moment, second moment, bias correction, and parameter update steps. Explain the role of each component in the algorithm.
> * __Explain bias correction:__ Derive why the raw moment estimates are biased toward zero at early iterations and show how the correction factors $1/(1 - \beta_1^t)$ and $1/(1 - \beta_2^t)$ remove this bias.
> * __Select Adam hyperparameters:__ State the default values for $\beta_1$, $\beta_2$, $\epsilon$, and the learning rate $\eta$, and describe when and how to adjust them.

Let's get started!

___

## Motivation: Limitations of SGD and AdaGrad
Stochastic gradient descent (SGD) updates all parameters with the same learning rate $\eta$:
$$\theta_{t+1} = \theta_{t} - \eta\,\mathbf{g}_{t}$$
where $\mathbf{g}_{t} = \nabla_{\theta}\mathcal{L}(\theta_{t})$ is the gradient at step $t$. This has two problems for training neural networks like autoencoders: each update depends only on the current gradient (no smoothing of noisy estimates across steps), and all parameters share the same step size (regardless of whether they receive large or small gradients).

AdaGrad (L10a Advanced) addresses the second problem by maintaining a per-parameter accumulator of squared gradients $\mathbf{G}_{t} = \sum_{s=1}^{t}\mathbf{g}_{s}\odot\mathbf{g}_{s}$ and scaling the learning rate by $1/\sqrt{\mathbf{G}_{t}}$. This gives each parameter its own effective learning rate, but introduces new limitations:

* **Monotonic decay:** The accumulator $\mathbf{G}_{t}$ only grows, so the effective learning rate $\eta/\sqrt{G_{t,i}}$ decreases monotonically for every parameter. For long training runs, this can reduce the learning rate to near zero before the model converges.
* **No momentum:** Like SGD, AdaGrad uses only the current gradient direction, with no smoothing of noisy gradient estimates across steps.

Adam combines solutions to both problems: exponential moving averages provide momentum and adaptive per-parameter learning rates without monotonic decay.

___

## The Adam Algorithm
Adam (Adaptive Moment Estimation) maintains two exponential moving averages: a first moment estimate $\mathbf{m}_{t}$ that tracks the mean gradient direction (momentum), and a second moment estimate $\mathbf{v}_{t}$ that tracks the mean squared gradient magnitude (adaptive scaling). Both are initialized to zero and updated at each step.

> __Definition (Adam update rule):__
>
> Given learning rate $\eta > 0$, decay rates $\beta_1, \beta_2 \in [0,1)$, and stability constant $\epsilon > 0$, initialize $\mathbf{m}_{0} = \mathbf{0}$ and $\mathbf{v}_{0} = \mathbf{0}$ with the same shape as the parameter vector $\theta$. At each step $t = 1, 2, \dots$, compute the gradient $\mathbf{g}_{t} = \nabla_{\theta}\mathcal{L}(\theta_{t-1})$ and update:
>
> $$\begin{align*}
> \mathbf{m}_{t} &= \beta_1\,\mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_{t} && \text{(first moment update)} \\
> \mathbf{v}_{t} &= \beta_2\,\mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_{t} \odot \mathbf{g}_{t} && \text{(second moment update)} \\
> \hat{\mathbf{m}}_{t} &= \mathbf{m}_{t}\,/\,(1 - \beta_1^{t}) && \text{(bias-corrected first moment)} \\
> \hat{\mathbf{v}}_{t} &= \mathbf{v}_{t}\,/\,(1 - \beta_2^{t}) && \text{(bias-corrected second moment)} \\
> \theta_{t} &= \theta_{t-1} - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_{t}} + \epsilon} \odot \hat{\mathbf{m}}_{t} && \text{(parameter update)}
> \end{align*}$$
>
> where $\odot$ denotes element-wise multiplication and all operations (square root, division, addition of $\epsilon$) are element-wise.

The rest of this section examines each component of the algorithm: what the two moments do, why bias correction is needed, and what the resulting update looks like in practice.

### First Moment: Momentum
The first moment $\mathbf{m}_{t} = \beta_1\,\mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_{t}$ is an exponential moving average of the gradient. Expanding the recurrence shows that $\mathbf{m}_{t}$ is a weighted sum of all past gradients with exponentially decaying weights:

> __Expansion of the first moment:__
>
> $$\mathbf{m}_{t} = (1 - \beta_1)\sum_{s=1}^{t} \beta_1^{t-s}\,\mathbf{g}_{s}$$
>
> The gradient from $s$ steps ago is weighted by $\beta_1^{t-s}$. With the default $\beta_1 = 0.9$, the weight halves roughly every 7 steps ($0.9^{7} \approx 0.48$), so $\mathbf{m}_{t}$ reflects approximately the last 10 gradients.

This provides momentum: if the gradient consistently points in the same direction, $\mathbf{m}_{t}$ accumulates magnitude in that direction, accelerating convergence. If the gradient oscillates (alternating signs), the contributions cancel, damping the oscillation. SGD and AdaGrad lack this smoothing and are more sensitive to noisy gradients.

### Second Moment: Adaptive Learning Rates
The second moment $\mathbf{v}_{t} = \beta_2\,\mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_{t} \odot \mathbf{g}_{t}$ plays the same role as AdaGrad's accumulator $\mathbf{G}_{t}$, providing per-parameter scaling of the learning rate. The key difference is how old gradients are weighted.

> __AdaGrad vs. Adam second moment:__
>
> * **AdaGrad:** $\mathbf{G}_{t} = \sum_{s=1}^{t} \mathbf{g}_{s} \odot \mathbf{g}_{s}$. Every past gradient contributes equally and permanently. The accumulator grows monotonically, so the effective learning rate $\eta/\sqrt{G_{t,i}}$ can only decrease.
> * **Adam:** $\mathbf{v}_{t} = (1 - \beta_2)\sum_{s=1}^{t}\beta_2^{t-s}\,\mathbf{g}_{s} \odot \mathbf{g}_{s}$. Old gradients are exponentially forgotten. With $\beta_2 = 0.999$, the effective window is approximately 1000 steps ($0.999^{1000} \approx 0.37$). If a parameter's recent gradients are small, $\mathbf{v}_{t}$ decreases and the effective learning rate increases.

This bounded memory is what prevents Adam's learning rate from decaying to zero. If the gradient landscape changes during training (which is common in neural networks as different layers learn at different rates), Adam adapts its step sizes to the current gradient statistics rather than being anchored to the entire history.

### Bias Correction
Both moment estimates are initialized to $\mathbf{m}_{0} = \mathbf{0}$ and $\mathbf{v}_{0} = \mathbf{0}$. This introduces a bias toward zero in the early iterations. To see why, consider the second moment under stationary gradient statistics with $\mathbb{E}[\mathbf{g}_{t} \odot \mathbf{g}_{t}] = \boldsymbol{\nu}$ for all $t$.

> __Derivation of bias in $\mathbf{v}_{t}$:__
>
> Expanding the recurrence with $\mathbf{v}_{0} = \mathbf{0}$:
> $$\mathbb{E}[\mathbf{v}_{t}] = (1-\beta_2)\sum_{s=1}^{t}\beta_2^{t-s}\,\boldsymbol{\nu} = \boldsymbol{\nu}\,(1-\beta_2)\cdot\frac{1-\beta_2^{t}}{1-\beta_2} = \boldsymbol{\nu}\,(1 - \beta_2^{t})$$
>
> The factor $(1 - \beta_2^{t})$ is less than 1 for all finite $t$, so $\mathbb{E}[\mathbf{v}_{t}] < \boldsymbol{\nu}$. At $t = 1$ with $\beta_2 = 0.999$: $\mathbb{E}[\mathbf{v}_{1}] = 0.001\,\boldsymbol{\nu}$, meaning the raw estimate captures only 0.1\% of the true second moment. The same argument applies to $\mathbf{m}_{t}$: $\mathbb{E}[\mathbf{m}_{t}] = \boldsymbol{\mu}\,(1 - \beta_1^{t})$ where $\boldsymbol{\mu} = \mathbb{E}[\mathbf{g}_{t}]$.

Dividing by the bias factor gives unbiased estimates: $\hat{\mathbf{m}}_{t} = \mathbf{m}_{t}/(1 - \beta_1^{t})$ and $\hat{\mathbf{v}}_{t} = \mathbf{v}_{t}/(1 - \beta_2^{t})$, since $\mathbb{E}[\hat{\mathbf{v}}_{t}] = \boldsymbol{\nu}(1 - \beta_2^{t})/(1 - \beta_2^{t}) = \boldsymbol{\nu}$. The correction is largest at early steps: at $t = 1$ it multiplies $\mathbf{v}_{1}$ by $1/0.001 = 1000$. By $t = 1000$ the factor is approximately 1.58, and by $t = 5000$ it is approximately 1.007. In practice, the bias correction matters for roughly the first few hundred steps and is negligible thereafter.

### Effective Step Size
The parameter update for component $i$ is $\Delta\theta_{t,i} = -\eta\,\hat{m}_{t,i}/(\sqrt{\hat{v}_{t,i}} + \epsilon)$. When $|\hat{m}_{t,i}|$ and $\sqrt{\hat{v}_{t,i}}$ are both dominated by recent gradients of similar magnitude, their ratio is approximately 1, so the step size is approximately $\eta$ regardless of the gradient magnitude. This means Adam's updates are roughly bounded by the learning rate, providing stability even when gradients are very large during early training. This approximate bound is one reason Adam is less sensitive to learning rate selection than SGD, where a learning rate that is too large causes divergence with no built-in safeguard.

___

## Practical Considerations
This section compares Adam to the other optimizers we have studied and provides guidance on hyperparameter selection.

### Comparison of Optimizers
The table below compares SGD, AdaGrad, and Adam on properties relevant to training neural networks such as autoencoders.

| Property | SGD | AdaGrad | Adam |
|----------|-----|---------|------|
| Per-parameter learning rates | No | Yes | Yes |
| Momentum | No | No | Yes (first moment $\mathbf{m}_{t}$) |
| Learning rate decay behavior | Constant $\eta$ | Monotonic decay to zero | Bounded: adapts to recent gradient history |
| Memory per parameter | $0\times$ | $1\times$ (accumulator $\mathbf{G}$) | $2\times$ (moments $\mathbf{m}$, $\mathbf{v}$) |
| Hyperparameters | $\eta$ | $\eta$ | $\eta$, $\beta_1$, $\beta_2$, $\epsilon$ |
| Sensitivity to $\eta$ | High | Moderate | Low |

Adam's main cost relative to AdaGrad is memory: it stores two state vectors per parameter instead of one. For the autoencoder in L10c with 202,258 parameters, this means storing approximately 404,516 additional floating-point values (about 1.6 MB in Float32), which is negligible.

### Hyperparameter Guidelines
The original Adam paper proposes default values that work well across a wide range of problems.

> __Default hyperparameters [(Kingma & Ba, 2015)](https://arxiv.org/abs/1412.6980):__
>
> | Hyperparameter | Default | Role |
> |----------------|---------|------|
> | $\eta$ (learning rate) | $0.001$ | Controls the overall step size |
> | $\beta_1$ (first moment decay) | $0.9$ | Controls the momentum window (roughly the last 10 gradients) |
> | $\beta_2$ (second moment decay) | $0.999$ | Controls the adaptive scaling window (roughly the last 1000 gradients) |
> | $\epsilon$ (stability constant) | $10^{-8}$ | Prevents division by zero when $\hat{\mathbf{v}}_{t} \approx \mathbf{0}$ |

In practice, $\beta_1$, $\beta_2$, and $\epsilon$ rarely need tuning. The learning rate $\eta$ is the primary hyperparameter to adjust:

* **Learning rate $\eta$:** If training loss does not decrease, try increasing $\eta$ by a factor of 3-10. If training is unstable (loss spikes or diverges), reduce $\eta$. Learning rate schedules that reduce $\eta$ over time can improve final performance.
* **$\beta_1$:** Reducing $\beta_1$ (e.g., to 0.5) decreases momentum, which can help when the loss surface changes rapidly. Setting $\beta_1 = 0$ disables momentum entirely, reducing Adam to a bias-corrected version of RMSProp.
* **$\beta_2$:** Reducing $\beta_2$ (e.g., to 0.99) shortens the adaptive scaling window, making Adam more responsive to recent gradient changes. This can help if the gradient distribution shifts significantly during training.
* **$\epsilon$:** Increasing $\epsilon$ (e.g., to $10^{-4}$) can stabilize training when gradients are very small, as it prevents excessively large effective learning rates.

___

## Summary
Adam combines momentum (exponential moving average of gradients) with adaptive per-parameter learning rates (exponential moving average of squared gradients) to produce an optimizer that is stable, efficient, and requires minimal hyperparameter tuning.

> __Key Takeaways:__
>
> * __Two moving averages:__ The first moment $\mathbf{m}_{t}$ smooths noisy gradients by averaging over recent steps, providing momentum. The second moment $\mathbf{v}_{t}$ scales each parameter's learning rate based on recent gradient magnitudes, providing per-parameter adaptation.
> * __Bias correction compensates for zero initialization:__ Because $\mathbf{m}_{0} = \mathbf{v}_{0} = \mathbf{0}$, the raw moment estimates are biased toward zero at early steps. Dividing by $(1 - \beta^{t})$ removes this bias, with the correction vanishing as $t$ grows.
> * __Bounded effective step size:__ Adam's update magnitude is approximately bounded by the learning rate $\eta$, regardless of gradient scale. This makes Adam less sensitive to learning rate selection than SGD and avoids the monotonic learning rate decay of AdaGrad.

___

## References
* [Kingma, D.P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR 2015.](https://arxiv.org/abs/1412.6980)
* [Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. JMLR, 12, 2121-2159.](https://jmlr.org/papers/v12/duchi11a.html)
* [Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning, Chapter 8: Optimization for Training Deep Models.](https://www.deeplearningbook.org/contents/optimization.html)

___
