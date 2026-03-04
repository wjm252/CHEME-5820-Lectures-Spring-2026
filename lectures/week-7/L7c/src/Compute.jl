"""
    learn(model::MySimpleBoltzmannMachineModel, X::Matrix{Int};
          η::Float64 = 0.01, β::Float64 = 1.0,
          T::Int = 10000, N_epochs::Int = 100,
          λ::Float64 = 0.0) -> Tuple{MySimpleBoltzmannMachineModel, Vector{Float64}}

Train a Boltzmann machine using exact gradient ascent.

For each epoch, the function runs Gibbs sampling until the chain reaches its stationary
distribution, discards the first half of samples as burn-in, then updates weights and
biases by gradient ascent on the log-likelihood.

### Arguments
- `model::MySimpleBoltzmannMachineModel`: Boltzmann machine model to train (mutated in place).
- `X::Matrix{Int}`: training data matrix of size `n × m`, where each column is a binary pattern in {-1,1}.
- `η::Float64`: learning rate (default: 0.01).
- `β::Float64`: inverse temperature used for Gibbs sampling (default: 1.0).
- `T::Int`: number of Gibbs steps per epoch (default: 10000).
- `N_epochs::Int`: number of training epochs (default: 100).
- `λ::Float64`: L2 weight decay coefficient (default: 0.0). A positive value penalizes large
  weights and prevents the model distribution from collapsing to a single configuration.

### Returns
- `model::MySimpleBoltzmannMachineModel`: updated model with trained weights and biases.
- `history::Vector{Float64}`: Frobenius norm of ΔW at each epoch.
"""
function learn(model::MySimpleBoltzmannMachineModel, X::Matrix{Int};
               η::Float64 = 0.01, β::Float64 = 1.0,
               T::Int = 10000, N_epochs::Int = 100,
               λ::Float64 = 0.0)::Tuple{MySimpleBoltzmannMachineModel, Vector{Float64}}

    # initialize -
    n = size(X, 1);           # number of nodes
    m = size(X, 2);           # number of training patterns
    history = Vector{Float64}(undef, N_epochs);  # convergence history

    # precompute data expectations (done once, outside the loop) -
    C_data = (1/m) * (X * X');        # pairwise correlation matrix (n×n)
    b_data = vec(mean(X, dims=2));    # single-node averages (n-vector)

    # train -
    for e ∈ 1:N_epochs

        # pick a random initial state -
        sₒ = rand([-1, 1], n);

        # run Gibbs sampling to stationarity -
        S = VLDataScienceMachineLearningPackage.sample(model, sₒ, T = T, β = β);

        # discard burn-in (first T÷2 columns), use remaining samples -
        S_stat = S[:, (T ÷ 2 + 1):end];

        # compute model expectations from stationary samples -
        m_stat = size(S_stat, 2);
        C_model = (1/m_stat) * (S_stat * S_stat');
        b_model = vec(mean(S_stat, dims=2));

        # compute updates (with optional L2 weight decay) -
        ΔW = η * (C_data - C_model) - λ .* model.W;
        Δb = η * (b_data - b_model);

        # symmetrize and zero the diagonal of ΔW -
        ΔW = (ΔW + ΔW') / 2 - diagm(diag(ΔW));

        # apply updates -
        model.W += ΔW;
        model.b += Δb;

        # record Frobenius norm of ΔW -
        history[e] = norm(vec(ΔW));
    end

    # return -
    return (model, history);
end
