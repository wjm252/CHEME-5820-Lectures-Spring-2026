"""
    build_noise_distribution(sentences::Array{String,1}, vocabulary::Dict{String, Int64}) -> Vector{Float64}

Build the noise distribution for Negative Sampling, where each word's sampling probability
is proportional to its corpus frequency raised to the 3/4 power: P_n(w) ∝ f(w)^(3/4).

### Arguments
- `sentences::Array{String,1}`: array of sentence strings.
- `vocabulary::Dict{String, Int64}`: dictionary mapping words to vocabulary indices.

### Returns
- `Vector{Float64}`: probability vector of length `vocab_size`, summing to 1.
"""
function build_noise_distribution(sentences::Array{String,1}, vocabulary::Dict{String, Int64})::Vector{Float64}

    # initialize -
    vocab_size = length(vocabulary);
    word_counts = zeros(Float64, vocab_size);

    # count word frequencies -
    for sentence in sentences
        augmented_sentence = "<bos> " * sentence * " <eos>";
        words = split(lowercase(augmented_sentence)) .|> String;
        for word in words
            idx = get(vocabulary, word, vocabulary["<unk>"]);
            word_counts[idx] += 1.0;
        end
    end

    # apply 3/4 power and normalize -
    freq_34 = word_counts .^ 0.75;
    return freq_34 / sum(freq_34);
end

"""
    train_skipgram_ns(training_pairs::Array{Tuple{Vector{Float64}, Vector{Float64}}, 1},
        vocab_size::Int64, noise_distribution::Vector{Float64};
        d_h::Int64 = 5, η::Float64 = 0.01, num_epochs::Int64 = 500,
        k::Int64 = 5) -> Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}}

Train a Skip-Gram model using Negative Sampling. For each (target, context) pair,
computes sigmoid scores for the positive context word and `k` randomly sampled negative words.

### Arguments
- `training_pairs::Array{Tuple{Vector{Float64}, Vector{Float64}}, 1}`: array of (target one-hot, context one-hot) pairs.
- `vocab_size::Int64`: size of the vocabulary.
- `noise_distribution::Vector{Float64}`: probability vector for sampling negative words.
- `d_h::Int64`: embedding dimension (default: 5).
- `η::Float64`: learning rate (default: 0.01).
- `num_epochs::Int64`: number of training epochs (default: 500).
- `k::Int64`: number of negative samples per positive example (default: 5).

### Returns
- `Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}}`: tuple of (W₁, W₂, loss_history).
"""
function train_skipgram_ns(training_pairs::Array{Tuple{Vector{Float64}, Vector{Float64}}, 1},
    vocab_size::Int64, noise_distribution::Vector{Float64};
    d_h::Int64 = 5, η::Float64 = 0.01, num_epochs::Int64 = 500,
    k::Int64 = 5)::Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}}

    # initialize weight matrices -
    W₁ = randn(d_h, vocab_size) * 0.1;
    W₂ = randn(vocab_size, d_h) * 0.1;

    # sigmoid helper -
    σ(x) = 1.0 / (1.0 + exp(-x));

    # training loop -
    loss_history = zeros(num_epochs);
    sampler = Categorical(noise_distribution);
    for epoch ∈ 1:num_epochs
        epoch_loss = 0.0;
        for (x_target, y_context) in training_pairs

            # get indices -
            target_idx = argmax(x_target);
            context_idx = argmax(y_context);

            # get embeddings -
            h = W₁[:, target_idx];
            w_pos = W₂[context_idx, :];

            # sample k negative words -
            neg_indices = [rand(sampler) for _ in 1:k];

            # forward pass: positive score -
            s_pos = dot(w_pos, h);
            loss = -log(σ(s_pos) + 1e-12);

            # gradient for positive example -
            ∂h = (σ(s_pos) - 1.0) * w_pos;
            W₂[context_idx, :] .-= η * (σ(s_pos) - 1.0) * h;

            # forward pass: negative scores -
            for ni in neg_indices
                w_neg = W₂[ni, :];
                s_neg = dot(w_neg, h);
                loss += -log(σ(-s_neg) + 1e-12);

                # gradient for negative example -
                ∂h .+= σ(s_neg) * w_neg;
                W₂[ni, :] .-= η * σ(s_neg) * h;
            end

            # update target embedding -
            W₁[:, target_idx] .-= η * ∂h;
            epoch_loss += loss;
        end
        loss_history[epoch] = epoch_loss / length(training_pairs);
    end

    return (W₁, W₂, loss_history);
end
