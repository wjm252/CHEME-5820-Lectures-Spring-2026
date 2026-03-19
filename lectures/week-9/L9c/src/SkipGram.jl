"""
    build_skipgram_pairs(sentences::Array{String,1}, vocabulary::Dict{String, Int64};
        window_size::Int64 = 2) -> Array{Tuple{Vector{Float64}, Vector{Float64}}, 1}

Generate Skip-Gram training pairs from a corpus. For each word position in each sentence,
each surrounding word within the window produces a separate (target one-hot, context one-hot) pair.

### Arguments
- `sentences::Array{String,1}`: array of sentence strings.
- `vocabulary::Dict{String, Int64}`: dictionary mapping words to vocabulary indices.
- `window_size::Int64`: number of tokens on each side to include in context (default: 2).

### Returns
- `Array{Tuple{Vector{Float64}, Vector{Float64}}, 1}`: array of (target one-hot, context one-hot) tuples.
"""
function build_skipgram_pairs(sentences::Array{String,1}, vocabulary::Dict{String, Int64};
    window_size::Int64 = 2)::Array{Tuple{Vector{Float64}, Vector{Float64}}, 1}

    # initialize -
    vocab_size = length(vocabulary);
    pairs = Array{Tuple{Vector{Float64}, Vector{Float64}}, 1}();

    # generate training pairs -
    for sentence in sentences
        augmented_sentence = "<bos> " * sentence * " <eos>";
        words = split(lowercase(augmented_sentence)) .|> String;
        word_indices = [get(vocabulary, word, vocabulary["<unk>"]) for word in words];

        for i ∈ eachindex(word_indices)
            target_index = word_indices[i];
            target_onehot = zeros(Float64, vocab_size);
            target_onehot[target_index] = 1.0;

            left = max(1, i - window_size);
            right = min(length(word_indices), i + window_size);
            for j ∈ left:right
                if j == i
                    continue
                end
                context_onehot = zeros(Float64, vocab_size);
                context_onehot[word_indices[j]] = 1.0;
                push!(pairs, (target_onehot, context_onehot));
            end
        end
    end

    return pairs;
end

"""
    train_skipgram_softmax(training_pairs::Array{Tuple{Vector{Float64}, Vector{Float64}}, 1}, vocab_size::Int64;
        d_h::Int64 = 5, η::Float64 = 0.01, num_epochs::Int64 = 500) -> Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}}

Train a Skip-Gram model using gradient descent with full softmax.
The forward pass computes h = W₁ * x_target, u = W₂ * h, ŷ = softmax(u).

### Arguments
- `training_pairs::Array{Tuple{Vector{Float64}, Vector{Float64}}, 1}`: array of (target one-hot, context one-hot) pairs.
- `vocab_size::Int64`: size of the vocabulary.
- `d_h::Int64`: embedding dimension (default: 5).
- `η::Float64`: learning rate (default: 0.01).
- `num_epochs::Int64`: number of training epochs (default: 500).

### Returns
- `Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}}`: tuple of (W₁, W₂, loss_history).
"""
function train_skipgram_softmax(training_pairs::Array{Tuple{Vector{Float64}, Vector{Float64}}, 1}, vocab_size::Int64;
    d_h::Int64 = 5, η::Float64 = 0.01, num_epochs::Int64 = 500)::Tuple{Matrix{Float64}, Matrix{Float64}, Vector{Float64}}

    # initialize weight matrices -
    W₁ = randn(d_h, vocab_size) * 0.1;
    W₂ = randn(vocab_size, d_h) * 0.1;

    # training loop -
    loss_history = zeros(num_epochs);
    for epoch ∈ 1:num_epochs
        epoch_loss = 0.0;
        for (x_target, y_context) in training_pairs

            # forward pass -
            h = W₁ * x_target;
            u = W₂ * h;
            ŷ = softmax(u);

            # cross-entropy loss -
            loss = -sum(y_context .* log.(ŷ .+ 1e-12));
            epoch_loss += loss;

            # backward pass -
            δ_u = ŷ - y_context;
            ∂L_∂W₂ = δ_u * h';
            δ_h = W₂' * δ_u;
            ∂L_∂W₁ = δ_h * x_target';

            # gradient descent update -
            W₁ .-= η * ∂L_∂W₁;
            W₂ .-= η * ∂L_∂W₂;
        end
        loss_history[epoch] = epoch_loss / length(training_pairs);
    end

    return (W₁, W₂, loss_history);
end
