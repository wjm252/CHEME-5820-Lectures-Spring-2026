"""
    build_cbow_pairs(sentences, vocabulary; window_size=2) -> Vector

Build context-target training pairs from a tokenized corpus for CBOW training.

For each target word at position `i` in a sentence, the context is the average of
one-hot vocabulary vectors for all words within `window_size` positions on either
side of `i`. Returns a vector of `(context_avg, target)` tuples where both
`context_avg` and `target` are `Float64` vectors of length `|vocabulary|`.

### Arguments
- `sentences::Array{String,1}`: raw sentences; each sentence is lowercased and split on whitespace.
- `vocabulary::Dict{String,Int64}`: mapping from token to 1-based vocabulary index.
- `window_size::Int64`: half-width of the context window (default: 2).

### Returns
- `::Vector`: a vector of `(context_avg, target)` tuples, one per valid target word.
"""
function build_cbow_pairs(sentences::Array{String,1}, vocabulary::Dict{String,Int64}; window_size::Int64=2)

    # initialize -
    vocab_size = length(vocabulary);
    training_pairs = [];

    # main loop: iterate over sentences -
    for sentence in sentences

        # tokenize and get sentence length -
        words = split(lowercase(sentence));
        n = length(words);

        for i in 1:n

            # look up target word index; skip if unknown -
            target_word = get(vocabulary, words[i], get(vocabulary, "<unk>", 0));
            if target_word == 0
                continue
            end

            # build one-hot target vector -
            target = zeros(Float64, vocab_size);
            target[target_word] = 1.0;

            # accumulate context word vectors in the window -
            context_sum = zeros(Float64, vocab_size);
            count = 0;
            for j in max(1, i - window_size):min(n, i + window_size)
                if j == i
                    continue
                end
                ctx_word = get(vocabulary, words[j], get(vocabulary, "<unk>", 0));
                if ctx_word > 0
                    context_sum[ctx_word] += 1.0;
                    count += 1;
                end
            end

            # store the averaged context vector and target -
            if count > 0
                push!(training_pairs, (context_sum ./ count, target));
            end
        end
    end

    # return -
    return training_pairs;
end

"""
    train_cbow(training_pairs, vocab_size; d_h=5, eta=0.01, num_epochs=500, verbose=true, print_every=100) -> Tuple

Train a CBOW model on `training_pairs` using stochastic gradient descent.

The model has a single hidden layer of size `d_h`. The input weight matrix
`W1 ∈ ℝ^{d_h × vocab_size}` maps context vectors to the hidden layer, and
`W2 ∈ ℝ^{vocab_size × d_h}` maps the hidden layer to the output distribution.
Training minimizes cross-entropy loss with a softmax output.

### Arguments
- `training_pairs`: vector of `(context_avg, target)` tuples produced by `build_cbow_pairs`.
- `vocab_size::Int64`: number of tokens in the vocabulary.
- `d_h::Int64`: embedding (hidden layer) dimension (default: 5).
- `eta::Float64`: SGD learning rate (default: 0.01).
- `num_epochs::Int64`: number of full passes over `training_pairs` (default: 500).
- `verbose::Bool`: print loss every `print_every` epochs when `true` (default: true).
- `print_every::Int64`: print interval in epochs when `verbose=true` (default: 100).

### Returns
- `W1::Matrix{Float64}`: input weight matrix of size `d_h × vocab_size`; columns are word embeddings.
- `W2::Matrix{Float64}`: output weight matrix of size `vocab_size × d_h`.
- `loss_history::Vector{Float64}`: average cross-entropy loss per epoch.
"""
function train_cbow(training_pairs, vocab_size::Int64;
    d_h::Int64=5, eta::Float64=0.01, num_epochs::Int64=500, verbose::Bool=true, print_every::Int64=100)

    # initialize weight matrices with small random values -
    W1 = randn(d_h, vocab_size) * 0.01;
    W2 = randn(vocab_size, d_h) * 0.01;
    loss_history = zeros(Float64, num_epochs);

    # main training loop -
    for epoch in 1:num_epochs

        total_loss = 0.0;

        for (x, y) in training_pairs

            # forward pass: hidden layer and output scores -
            h = W1 * x;
            u = W2 * h;

            # stable softmax -
            exp_u = exp.(u .- maximum(u));
            y_hat = exp_u ./ sum(exp_u);

            # cross-entropy loss -
            loss = -sum(y .* log.(y_hat .+ 1e-10));
            total_loss += loss;

            # backward pass: output layer gradient -
            delta_u = y_hat .- y;
            dW2 = delta_u * h';

            # backward pass: hidden layer gradient -
            delta_h = W2' * delta_u;
            dW1 = delta_h * x';

            # SGD weight update -
            W2 .-= eta .* dW2;
            W1 .-= eta .* dW1;
        end

        # record average loss for this epoch -
        loss_history[epoch] = total_loss / length(training_pairs);

        # print progress -
        if verbose && (epoch % print_every == 0 || epoch == 1)
            println("Epoch $(lpad(epoch, ndigits(num_epochs)))/$(num_epochs)  loss = $(round(loss_history[epoch]; digits=4))");
        end
    end

    # return -
    return W1, W2, loss_history;
end
