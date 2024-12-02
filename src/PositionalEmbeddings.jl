module PositionalEmbeddings
using Functors
export RoPE, AbsolutePE

"""
    compute_frequencies(dim::Int, seq_len::Int, base::Number=10_000)

Compute frequency bands for rotary position embeddings.

# Arguments
- `dim::Int`: Number of dimensions for the frequency bands
- `seq_len::Int`: Maximum sequence length to compute frequencies for
- `base::Number=10_000`: Base for geometric progression of frequencies

# Returns
- Matrix of shape (dim, seq_len) containing frequency values
"""
function compute_frequencies(dim::Int, seq_len::Int, base::Number=10_000)
    θ = 1 ./ (base .^ (collect(0:2:dim-1) ./ dim))
    positions = collect(0:seq_len-1)
    return θ * positions'
end

"""
    AbsolutePE{T<:AbstractArray}
    AbsolutePE(embedding_size::Int, max_length::Int; base::Number=10_000)

Absolute Position Embeddings using sinusoidal frequencies from "Attention Is All You Need" paper.
Formula: PE(pos,2i) = sin(pos/10000^(2i/d_model))
        PE(pos,2i+1) = cos(pos/10000^(2i/d_model))

# Fields
- `embedding_size::Int`: Size of the embedding dimension (d_model)
- `max_length::Int`: Maximum sequence length supported
- `embeddings::T`: Positional embeddings
"""
struct AbsolutePE{T<:AbstractArray}
    embedding_size::Int
    max_length::Int
    embeddings::T
end
Functors.@functor AbsolutePE

function AbsolutePE(embedding_size::Int, max_length::Int; base::Number=10_000)
    freqs = compute_frequencies(embedding_size, max_length, base)
    embeddings = zeros(Float32, embedding_size, max_length)
    embeddings[1:2:end, :] .= sin.(freqs)
    embeddings[2:2:end, :] .= cos.(freqs)
    AbsolutePE(embedding_size, max_length, embeddings)
end

function (layer::AbsolutePE)(x::AbstractArray)
    seq_len, channels, batch_size = size(x)
    @assert channels == layer.embedding_size "Channel dimension must match embedding size"
    @assert seq_len <= layer.max_length "Sequence length exceeds maximum length"
    pos_embeddings = view(layer.embeddings, :, 1:seq_len)
    embeddings_broadcast = reshape(permutedims(pos_embeddings, (2, 1)), seq_len, channels, 1)
    return x .+ embeddings_broadcast
end

"""
    RoPE(head_size::Int, seq_len::Int;
        base::Number=10_000,
        scale::Number=1.0)

Rotary Position Embeddings (RoPE) implementation as described in the paper
"RoFormer: Enhanced Transformer with Rotary Position Embedding".

Construct a RoPE object with the following arguments:
- `head_size::Int`: Head size to apply rotation to (must be multiple of 2)
- `seq_len::Int`: Maximum sequence length to support
- `base::Number=10_000`: Base for geometric progression of frequencies
- `scale::Number=1.0`: Scaling factor for the frequencies

# Examples
```julia
# Create RoPE for a model with 512 head size and max sequence length of 1024
rope = RoPE(512, 1024)

# Apply RoPE to input tensor of shape (head_size, seq_len, nheads*batch_size)
Q = randn(Float32, 512, 100, 32)
Q_positioned = rope(x)
```
"""
struct RoPE{T<:AbstractFloat, A<:AbstractArray{T}}
    head_size::Int
    cos_cached::A
    sin_cached::A
end
Functors.@functor RoPE

function RoPE(head_size::Int, seq_len::Int;
    base::Number=10_000,
    scale::Number=1.0,
    T::Type=Float32)

    @assert head_size % 2 == 0 "Head dimension should be multiple of 2, got $head_size"

    # Shape: head_size × seq_len
    freqs = T.(compute_frequencies(head_size, seq_len, base) .* scale)
    freqs_extended = vcat(freqs, freqs)
    cos_cached = cos.(freqs_extended)
    sin_cached = sin.(freqs_extended)
    RoPE(head_size, cos_cached, sin_cached)
end

"""
    neg_half(x::AbstractArray, dim::Int=1)

Helper function that negates the second half of the array along dimension `dim`.
This implementatio uses half negative array instead of interleaving pairs, as in LlaMA
https://github.com/huggingface/transformers/issues/25199

# Arguments
- `x::AbstractArray`: Input array
- `dim::Int=1`: Dimension along which to perform the operati    on

# Returns
- Array with second half negated along specified dimension
"""
function neg_half(x::AbstractArray)
    d_2 = size(x, 1) ÷ 2
    return vcat(-view(x, d_2+1:size(x, 1), :, :, ),
                view(x, 1:d_2, :, :, ))
end

function (rope::RoPE)(x::AbstractArray)
    head_size, seq_len, combined = size(x)
    @assert head_size == rope.head_size "Head dimension must match, expected $(rope.head_size), got $head_size"
    @assert seq_len <= size(rope.cos_cached, 2) "Sequence length exceeds maximum length"

    # Create negated version for rotation
    x_neg = neg_half(x)

    # Broadcasting will naturally handle the combined dimension
    x_rotated = @. x * rope.cos_cached + x_neg * rope.sin_cached

    return x_rotated
end

end
