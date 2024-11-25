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

Absolute Position Embeddings using sinusoidal frequencies from "Attention Is All You Need" paper.
Formula: PE(pos,2i) = sin(pos/10000^(2i/d_model))
        PE(pos,2i+1) = cos(pos/10000^(2i/d_model))

# Fields
- `embedding_size::Int`: Size of the embedding dimension (d_model)
- `max_length::Int`: Maximum sequence length supported
- `embeddings::T`: Pre-computed positional embeddings
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
    RoPE{T, A<:AbstractArray{T}}
    RoPE(features::Int, seq_len::Int;
        base::Number=10_000,
        scale::Number=1.0,
        T::Type=Float32)

Rotary Position Embeddings (RoPE) implementation as described in the paper
"RoFormer: Enhanced Transformer with Rotary Position Embedding".

Construct a RoPE object with the following arguments:
- `features::Int`: Number of features to apply rotation to (must be multiple of 8)
- `seq_len::Int`: Maximum sequence length to support
- `base::Number=10_000`: Base for geometric progression of frequencies
- `scale::Number=1.0`: Scaling factor for the rotations
- `T::Type=Float32`: Data type for the embeddings

# Examples
```jldoctest
# Create RoPE for a model with 512 features and max sequence length of 1024
rope = RoPE(512, 1024)

# Apply RoPE to input tensor of shape (features, seq_len, batch)
Q = randn(Float32, 512, 100, 32)
Q_positioned = rope(x)
```
"""
struct RoPE{T, A<:AbstractArray{T}}
    features::Int
    cos_cached::A
    sin_cached::A
    scale::T
end
Functors.@functor RoPE

function RoPE(features::Int, seq_len::Int;
            base::Number=10_000,
            scale::Number=1.0,
            T::Type=Float32)

    @assert features % 8 == 0 "Number of features should be multiple of 8 for optimal performance, got $features"

    freqs = T.(compute_frequencies(features, seq_len, base))
    freqs_extended = vcat(freqs, freqs)

    cos_cached = cos.(freqs_extended)
    sin_cached = sin.(freqs_extended)
    cos_cached = reshape(cos_cached, size(cos_cached,1), size(cos_cached,2), 1)
    sin_cached = reshape(sin_cached, size(sin_cached,1), size(sin_cached,2), 1)

    RoPE(features, cos_cached, sin_cached, T(scale))
end

"""
    neg_half(x::AbstractArray{T}, dim::Int=1) where T

Helper function that negates the second half of the array along dimension `dim`.
This implementatio uses half negative array instead of interleaving pairs, as in LlaMA
https://github.com/huggingface/transformers/issues/25199

# Arguments
- `x::AbstractArray{T}`: Input array
- `dim::Int=1`: Dimension along which to perform the operati    on

# Returns
- Array with second half negated along specified dimension
"""
function neg_half(x::AbstractArray{T}, dim::Int=1) where T
    d_2 = size(x, dim) ÷ 2
    return vcat(view(x, d_2+1:size(x,dim), :, :) .* -1,
                view(x, 1:d_2, :, :))
end

function (rope::RoPE)(x::AbstractArray{T}) where T
    features_to_rotate = min(rope.features, size(x, 1))

    x_rope = view(x, 1:features_to_rotate, :, :)
    x_pass = view(x, features_to_rotate+1:size(x,1), :, :)

    x_neg = neg_half(x_rope)
    cos_mat = view(rope.cos_cached, 1:size(x_rope,1), 1:size(x_rope,2), :)
    sin_mat = view(rope.sin_cached, 1:size(x_rope,1), 1:size(x_rope,2), :)

    x_rotated = @. muladd(x_rope * rope.scale, cos_mat, x_neg * rope.scale * sin_mat)

    return vcat(x_rotated, x_pass)
end

end
