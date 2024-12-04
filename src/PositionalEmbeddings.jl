module PositionalEmbeddings
using Functors
using LinearAlgebra
export RoPE, AbsolutePE
export create_causal_mask, create_padding_mask

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

"""
    (rope::RoPE)(x) -> AbstractArray

Apply Rotary Position Embeddings to the input array `x` of shape `(head_size, seq_len, batch * num_heads)`.

# Arguments
- `x`: Input array where first dimension must match `rope.head_size` and second dimension must not exceed
       the maximum cached sequence length.

See also: [`RoPE`](@ref)
"""
function (rope::RoPE)(x::AbstractArray)
    head_size, seq_len, combined = size(x)
    @assert head_size == rope.head_size "Head dimension must match, expected $(rope.head_size), got $head_size"
    @assert seq_len <= size(rope.cos_cached, 2) "Sequence length exceeds maximum length"

    # Create negated version for rotation
    x_neg = neg_half(x)

    # Use views to slice cached values up to seq_len
    cos_cache = @view rope.cos_cached[:, 1:seq_len]
    sin_cache = @view rope.sin_cached[:, 1:seq_len]

    # Broadcasting will naturally handle the combined dimension
    x_rotated = @. x * cos_cache + x_neg * sin_cache

    return x_rotated
end

"""
    create_causal_mask(seq_len::Int)

Create a causal (autoregressive) attention mask that prevents positions from attending to future positions.
This is commonly used in language models to ensure predictions only depend on previous tokens.

The mask ensures that position i can only attend to positions j ≤ i, creating a triangular pattern
where the upper triangle including diagonal is masked (True) and the lower triangle is unmasked (False).

# Arguments
- `seq_len::Int`: Length of the sequence to create mask for

# Returns
- 3D boolean array of shape (seq_len, seq_len, 1) where True indicates positions to mask

# Examples
```
julia> mask = create_causal_mask(3)[:,:,1]
3×3 Matrix{Bool}:
 1  1  1  # First position can't attend anything
 0  1  1  # Second position can attend to first only
 0  0  1  # Third position can attend to first and second
```
"""
function causal_mask(seq_len::Int)
    return reshape(triu(trues(seq_len, seq_len), 0), seq_len, seq_len, 1)
end

"""
    create_padding_mask(lengths::Vector{Int}, max_len::Int)

Create padding masks for batched sequences of varying lengths. This ensures that padded positions
(positions beyond each sequence's actual length) are masked out and don't participate in attention.

# Arguments
- `lengths::Vector{Int}`: Actual length of each sequence in the batch
- `max_len::Int`: Maximum sequence length (padded length)

# Returns
- 3D boolean array of shape (batch_size, max_len, 1) where True indicates padded positions

# Examples
```
# For 2 sequences of lengths 2 and 3, padded to length 4:
julia> mask = create_padding_mask([2, 3], 4)[:,:,1]
2×4 Matrix{Bool}:
 0  0  1  1  # First sequence: length 2, positions 3-4 are padding
 0  0  0  1  # Second sequence: length 3, position 4 is padding
```

# Usage with Causal Mask
Padding and causal masks are often combined for batched autoregressive tasks:

```
seq_len = 5
batch_lengths = [3, 4]

# Create both masks
causal = create_causal_mask(seq_len)                # Shape: (5, 5, 1)
padding = create_padding_mask(batch_lengths, seq_len) # Shape: (2, 5, 1)

# Combine masks which will either prevent attending to future tokens or padding tokens
combined = causal .| padding

# final_mask will prevent:
# 1. Attending to future tokens (from causal mask)
# 2. Attending to padding tokens (from padding mask)
```
"""
function create_padding_mask(lengths::Vector{Int}, max_len::Int)
    return reshape(.!(lengths' .>= (1:max_len)), :, max_len, length(lengths))
end

end
