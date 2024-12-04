```@meta
CurrentModule = PositionalEmbeddings
```

# PositionalEmbeddings

Documentation for [PositionalEmbeddings](https://github.com/mashu/PositionalEmbeddings.jl).

The PositionalEmbeddings package provides implementations of positional embeddings for encoding sequential position information into feature vectors. This encoding is essential for models where the order of sequence elements must be preserved during processing.

The package implements two foundational approaches to positional encoding:

- Rotary Position Embeddings (RoPE) encode positions by rotating vectors in 2D subspaces, enabling explicit relative position modeling through geometric transformations.

- Absolute Positional Embeddings (AbsolutePE) create unique position markers using sinusoidal functions, following the original approach from "Attention Is All You Need."

## API Reference

```@autodocs
Modules = [PositionalEmbeddings]
```

## Usage Examples

```julia
# Create RoPE for head dimension of 64 and maximum sequence length of 1024
rope = RoPE(64, 1024)

# Apply to input tensor of shape (head_size, seq_len, nheads*batch)
# For example, with 64-dim heads, sequence length 100, 8 heads × 32 batch size:
x = randn(Float32, 64, 100, 256)  # 256 = 8 heads × 32 batch
x_with_positions = rope(x)
```

Input tensors for RoPE must follow the shape (head_size, seq_len, nheads*batch). The head_size parameter must be even, seq_len represents your sequence length, and the final dimension combines the number of attention heads and batch size.

The RoPE constructor accepts several parameters:

```julia
function RoPE(head_size::Int, seq_len::Int;
    base::Number=10_000,
    scale::Number=1.0,
    T::Type=Float32)
```

The base parameter controls frequency bands for position encoding, with higher values creating slower-changing position representations. The scale parameter allows adjusting the positional encoding's influence.


### Absolute Positional Embeddings

AbsolutePE implements fixed positional patterns through sinusoidal encoding:

```julia
# Create embeddings for 512-dimensional features up to length 1000
pe = AbsolutePE(512, 1000)

# Apply to input tensor of shape (seq_len, features, batch)
x = randn(Float32, 100, 512, 32)
x_with_positions = pe(x)
```

For AbsolutePE, tensors require the shape (seq_len, features, batch), where features matches your model's dimension and seq_len represents the sequence length.

The AbsolutePE constructor allows customization through:

```julia
function AbsolutePE(embedding_size::Int, max_length::Int; base::Number=10_000)
```
![AbsolutePE](assets/AbsolutePE-128-100.svg)

The base parameter influences the wavelength pattern of sinusoidal embeddings, with each dimension using a different frequency derived from this base value.

## Flux Integration Example

This example that adds `RoPERoPEMultiHeadAttention` that. Here's the complete implementation:

```julia
using PositionalEmbeddings
using LinearAlgebra
using WeightInitializers
using Functors
using NNlib

struct RoPEMultiHeadAttention{T<:AbstractFloat, A<:AbstractArray{T, 2}}
    Wq::A
    Wk::A
    Wv::A
    Wo::A
    num_heads::Int
    head_dim::Int
    scale::T
    rope::RoPE
end

function RoPEMultiHeadAttention(d_model::Int, num_heads::Int; maxlen=1000)
    head_dim = d_model ÷ num_heads
    @assert head_dim * num_heads == d_model "d_model ($d_model) must be divisible by num_heads ($num_heads)"

    Wq = kaiming_normal(d_model, d_model)
    Wk = kaiming_normal(d_model, d_model)
    Wv = kaiming_normal(d_model, d_model)
    Wo = kaiming_normal(d_model, d_model)

    scale = Float32(sqrt(head_dim))
    rope = RoPE(head_dim, maxlen)

    RoPEMultiHeadAttention(Wq, Wk, Wv, Wo, num_heads, head_dim, scale, rope)
end

# Split: (d_model, seqlen, batch) -> (head_dim, seqlen, num_heads * batch)
function split_heads(x::AbstractArray, head_dim::Int, num_heads::Int)
    d_model, seqlen, batch = size(x)
    return reshape(permutedims(reshape(x, head_dim, num_heads, seqlen, batch), (1, 3, 2, 4)),
                head_dim, seqlen, num_heads * batch)
end

# Join: (head_dim, seqlen, num_heads * batch) -> (d_model, seqlen, batch)
function join_heads(x::AbstractArray, head_dim::Int, num_heads::Int, batch_size::Int)
    return reshape(permutedims(reshape(x, head_dim, size(x, 2), num_heads, batch_size), (1, 3, 2, 4)),
                head_dim * num_heads, size(x, 2), batch_size)
end

function apply_mask(logits, mask)
    neginf = typemin(eltype(logits))
    ifelse.(mask, logits, neginf)
end

function (mha::RoPEMultiHeadAttention)(x::AbstractArray, mask=nothing)
    d_model, seqlen, batch_size = size(x)

    # Project and split heads in one go
    q = split_heads(reshape(mha.Wq * reshape(x, d_model, :), d_model, seqlen, batch_size),
                mha.head_dim, mha.num_heads)
    k = split_heads(reshape(mha.Wk * reshape(x, d_model, :), d_model, seqlen, batch_size),
                mha.head_dim, mha.num_heads)
    v = split_heads(reshape(mha.Wv * reshape(x, d_model, :), d_model, seqlen, batch_size),
                mha.head_dim, mha.num_heads)

    # Apply RoPE
    q = mha.rope(q)
    k = mha.rope(k)

    # All operations now work with (head_dim, seqlen, num_heads * batch)
    attention_scores = NNlib.batched_mul(NNlib.batched_transpose(k), (q ./ mha.scale))

    if !isnothing(mask)
        neginf = typemin(eltype(attention_scores))
        attention_scores = ifelse.(mask, attention_scores, neginf)
    end

    attention_probs = softmax(attention_scores; dims=1)
    attention_output = NNlib.batched_mul(v, attention_probs)

    # Join heads only at the very end
    output = join_heads(attention_output, mha.head_dim, mha.num_heads, batch_size)
    return reshape(mha.Wo * reshape(output, d_model, :), d_model, seqlen, batch_size)
end
Functors.@functor RoPEMultiHeadAttention
x = rand(512, 20, 32);
mha = RoPEMultiHeadAttention(512, 8)
mha(x)
```

```@index
```
