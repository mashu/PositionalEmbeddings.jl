```@meta
CurrentModule = PositionalEmbeddings
```

# PositionalEmbeddings

Documentation for [PositionalEmbeddings](https://github.com/mashu/PositionalEmbeddings.jl).

This package provides various implementations of positional embeddings - techniques for encoding position information into feature vectors. These can be used in any architecture where sequence order or position needs to be represented in the model's computations.

Currently implemented:
- Rotary Position Embeddings (RoPE) - A method that encodes positions by rotating vectors in 2D subspaces
- Positional Embeddings (AbsolutePE) - The original positional encoding scheme from "Attention Is All You Need"

## API Reference

```@autodocs
Modules = [PositionalEmbeddings]
```

## Usage Examples

### Rotary Position Embeddings (RoPE)

RoPE encodes positional information by applying rotations to feature vectors:

```julia
# Create RoPE for head_dim of dimension 512 and maximum sequence length of 1024
rope = RoPE(512, 1024)

# Apply to any feature tensor of shape (head_dim, n_heads,sequence_length, batch)
features = randn(Float32, 512, 2, 100, 32)
features_with_pos = rope(features)
```

### Basic AbsolutePE Usage

```julia
# Create embeddings for 128-dimensional features up to length 100
pe = AbsolutePE(128, 100)

# Apply to input tensor of shape (features, seq_len, batch)
x = randn(Float32, 128, 100, 32)
x_positioned = pe(x)
```
![AbsolutePE](assets/AbsolutePE-128-100.svg)

## Rotary Position Embeddings (RoPE) with MultiHeadAttention from Flux

This example can add `RoPEMultiHeadAttention` that with Rotary Position Embeddings to Flux's MultiHeadAttention. Here's the complete implementation:

```julia
using Flux: MultiHeadAttention
using NNlib: dot_product_attention

struct RoPEMultiHeadAttention{M<:MultiHeadAttention, F<:Real}
    mha::M
    rope::RoPE
    rope_fraction::F
end

Flux.@functor RoPEMultiHeadAttention

# Constructor
function RoPEMultiHeadAttention(args...;
                             max_seq_len::Int=2048,
                             rope_fraction::Real=1.0,
                             rope_scale::Real=1.0,
                             T::Type=Float32,
                             kwargs...)
    # Create base MultiHeadAttention
    mha = MultiHeadAttention(args...; kwargs...)

    # Calculate number of features to rotate based on fraction
    input_dim = size(mha.out_proj.weight, 1)
    d = input_dim ÷ mha.nheads
    features_to_rotate = floor(Int, d * rope_fraction)

    # Ensure feature count is valid
    @assert features_to_rotate % 8 == 0 "Number of features to rotate should be multiple of 8 for optimal performance, got $features_to_rotate. Adjust rope_fraction accordingly."

    # Create RoPE layer
    rope = RoPE(features_to_rotate, max_seq_len; scale=rope_scale, T=T)

    RoPEMultiHeadAttention(mha, rope, T(rope_fraction))
end

# Only make MHA parameters trainable, not RoPE
Flux.trainable(rmha::RoPEMultiHeadAttention) = (; mha=rmha.mha)

split_heads(x, nheads) = reshape(x, size(x, 1) ÷ nheads, nheads, size(x)[2:end]...)
join_heads(x) = reshape(x, :, size(x)[3:end]..

# Forward pass for separate q, k, v
function (rmha::RoPEMultiHeadAttention)(q_in::A3, k_in::A3, v_in::A3, bias=nothing; mask=nothing)
    mha = rmha.mha

    # Project inputs
    q = mha.q_proj(q_in)
    k = mha.k_proj(k_in)
    v = mha.v_proj(v_in)

    # Apply RoPE to Q and K
    q_rot = join_heads(rmha.rope(split_heads(q, mha.nheads)))
    k_rot = join_heads(rmha.rope(split_heads(k, mha.nheads)))

    # Compute attention
    x, α = NNlib.dot_product_attention(q_rot, k_rot, v, bias;
                                     mha.nheads, mask, fdrop=mha.attn_drop)

    # Project output
    return mha.out_proj(x), α
end

# Self-attention case
(rmha::RoPEMultiHeadAttention)(x::A3, bias=nothing; mask=nothing) = rmha(x, x, x, bias; mask=mask)
```

```@index
```
