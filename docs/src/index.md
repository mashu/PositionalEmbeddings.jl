```@meta
CurrentModule = PositionalEmbeddings
```

# PositionalEmbeddings

Documentation for [PositionalEmbeddings](https://github.com/mashu/PositionalEmbeddings.jl).

This package provides various implementations of positional embeddings - techniques for encoding position information into feature vectors. These can be used in any architecture where sequence order or position needs to be represented in the model's computations.

Currently implemented:
- Rotary Position Embeddings (RoPE) - A method that encodes positions by rotating vectors in 2D subspaces
- Frequency Positional Embeddings (FrequencyPE) - The original positional encoding scheme from "Attention Is All You Need"

## API Reference

```@autodocs
Modules = [PositionalEmbeddings]
```

## Usage Examples

### Rotary Position Embeddings (RoPE)

RoPE encodes positional information by applying rotations to feature vectors:

```julia
# Create RoPE for features of dimension 512 and maximum sequence length of 1024
rope = RoPE(512, 1024)

# Apply to any feature tensor of shape (features, sequence_length, batch)
features = randn(Float32, 512, 100, 32)
features_with_pos = rope(features)
```

### Basic FrequencyPE Usage

```julia
# Create embeddings for 512-dimensional features up to length 1024
pe = FrequencyPE(512, 1024)

# Apply to input tensor of shape (features, seq_len, batch)
x = randn(Float32, 512, 100, 32)
x_positioned = pe(x)
```

### Example with Query/Key Matrices

Here's a complete example showing how to use RoPE with attention mechanisms:

```julia
using Flux: MultiHeadAttention
using PositionalEmbeddings: RoPE, RoPEMultiHeadAttention
using NNlib: dot_product_attention

# Initialize RoPE with specific feature dimensions to rotate
dim = 64           # embedding dimension
nheads = 4         # number of attention heads
max_seq_len = 2048
seq_len = 100
batch_size = 32

# Create attention layer with RoPE
rmha = RoPEMultiHeadAttention(
    dim,           # embedding dimension
    nheads;        # number of heads
    max_seq_len = max_seq_len,
    rope_fraction = 1.0    # apply RoPE to all features
)

# Sample input tensors (dim, seq_len, batch)
q_in = randn(Float32, dim, seq_len, batch_size)
k_in = randn(Float32, dim, seq_len, batch_size)
v_in = randn(Float32, dim, seq_len, batch_size)

# The forward pass will:
# 1. Project inputs
mha = rmha.mha
q = mha.q_proj(q_in)
k = mha.k_proj(k_in)
v = mha.v_proj(v_in)

# 2. Apply RoPE to queries and keys
q = rmha.rope(q)
k = rmha.rope(k)

# 3. Compute attention
bias = nothing
mask = nothing
x, α = dot_product_attention(q, k, v, bias;
                           nheads=mha.nheads,
                           mask=mask,
                           fdrop=mha.attn_drop)

# 4. Project output
output = mha.out_proj(x)

# Alternatively, use the provided wrapper:
output, attention_weights = rmha(q_in, k_in, v_in)

# Or for self-attention:
output, attention_weights = rmha(q_in)
```

## Flux Integration

The package provides a wrapper type `RoPEMultiHeadAttention` that adds Rotary Position Embeddings to Flux's MultiHeadAttention. Here's the complete implementation:

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
    d = size(mha.out_proj.weight, 1)
    features_to_rotate = floor(Int, d * rope_fraction)

    # Ensure feature count is valid
    @assert features_to_rotate % 8 == 0 "Number of features to rotate should be multiple of 8 for optimal performance, got $features_to_rotate. Adjust rope_fraction accordingly."

    # Create RoPE layer
    rope = RoPE(features_to_rotate, max_seq_len; scale=rope_scale, T=T)

    RoPEMultiHeadAttention(mha, rope, T(rope_fraction))
end

# Only make MHA parameters trainable, not RoPE
Flux.trainable(rmha::RoPEMultiHeadAttention) = (; mha=rmha.mha)

# Forward pass for separate q, k, v
function (rmha::RoPEMultiHeadAttention)(q_in::A3, k_in::A3, v_in::A3, bias=nothing; mask=nothing)
    mha = rmha.mha

    # Project inputs
    q = mha.q_proj(q_in)
    k = mha.k_proj(k_in)
    v = mha.v_proj(v_in)

    # Apply RoPE
    q = rmha.rope(q)
    k = rmha.rope(k)

    # Compute attention
    x, α = NNlib.dot_product_attention(q, k, v, bias;
                                     mha.nheads, mask, fdrop=mha.attn_drop)

    # Project output
    return mha.out_proj(x), α
end

# Self-attention case
(rmha::RoPEMultiHeadAttention)(x::A3, bias=nothing; mask=nothing) = rmha(x, x, x, bias; mask=mask)
```

```@index
```
