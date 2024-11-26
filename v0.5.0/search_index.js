var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = PositionalEmbeddings","category":"page"},{"location":"#PositionalEmbeddings","page":"Home","title":"PositionalEmbeddings","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for PositionalEmbeddings.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package provides various implementations of positional embeddings - techniques for encoding position information into feature vectors. These can be used in any architecture where sequence order or position needs to be represented in the model's computations.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Currently implemented:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Rotary Position Embeddings (RoPE) - A method that encodes positions by rotating vectors in 2D subspaces\nPositional Embeddings (AbsolutePE) - The original positional encoding scheme from \"Attention Is All You Need\"","category":"page"},{"location":"#API-Reference","page":"Home","title":"API Reference","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [PositionalEmbeddings]","category":"page"},{"location":"#PositionalEmbeddings.AbsolutePE","page":"Home","title":"PositionalEmbeddings.AbsolutePE","text":"AbsolutePE{T<:AbstractArray}\n\nAbsolute Position Embeddings using sinusoidal frequencies from \"Attention Is All You Need\" paper. Formula: PE(pos,2i) = sin(pos/10000^(2i/dmodel))         PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))\n\nFields\n\nembedding_size::Int: Size of the embedding dimension (d_model)\nmax_length::Int: Maximum sequence length supported\nembeddings::T: Pre-computed positional embeddings\n\n\n\n\n\n","category":"type"},{"location":"#PositionalEmbeddings.RoPE","page":"Home","title":"PositionalEmbeddings.RoPE","text":"RoPE(features::Int, seq_len::Int;\n    base::Number=10_000,\n    scale::Number=1.0,\n    T::Type=Float32)\n\nRotary Position Embeddings (RoPE) implementation as described in the paper \"RoFormer: Enhanced Transformer with Rotary Position Embedding\".\n\nConstruct a RoPE object with the following arguments:\n\nfeatures::Int: Number of features to apply rotation to (must be multiple of 8)\nseq_len::Int: Maximum sequence length to support\nbase::Number=10_000: Base for geometric progression of frequencies\nscale::Number=1.0: Scaling factor for the rotations\nT::Type=Float32: Data type for the embeddings\n\nExamples\n\n# Create RoPE for a model with 512 features and max sequence length of 1024\nrope = RoPE(512, 1024)\n\n# Apply RoPE to input tensor of shape (features, seq_len, batch)\nQ = randn(Float32, 512, 100, 32)\nQ_positioned = rope(x)\n\n\n\n\n\n","category":"type"},{"location":"#PositionalEmbeddings.compute_frequencies","page":"Home","title":"PositionalEmbeddings.compute_frequencies","text":"compute_frequencies(dim::Int, seq_len::Int, base::Number=10_000)\n\nCompute frequency bands for rotary position embeddings.\n\nArguments\n\ndim::Int: Number of dimensions for the frequency bands\nseq_len::Int: Maximum sequence length to compute frequencies for\nbase::Number=10_000: Base for geometric progression of frequencies\n\nReturns\n\nMatrix of shape (dim, seq_len) containing frequency values\n\n\n\n\n\n","category":"function"},{"location":"#PositionalEmbeddings.neg_half-Tuple{AbstractArray}","page":"Home","title":"PositionalEmbeddings.neg_half","text":"neg_half(x::AbstractArray, dim::Int=1)\n\nHelper function that negates the second half of the array along dimension dim. This implementatio uses half negative array instead of interleaving pairs, as in LlaMA https://github.com/huggingface/transformers/issues/25199\n\nArguments\n\nx::AbstractArray: Input array\ndim::Int=1: Dimension along which to perform the operati    on\n\nReturns\n\nArray with second half negated along specified dimension\n\n\n\n\n\n","category":"method"},{"location":"#Usage-Examples","page":"Home","title":"Usage Examples","text":"","category":"section"},{"location":"#Rotary-Position-Embeddings-(RoPE)","page":"Home","title":"Rotary Position Embeddings (RoPE)","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"RoPE encodes positional information by applying rotations to feature vectors:","category":"page"},{"location":"","page":"Home","title":"Home","text":"# Create RoPE for head_dim of dimension 512 and maximum sequence length of 1024\nrope = RoPE(512, 1024)\n\n# Apply to any feature tensor of shape (head_dim, n_heads,sequence_length, batch)\nfeatures = randn(Float32, 512, 2, 100, 32)\nfeatures_with_pos = rope(features)","category":"page"},{"location":"#Basic-AbsolutePE-Usage","page":"Home","title":"Basic AbsolutePE Usage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"# Create embeddings for 128-dimensional features up to length 100\npe = AbsolutePE(128, 100)\n\n# Apply to input tensor of shape (features, seq_len, batch)\nx = randn(Float32, 128, 100, 32)\nx_positioned = pe(x)","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: AbsolutePE)","category":"page"},{"location":"#Rotary-Position-Embeddings-(RoPE)-with-MultiHeadAttention-from-Flux","page":"Home","title":"Rotary Position Embeddings (RoPE) with MultiHeadAttention from Flux","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This example can add RoPEMultiHeadAttention that with Rotary Position Embeddings to Flux's MultiHeadAttention. Here's the complete implementation:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Flux: MultiHeadAttention\nusing NNlib: dot_product_attention\n\nstruct RoPEMultiHeadAttention{M<:MultiHeadAttention, F<:Real}\n    mha::M\n    rope::RoPE\n    rope_fraction::F\nend\n\nFlux.@functor RoPEMultiHeadAttention\n\n# Constructor\nfunction RoPEMultiHeadAttention(args...;\n                             max_seq_len::Int=2048,\n                             rope_fraction::Real=1.0,\n                             rope_scale::Real=1.0,\n                             T::Type=Float32,\n                             kwargs...)\n    # Create base MultiHeadAttention\n    mha = MultiHeadAttention(args...; kwargs...)\n\n    # Calculate number of features to rotate based on fraction\n    d = size(mha.out_proj.weight, 1)\n    features_to_rotate = floor(Int, d * rope_fraction)\n\n    # Ensure feature count is valid\n    @assert features_to_rotate % 8 == 0 \"Number of features to rotate should be multiple of 8 for optimal performance, got $features_to_rotate. Adjust rope_fraction accordingly.\"\n\n    # Create RoPE layer\n    rope = RoPE(features_to_rotate, max_seq_len; scale=rope_scale, T=T)\n\n    RoPEMultiHeadAttention(mha, rope, T(rope_fraction))\nend\n\n# Only make MHA parameters trainable, not RoPE\nFlux.trainable(rmha::RoPEMultiHeadAttention) = (; mha=rmha.mha)\n\n# Forward pass for separate q, k, v\nfunction (rmha::RoPEMultiHeadAttention)(q_in::A3, k_in::A3, v_in::A3, bias=nothing; mask=nothing)\n    mha = rmha.mha\n\n    # Project inputs\n    q = mha.q_proj(q_in)\n    k = mha.k_proj(k_in)\n    v = mha.v_proj(v_in)\n\n    # Apply RoPE to Q and K (TODO)\n\n    # Compute attention\n    x, α = NNlib.dot_product_attention(q, k, v, bias;\n                                     mha.nheads, mask, fdrop=mha.attn_drop)\n\n    # Project output\n    return mha.out_proj(x), α\nend\n\n# Self-attention case\n(rmha::RoPEMultiHeadAttention)(x::A3, bias=nothing; mask=nothing) = rmha(x, x, x, bias; mask=mask)","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}