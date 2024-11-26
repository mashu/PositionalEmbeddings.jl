![PositionalEmbeddings.jl](docs/src/assets/logo.svg)

# PositionalEmbeddings.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mashu.github.io/PositionalEmbeddings.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mashu.github.io/PositionalEmbeddings.jl/dev/)
[![Build Status](https://github.com/mashu/PositionalEmbeddings.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mashu/PositionalEmbeddings.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/mashu/PositionalEmbeddings.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mashu/PositionalEmbeddings.jl)

A Julia package providing various positional embedding implementations for enriching sequence data with position information.

## Features

- **Absolute Positional Embeddings (AbsolutePE)**: Implementation of the sinusoidal position embeddings from [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **Rotary Position Embeddings (RoPE)**: Implementation of the rotary position embeddings from [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

## Installation

```julia
using Pkg
Pkg.add("PositionalEmbeddings")
```

## Quick Start

```julia
using PositionalEmbeddings

# Absolute Positional Embeddings
pe = AbsolutePE(512, 1024)  # embedding_size=512, max_length=1024
x = randn(Float32, 100, 512, 32)  # (seq_len, channels, batch)
x_with_pos = pe(x)

# Rotary Position Embeddings
rope = RoPE(512, 1024)  # head_dim=512, max_length=1024
x = randn(Float32, 512, 2, 100, 32)  # (head_dim, heads, seq_len, batch)
x_with_pos = rope(x)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
