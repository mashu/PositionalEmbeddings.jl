using Test
using CUDA
using Zygote
using PositionalEmbeddings

function has_working_cuda()
    return CUDA.has_cuda() && CUDA.functional()
end

# # https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/rope/__init__.py
# # Code from the above link was used to generate the test data
# import torch
# import numpy as np

# base = 10000
# d = 16
# seq_len = 10  # Changed from 16 to 10

# # Calculate frequencies
# theta = 1. / (base ** (torch.arange(0, d, 2).float() / d))
# seq_idx = torch.arange(seq_len).float()  # Now 0-9 instead of 0-15
# idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
# idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

# # Cache sin/cos values - now 10x1x1x16 instead of 16x1x1x16
# cos_cached = idx_theta2.cos()[:, None, None, :]
# sin_cached = idx_theta2.sin()[:, None, None, :]

# # Input will be 16x10x1 instead of 16x16x1
# input_original = torch.arange(1, 16*10 + 1, dtype=torch.float32).reshape(16, 10, 1)
# x = input_original.permute(1, 2, 0).unsqueeze(1)  # [10, 1, 1, 16]

# # Split head_dim
# d_2 = d // 2
# x_rope, x_pass = x[..., :d], x[..., d:]

# # Calculate neg_half
# neg_half_x = torch.cat([-x_rope[:, :, :, d_2:], x_rope[:, :, :, :d_2]], dim=-1)

# # Final output
# final_output = (x_rope * cos_cached[:x.shape[0]]) + (neg_half_x * sin_cached[:x.shape[0]])

# # Save results
# np.savez('rope_test_data.npz',
#         input_original=input_original,
#         input_permuted=x.numpy(),
#         theta=theta.numpy(),
#         idx_theta=idx_theta.numpy(),
#         idx_theta2=idx_theta2.numpy(),
#         cos_cached=cos_cached.numpy(),
#         sin_cached=sin_cached.numpy(),
#         neg_half=neg_half_x.numpy(),
#         final_output=final_output.numpy())

@testset "RoPE Tests" begin
    @testset "Cached Values Test" begin
        head_dim, seq_len = 16, 10
        rope = RoPE(head_dim, seq_len)

        expected_cos = reshape([
            1.0 0.540302 -0.416147 -0.989992 -0.653644 0.283662 0.96017 0.753902 -0.1455 -0.91113;
            1.0 0.950415 0.806578 0.582754 0.301137 -0.0103423 -0.320796 -0.599437 -0.818632 -0.956644;
            1.0 0.995004 0.980067 0.955337 0.921061 0.877583 0.825336 0.764842 0.696707 0.62161;
            1.0 0.9995 0.998001 0.995503 0.992011 0.987526 0.982054 0.9756 0.96817 0.959773;
            1.0 0.99995 0.9998 0.99955 0.9992 0.99875 0.998201 0.997551 0.996802 0.995953;
            1.0 0.999995 0.99998 0.999955 0.99992 0.999875 0.99982 0.999755 0.99968 0.999595;
            1.0 1.0 0.999998 0.999996 0.999992 0.999987 0.999982 0.999976 0.999968 0.99996;
            1.0 1.0 1.0 1.0 0.999999 0.999999 0.999998 0.999998 0.999997 0.999996;
            1.0 0.540302 -0.416147 -0.989992 -0.653644 0.283662 0.96017 0.753902 -0.1455 -0.91113;
            1.0 0.950415 0.806578 0.582754 0.301137 -0.0103423 -0.320796 -0.599437 -0.818632 -0.956644;
            1.0 0.995004 0.980067 0.955337 0.921061 0.877583 0.825336 0.764842 0.696707 0.62161;
            1.0 0.9995 0.998001 0.995503 0.992011 0.987526 0.982054 0.9756 0.96817 0.959773;
            1.0 0.99995 0.9998 0.99955 0.9992 0.99875 0.998201 0.997551 0.996802 0.995953;
            1.0 0.999995 0.99998 0.999955 0.99992 0.999875 0.99982 0.999755 0.99968 0.999595;
            1.0 1.0 0.999998 0.999996 0.999992 0.999987 0.999982 0.999976 0.999968 0.99996;
            1.0 1.0 1.0 1.0 0.999999 0.999999 0.999998 0.999998 0.999997 0.999996
        ], (16,1,10,1))

        expected_sin = reshape([
            0.0 0.841471 0.909297 0.14112 -0.756802 -0.958924 -0.279415 0.656987 0.989358 0.412118;
            0.0 0.310984 0.591127 0.812649 0.953581 0.999947 0.947148 0.800422 0.574318 0.291259;
            0.0 0.0998334 0.198669 0.29552 0.389418 0.479426 0.564642 0.644218 0.717356 0.783327;
            0.0 0.0316175 0.0632034 0.0947261 0.126154 0.157456 0.1886 0.219556 0.250292 0.280778;
            0.0 0.00999983 0.0199987 0.0299955 0.0399893 0.0499792 0.059964 0.0699428 0.0799147 0.0898785;
            0.0 0.00316227 0.00632451 0.00948669 0.0126488 0.0158107 0.0189725 0.0221341 0.0252955 0.0284567;
            0.0 0.001 0.002 0.003 0.00399999 0.00499998 0.00599996 0.00699994 0.00799991 0.00899988;
            0.0 0.000316228 0.000632456 0.000948683 0.00126491 0.00158114 0.00189737 0.00221359 0.00252982 0.00284605;
            0.0 0.841471 0.909297 0.14112 -0.756802 -0.958924 -0.279415 0.656987 0.989358 0.412118;
            0.0 0.310984 0.591127 0.812649 0.953581 0.999947 0.947148 0.800422 0.574318 0.291259;
            0.0 0.0998334 0.198669 0.29552 0.389418 0.479426 0.564642 0.644218 0.717356 0.783327;
            0.0 0.0316175 0.0632034 0.0947261 0.126154 0.157456 0.1886 0.219556 0.250292 0.280778;
            0.0 0.00999983 0.0199987 0.0299955 0.0399893 0.0499792 0.059964 0.0699428 0.0799147 0.0898785;
            0.0 0.00316227 0.00632451 0.00948669 0.0126488 0.0158107 0.0189725 0.0221341 0.0252955 0.0284567;
            0.0 0.001 0.002 0.003 0.00399999 0.00499998 0.00599996 0.00699994 0.00799991 0.00899988;
            0.0 0.000316228 0.000632456 0.000948683 0.00126491 0.00158114 0.00189737 0.00221359 0.00252982 0.00284605
        ], (16,1,10,1))

        @test isapprox(rope.cos_cached, expected_cos)
        @test isapprox(rope.sin_cached, expected_sin)
    end

    @testset "neg_half Function Test" begin
        x = reshape([
            1.0    2.0    3.0    4.0    5.0    6.0    7.0    8.0    9.0   10.0;
           11.0   12.0   13.0   14.0   15.0   16.0   17.0   18.0   19.0   20.0;
           21.0   22.0   23.0   24.0   25.0   26.0   27.0   28.0   29.0   30.0;
           31.0   32.0   33.0   34.0   35.0   36.0   37.0   38.0   39.0   40.0;
           41.0   42.0   43.0   44.0   45.0   46.0   47.0   48.0   49.0   50.0;
           51.0   52.0   53.0   54.0   55.0   56.0   57.0   58.0   59.0   60.0;
           61.0   62.0   63.0   64.0   65.0   66.0   67.0   68.0   69.0   70.0;
           71.0   72.0   73.0   74.0   75.0   76.0   77.0   78.0   79.0   80.0;
           81.0   82.0   83.0   84.0   85.0   86.0   87.0   88.0   89.0   90.0;
           91.0   92.0   93.0   94.0   95.0   96.0   97.0   98.0   99.0  100.0;
          101.0  102.0  103.0  104.0  105.0  106.0  107.0  108.0  109.0  110.0;
          111.0  112.0  113.0  114.0  115.0  116.0  117.0  118.0  119.0  120.0;
          121.0  122.0  123.0  124.0  125.0  126.0  127.0  128.0  129.0  130.0;
          131.0  132.0  133.0  134.0  135.0  136.0  137.0  138.0  139.0  140.0;
          141.0  142.0  143.0  144.0  145.0  146.0  147.0  148.0  149.0  150.0;
          151.0  152.0  153.0  154.0  155.0  156.0  157.0  158.0  159.0  160.0
         ], (16,1,10,1))

        expected_neg_half = reshape([
            -81.0   -82.0   -83.0   -84.0   -85.0   -86.0   -87.0   -88.0   -89.0   -90.0;
            -91.0   -92.0   -93.0   -94.0   -95.0   -96.0   -97.0   -98.0   -99.0  -100.0;
            -101.0  -102.0  -103.0  -104.0  -105.0  -106.0  -107.0  -108.0  -109.0  -110.0;
            -111.0  -112.0  -113.0  -114.0  -115.0  -116.0  -117.0  -118.0  -119.0  -120.0;
            -121.0  -122.0  -123.0  -124.0  -125.0  -126.0  -127.0  -128.0  -129.0  -130.0;
            -131.0  -132.0  -133.0  -134.0  -135.0  -136.0  -137.0  -138.0  -139.0  -140.0;
            -141.0  -142.0  -143.0  -144.0  -145.0  -146.0  -147.0  -148.0  -149.0  -150.0;
            -151.0  -152.0  -153.0  -154.0  -155.0  -156.0  -157.0  -158.0  -159.0  -160.0;
            1.0     2.0     3.0     4.0     5.0     6.0     7.0     8.0     9.0    10.0;
            11.0    12.0    13.0    14.0    15.0    16.0    17.0    18.0    19.0    20.0;
            21.0    22.0    23.0    24.0    25.0    26.0    27.0    28.0    29.0    30.0;
            31.0    32.0    33.0    34.0    35.0    36.0    37.0    38.0    39.0    40.0;
            41.0    42.0    43.0    44.0    45.0    46.0    47.0    48.0    49.0    50.0;
            51.0    52.0    53.0    54.0    55.0    56.0    57.0    58.0    59.0    60.0;
            61.0    62.0    63.0    64.0    65.0    66.0    67.0    68.0    69.0    70.0;
            71.0    72.0    73.0    74.0    75.0    76.0    77.0    78.0    79.0    80.0
        ], (16,1,10,1))

        @test isapprox(PositionalEmbeddings.neg_half(x), expected_neg_half)
    end

    @testset "Forward Pass Test" begin
        x = permutedims(reshape(Float32.(1:160), (16, 10, 1))[:,:,:,:],(1, 3, 2, 4))
        pe = RoPE(16, 10)

        # Manual calculation
        neg_half_x = PositionalEmbeddings.neg_half(x)
        seq_len = size(x, 3)
        head_dim = size(x, 1)
        cos_mat = view(pe.cos_cached, 1:head_dim, 1:1, 1:seq_len, :)
        sin_mat = view(pe.sin_cached, 1:head_dim, 1:1, 1:seq_len, :)
        expected_output = @. muladd(x, cos_mat, neg_half_x * sin_mat)

        # Test the forward pass
        actual_output = pe(x)
        @test isapprox(actual_output, expected_output)
    end

    @testset "Gradient Tests (CPU, Float64)" begin
        eps = 1e-8
        rope = RoPE(8, 4; T=Float64)
        x = randn(8, 1, 4, 1)
        x_orig = copy(x)

        loss(x) = sum(abs2, rope(x))

        y, pb = Zygote.pullback(loss, x)
        analytical_grad = pb(1.0)[1]

        numerical_grad = similar(x)
        for i in eachindex(x)
            x_plus = copy(x)
            x_minus = copy(x)
            x_plus[i] += eps
            x_minus[i] -= eps
            numerical_grad[i] = (loss(x_plus) - loss(x_minus)) / (2eps)
        end

        @test isapprox(analytical_grad, numerical_grad, atol=1e-4)
    end

    if has_working_cuda()
        @testset "GPU Tests" begin
            @testset "Gradient Computation (GPU, Float32)" begin
                rope_gpu = RoPE(8, 4; T=Float32)
                rope_gpu = RoPE(
                    rope_gpu.head_dim,
                    cu(rope_gpu.cos_cached),
                    cu(rope_gpu.sin_cached)
                )
                x = CUDA.randn(Float32, 8, 1, 4, 1)

                loss(x) = sum(abs2, rope_gpu(x))
                @test_nowarn gradient(loss, x)
            end

            @testset "Forward Pass (GPU)" begin
                x = reshape(Float32.(1:160), (16, 1, 10,1))
                x_gpu = cu(x)
                pe = RoPE(16, 10)
                pe_gpu = RoPE(
                    pe.head_dim,
                    cu(pe.cos_cached),
                    cu(pe.sin_cached)
                )

                cpu_output = pe(x)
                gpu_output = Array(pe_gpu(x_gpu))
                @test isapprox(cpu_output, gpu_output)
            end
        end
    else
        @info "CUDA GPU not available or not working, skipping GPU tests"
    end
end
@testset "AbsolutePE" begin
    @testset "specific output values" begin
        head_dim, seq_len, batch_size = 128, 100, 64
        x = Float32.(reshape(collect(1:head_dim*seq_len*batch_size),
                           head_dim, seq_len, batch_size)) ./
            (head_dim*seq_len*batch_size)

        pe = AbsolutePE(head_dim, seq_len)
        output = pe(permutedims(x, (2,1,3)))

        expected = Float32[
            1.2207f-6   1.0        3.66211f-6   1.0       6.10352f-6;
            0.841628    0.540461   0.76188      0.648067  0.681724;
            0.909611   -0.415832   0.987362    -0.160119  0.997799;
            0.14159    -0.989521   0.517778    -0.855327  0.778747;
            -0.756176   -0.653016  -0.316087    -0.947891  0.14217
        ]

        @test size(output) == (seq_len, head_dim, batch_size)
        @test output[1:5, 1:5, 1] ≈ expected rtol=1e-5
    end
end