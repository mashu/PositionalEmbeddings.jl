using Test
using CUDA
using Zygote
using PositionalEmbeddings

function has_working_cuda()
    return CUDA.has_cuda() && CUDA.functional()
end
@testset "RoPE Tests" begin
    @testset "Gradient Tests (CPU, Float64)" begin
        eps = 1e-8
        rope = RoPE(8, 4; T=Float64)
        x = randn(8, 4, 1)
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

    @testset "Reference Output Test (CPU)" begin
        features, seq_len, batch_size = 16, 10, 64
        x = Float32.(reshape(collect(1:features*seq_len*batch_size),
                            features, seq_len, batch_size)) ./
            (features*seq_len*batch_size)

        reference = Float32[
            9.76563f-5    0.000195313  0.000292969  0.000390625  0.000488281
            -0.000582162   0.00106326   0.00163171   0.00187805   0.00202529
            -0.00436025    0.000599911  0.00261259   0.00326171   0.00353053
            -0.00542633   -0.00128126   0.00319962   0.00453726   0.00500356
            0.000728724  -0.00439145   0.00336443   0.00570055   0.00644391
        ]

        rope = RoPE(features, seq_len)
        output = rope(x)
        result = output[1:5,1:5,1]'

        @test all(isapprox.(result, reference, atol=1e-4))
    end

    if has_working_cuda()
        @testset "GPU Tests" begin
            @testset "Gradient Computation (GPU, Float32)" begin
                rope_gpu = RoPE(8, 4; T=Float32)
                rope_gpu = RoPE(
                    rope_gpu.features,
                    cu(rope_gpu.cos_cached),
                    cu(rope_gpu.sin_cached),
                    rope_gpu.scale
                )
                x = CUDA.randn(Float32, 8, 4, 1)

                # Just verify that gradient computation doesn't error
                loss(x) = sum(abs2, rope_gpu(x))
                @test_nowarn gradient(loss, x)
            end

            @testset "Reference Output Test (GPU)" begin
                features, seq_len, batch_size = 16, 10, 64
                x = Float32.(reshape(collect(1:features*seq_len*batch_size),
                                   features, seq_len, batch_size)) ./
                    (features*seq_len*batch_size)
                x = cu(x)

                reference = Float32[
                    9.76563f-5    0.000195313  0.000292969  0.000390625  0.000488281
                   -0.000582162   0.00106326   0.00163171   0.00187805   0.00202529
                   -0.00436025    0.000599911  0.00261259   0.00326171   0.00353053
                   -0.00542633   -0.00128126   0.00319962   0.00453726   0.00500356
                    0.000728724  -0.00439145   0.00336443   0.00570055   0.00644391
                ]

                rope = RoPE(features, seq_len)
                rope_gpu = RoPE(
                    rope.features,
                    cu(rope.cos_cached),
                    cu(rope.sin_cached),
                    rope.scale
                )
                output = rope_gpu(x)
                result = Array(output[1:5,1:5,1]')

                @test all(isapprox.(result, reference, atol=1e-4))
            end
        end
    else
        @info "CUDA GPU not available or not working, skipping GPU tests"
    end
end
@testset "AbsolutePE" begin
    @testset "specific output values" begin
        features, seq_len, batch_size = 128, 100, 64
        x = Float32.(reshape(collect(1:features*seq_len*batch_size),
                           features, seq_len, batch_size)) ./
            (features*seq_len*batch_size)

        pe = AbsolutePE(features, seq_len)
        output = pe(permutedims(x, (2,1,3)))

        expected = Float32[
            1.2207f-6   1.0        3.66211f-6   1.0       6.10352f-6;
            0.841628    0.540461   0.76188      0.648067  0.681724;
            0.909611   -0.415832   0.987362    -0.160119  0.997799;
            0.14159    -0.989521   0.517778    -0.855327  0.778747;
            -0.756176   -0.653016  -0.316087    -0.947891  0.14217
        ]

        @test size(output) == (seq_len, features, batch_size)
        @test output[1:5, 1:5, 1] ≈ expected rtol=1e-5
    end
end