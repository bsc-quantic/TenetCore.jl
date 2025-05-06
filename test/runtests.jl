using Test
using SafeTestsets
using TenetCore

@testset "Unit" verbose = true begin
    @testset "TensorNetwork" verbose = true include("unit/tensor_network.jl")
    @testset "Taggable" verbose = true include("unit/taggable.jl")
end
