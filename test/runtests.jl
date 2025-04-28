using Test
using SafeTestsets
using TenetCore

@testset "Unit" verbose = true begin
    @testset "TensorNetwork" verbose = true include("unit/tensor_network.jl")
    @testset "TaggedTensorNetwork" verbose = true include("unit/tagged_tensor_network.jl")
end
