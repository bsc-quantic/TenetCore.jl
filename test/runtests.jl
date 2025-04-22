using Test
using SafeTestsets
using TenetNext

@testset "Unit" verbose = true begin
    @testset "TensorNetwork" verbose = true include("unit/tensor_network.jl")
    @testset "TaggedTensorNetwork" verbose = false include("unit/tagged_tensor_network.jl")
end
