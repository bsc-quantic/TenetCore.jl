using Test
using SafeTestsets
using TenetNext

@testset "Unit" verbose = true begin
    @testset "Interfaces" verbose = true begin
        @testset "TensorNetwork" include("unit/interfaces/tensor_network.jl")
    end
end
