using Test
using SafeTestsets

@testset "Unit" verbose = true begin
    @testset "TensorNetwork" verbose = true include("unit/tensor_network.jl")
    @testset "Taggable" verbose = true include("unit/taggable.jl")
    @safetestset "Pluggable" include("unit/pluggable.jl")
end

@testset "Integration" verbose = true begin
    @safetestset "Reactant" include("integration/reactant.jl")
end
