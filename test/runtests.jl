using Test
using SafeTestsets

@testset "Unit" verbose = true begin
    @safetestset "TensorNetwork" include("unit/tensor_network.jl")
    @safetestset "Taggable" include("unit/taggable.jl")
    @safetestset "Pluggable" include("unit/pluggable.jl")
end

@testset "Integration" verbose = true begin
    @safetestset "Reactant" include("integration/reactant.jl")
end
