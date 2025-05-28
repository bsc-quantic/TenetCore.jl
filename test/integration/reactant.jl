using Test
using TenetCore
using Reactant
using Adapt
using Enzyme
using TenetCore: tensor_at_vertex, vertex_at_tensor

# TODO test `make_tracer`
# TODO test `create_result`
# TODO test `traced_getfield`

@testset "contract" begin
    A = Tensor([1.0 2.0; 3.0 4.0], (:i, :j))
    B = Tensor([5.0 6.0; 7.0 8.0], (:j, :k))
    tn = GenericTensorNetwork([A, B])
    tn_re = adapt(ConcreteRArray, tn)

    C = contract(tn)
    C_re = @jit contract(tn_re)

    @test C_re ≈ C
end

@testset "autodiff - contract" begin
    A = Tensor([1.0, 2.0], (:i,))
    B = Tensor([3.0, 4.0], (:i,))
    tn = GenericTensorNetwork([A, B])
    tn_re = adapt(ConcreteRArray, tn)

    grad_contract(x) = Enzyme.gradient(Reverse, contract, x)

    (grad_tn,) = @jit grad_contract(tn_re)
    @test tensor_at_vertex(grad_tn, vertex_at_tensor(tn, A)) ≈ B
    @test tensor_at_vertex(grad_tn, vertex_at_tensor(tn, B)) ≈ A
end
