using Test
using TenetCore

struct MockTensorNetwork <: TenetCore.AbstractTensorNetwork
    tensors::Vector{Tensor}
    unsafe_scope::Ref{Union{Nothing,TenetCore.UnsafeScope}}

    MockTensorNetwork(tensors; unsafe=nothing) = new(tensors, unsafe)
end

# UnsafeScopeable implementation
TenetCore.get_unsafe_scope(tn::MockTensorNetwork) = tn.unsafe_scope[]
TenetCore.set_unsafe_scope!(tn::MockTensorNetwork, uc) = tn.unsafe_scope[] = uc

# TensorNetwork implementation
function Base.copy(tn::MockTensorNetwork)
    unsafe = Ref{Union{Nothing,TenetCore.UnsafeScope}}(tn.unsafe_scope[])
    new_tn = MockTensorNetwork(copy(tn.tensors); unsafe)

    # register the new copy to the proper UnsafeScope
    if !isnothing(unsafe[])
        push!(unsafe[].refs, WeakRef(new_tn))
    end

    return new_tn
end

TenetCore.all_tensors(tn::MockTensorNetwork) = collect(tn.tensors)

function TenetCore.addtensor_inner!(tn::MockTensorNetwork, tensor::Tensor)
    if hastensor(tn, tensor)
        throw(ArgumentError("tensor already exists in the network"))
    end
    push!(tn.tensors, tensor)
    return tn
end

TenetCore.handle!(::MockTensorNetwork, ::TenetCore.PushEffect{<:Tensor}) = nothing

function TenetCore.rmtensor_inner!(tn::MockTensorNetwork, tensor::Tensor)
    if !hastensor(tn, tensor)
        throw(ArgumentError("tensor not found in the network"))
    end
    deleteat!(tn.tensors, findfirst(x -> x === tensor, tn.tensors))
    return tn
end

TenetCore.handle!(::MockTensorNetwork, ::TenetCore.DeleteEffect{<:Tensor}) = nothing

TenetCore.handle!(::MockTensorNetwork, ::TenetCore.ReplaceEffect{<:Tensor,<:Tensor}) = nothing
TenetCore.handle!(::MockTensorNetwork, ::TenetCore.ReplaceEffect{<:Index,<:Index}) = nothing

struct WrapperTensorNetwork{T} <: TenetCore.AbstractTensorNetwork
    tn::T
end

Base.copy(tn::WrapperTensorNetwork) = WrapperTensorNetwork(copy(tn.tn))
TenetCore.delegates(::TenetCore.UnsafeScopeable, ::WrapperTensorNetwork) = TenetCore.DelegateTo{:tn}()
TenetCore.delegates(::TenetCore.TensorNetwork, ::WrapperTensorNetwork) = TenetCore.DelegateTo{:tn}()

test_tensors = [
    Tensor(rand(ComplexF64, 2, 3), Index.([:i, :j])),
    Tensor(rand(ComplexF64, 3, 4), Index.([:j, :k])),
    Tensor(rand(ComplexF64, 3, 4), Index.([:j, :k])),
]

test_inds = Index.([:i, :j, :k])
test_inds_open = [Index(:i)]
test_inds_inner = [Index(:k)]
test_inds_hyper = [Index(:j)]

test_size = Dict(Index(:i) => 2, Index(:j) => 3, Index(:k) => 4)

@testset "$(typeof(tn))" for tn in [
    MockTensorNetwork(test_tensors),
    GenericTensorNetwork(test_tensors),
    WrapperTensorNetwork(MockTensorNetwork(test_tensors)),
    WrapperTensorNetwork(GenericTensorNetwork(test_tensors)),
]
    @testset "all_tensors" begin
        @test issetequal(tensors(tn), test_tensors)
    end

    @testset "all_inds" begin
        @test issetequal(all_inds(tn), test_inds)
    end

    @testset "hastensor" begin
        for tensor in test_tensors
            @test hastensor(tn, tensor)
        end

        # test a tensor with different data and indices
        @test !hastensor(tn, Tensor(rand(2, 2), Index.([:not_index, :not_index_2])))

        # test a tensor with different data but same indices
        @test !hastensor(tn, Tensor(rand(2, 2), Index.([:i, :j])))

        # test with copy (objectid is different so it's not the same)
        # TODO this behavior may change
        @test !hastensor(tn, copy(test_tensors[1]))
    end

    @testset "hasind" begin
        for ind in test_inds
            @test hasind(tn, ind)
        end

        @test !hasind(tn, Index(:not_index))
    end

    @testset "ntensors" begin
        @test ntensors(tn) == length(test_tensors)
    end

    @testset "ninds" begin
        @test ninds(tn) == length(test_inds)
    end

    @testset "size_ind(s)" begin
        @test size_inds(tn) == test_size

        for i in test_inds
            @test size_ind(tn, i) == test_size[i]
        end
    end

    @testset "tensors_with_inds" begin
        # `with_inds` returns the tensors with the exact same indices
        @test isempty(tensors_with_inds(tn, [Index(:i)]))
        @test issetequal(tensors_with_inds(tn, [Index(:i), Index(:j)]), [test_tensors[1]])

        # order doesn't matter
        @test issetequal(tensors_with_inds(tn, [Index(:j), Index(:i)]), [test_tensors[1]])

        # there can be more than one tensor with the same indices
        @test issetequal(tensors_with_inds(tn, [Index(:j), Index(:k)]), [test_tensors[2], test_tensors[3]])

        # and again, order doesn't matter
        @test issetequal(tensors_with_inds(tn, [Index(:k), Index(:j)]), [test_tensors[2], test_tensors[3]])

        # returning nothing should be type-stable
        @test isempty(tensors_with_inds(tn, [Index(:not_index)])) broken =
            tn isa GenericTensorNetwork || tn isa WrapperTensorNetwork{GenericTensorNetwork}
        @test tensors_with_inds(tn, [Index(:not_index)]) isa Vector{<:Tensor} broken =
            tn isa GenericTensorNetwork || tn isa WrapperTensorNetwork{GenericTensorNetwork}
    end

    @testset "tensors_contain_inds" begin
        # `contain_inds` returns the tensors for which the given indices are a subset
        @test issetequal(tensors_contain_inds(tn, [Index(:i)]), [test_tensors[1]])
        @test issetequal(tensors_contain_inds(tn, [Index(:j)]), test_tensors)
        @test issetequal(tensors_contain_inds(tn, [Index(:k)]), [test_tensors[2], test_tensors[3]])
        @test issetequal(tensors_contain_inds(tn, [Index(:i), Index(:j)]), [test_tensors[1]])
        @test issetequal(tensors_contain_inds(tn, [Index(:j), Index(:k)]), [test_tensors[2], test_tensors[3]])
        @test isempty(tensors_contain_inds(tn, [Index(:i), Index(:j), Index(:k)]))

        # order doesn't matter
        @test issetequal(tensors_contain_inds(tn, [Index(:j), Index(:i)]), [test_tensors[1]])

        # returning nothing should be type-stable
        @test isempty(tensors_contain_inds(tn, [Index(:not_index)])) broken =
            tn isa GenericTensorNetwork || tn isa WrapperTensorNetwork{GenericTensorNetwork}
        @test tensors_contain_inds(tn, [Index(:not_index)]) isa Vector{<:Tensor} broken =
            tn isa GenericTensorNetwork || tn isa WrapperTensorNetwork{GenericTensorNetwork}
    end

    @testset "tensors_intersect_inds" begin
        # `intersect_inds` returns the tensors for which the given indices intersect
        @test issetequal(tensors_intersect_inds(tn, [Index(:i)]), [test_tensors[1]])
        @test issetequal(tensors_intersect_inds(tn, [Index(:j)]), test_tensors)
        @test issetequal(tensors_intersect_inds(tn, [Index(:j), Index(:i)]), test_tensors)
        @test issetequal(tensors_intersect_inds(tn, [Index(:k)]), [test_tensors[2], test_tensors[3]])
        @test issetequal(tensors_intersect_inds(tn, [Index(:i), Index(:j), Index(:k)]), test_tensors)

        # order doesn't matter
        @test issetequal(tensors_intersect_inds(tn, [Index(:i), Index(:j)]), test_tensors)

        # returning nothing should be type-stable
        @test isempty(tensors_intersect_inds(tn, [Index(:not_index)]))
        @test tensors_intersect_inds(tn, [Index(:not_index)]) isa Vector{<:Tensor}
    end

    @testset "inds_set" begin
        @test issetequal(inds_set(tn, :all), test_inds)
        @test issetequal(inds_set(tn, :open), test_inds_open)
        @test issetequal(inds_set(tn, :inner), test_inds_inner)
        @test issetequal(inds_set(tn, :hyper), test_inds_hyper)
    end

    @testset "inds_parallel_to" begin
        @test isempty(inds_parallel_to(tn, Index(:i)))

        let tn = MockTensorNetwork(Tensor[test_tensors[2], test_tensors[3]])
            @test issetequal(inds_parallel_to(tn, Index(:j)), [Index(:k)])
            @test issetequal(inds_parallel_to(tn, Index(:k)), [Index(:j)])
        end
    end

    @testset "addtensor!" begin
        @testset let tn = copy(tn)
            new_tensor = Tensor(rand(2, 2), Index.([:m, :i]))
            addtensor!(tn, new_tensor)
            @test hastensor(tn, new_tensor)
        end
    end

    @testset "rmtensor!" begin
        @testset let tn = copy(tn)
            tensor_to_remove = test_tensors[1]
            rmtensor!(tn, tensor_to_remove)
            @test !hastensor(tn, tensor_to_remove)
        end
    end

    @testset "replace_tensor!" begin
        @testset let tn = copy(tn)
            tensor_to_replace = test_tensors[1]
            new_tensor = similar(test_tensors[1])
            replace_tensor!(tn, tensor_to_replace, new_tensor)
            @test hastensor(tn, new_tensor)
            @test !hastensor(tn, tensor_to_replace)
        end

        # replacement with index change
        @testset let tn = copy(tn)
            tensor_to_replace = test_tensors[1]
            new_tensor = Tensor(rand(2, 3), Index.([:m, :j]))

            # not allowed by default
            @test_throws ArgumentError replace_tensor!(tn, tensor_to_replace, new_tensor)

            # but allowed if inside an unsafe scope
            @unsafe_region tn replace_tensor!(tn, tensor_to_replace, new_tensor)
            @test hastensor(tn, new_tensor)
            @test !hastensor(tn, tensor_to_replace)
        end
    end

    @testset "replace_ind!" begin
        @testset let tn = copy(tn)
            replace_ind!(tn, Index(:i), Index(:x))
            @test !hasind(tn, Index(:i))
            @test hasind(tn, Index(:x))
        end
    end

    # TODO test aliases
    @testset "Base.in" begin
        # `hastensor`
        for tensor in test_tensors
            @test tensor ∈ tn
        end
        @test Tensor(zeros()) ∉ tn

        # `hasind`
        for i in test_inds
            @test i ∈ tn
        end
        @test Index(:not_index) ∉ tn
    end

    @testset "Base.size" begin
        # `size_inds`
        @test size(tn) == size_inds(tn)

        # `size_ind`
        for i in test_inds
            @test size(tn, i) == size_ind(tn, i)
        end
    end

    @testset "Base.eltype" begin
        @test eltype(tn) == Base.promote_eltype(test_tensors...)
    end

    @testset "Base.collect" begin
        @test issetequal(collect(tn), test_tensors)
    end

    @testset "Base.similar" begin
        similar_tn = similar(tn)
        @test !issetequal(tensors(similar_tn), test_tensors)
        @test issetequal(inds.(tensors(similar_tn)), inds.(test_tensors))
    end

    @testset "Base.zero" begin
        zero_tn = zero(tn)
        @test all(iszero, parent.(tensors(zero_tn)))
        @test issetequal(inds.(tensors(zero_tn)), inds.(test_tensors))
    end

    @testset "Base.conj" begin
        @test issetequal(tensors(conj(tn)), conj.(test_tensors))
    end

    @testset "Base.conj!" begin
        @testset let conj_tn = deepcopy(tn)
            @test issetequal(tensors(conj_tn), test_tensors)
            conj!(conj_tn)
            @test issetequal(tensors(conj_tn), conj.(test_tensors))
        end
    end

    @testset "Base.selectdim" begin end
    @testset "Base.view" begin end
    @testset "Base.neighbors" begin end
    @testset "Base.push!" begin end
    @testset "Base.append!" begin end
    @testset "Base.pop!" begin end
    @testset "Base.delete!" begin end
    @testset "Base.replace!" begin end

    # TODO test derived methods
    @testset "arrays" begin
        @test issetequal(arrays(tn), parent.(test_tensors))
    end

    @testset "slice!" begin end
    @testset "EinExprs.einexpr" begin end
    @testset "contract" begin end
    @testset "resetinds!" begin end
    @testset "fuse!" begin end
end
