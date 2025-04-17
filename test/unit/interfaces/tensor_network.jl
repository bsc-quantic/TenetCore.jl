using Test
using TenetNext

struct MockTensorNetwork <: TenetNext.AbstractTensorNetwork
    tensors::Vector{Tensor}
    unsafe_scope::Ref{Union{Nothing,TenetNext.UnsafeScope}}

    MockTensorNetwork(tensors; unsafe=nothing) = new(tensors, Ref{Union{Nothing,TenetNext.UnsafeScope}}(unsafe))
end

# TensorNetwork implementation
Base.copy(tn::MockTensorNetwork) = MockTensorNetwork(copy(tn.tensors))

TenetNext.all_tensors(tn::MockTensorNetwork) = tn.tensors

function TenetNext.addtensor_inner!(tn::MockTensorNetwork, tensor::Tensor)
    if hastensor(tn, tensor)
        throw(ArgumentError("tensor already exists in the network"))
    end
    push!(tn.tensors, tensor)
    return tn
end

function TenetNext.rmtensor_inner!(tn::MockTensorNetwork, tensor::Tensor)
    if !hastensor(tn, tensor)
        throw(ArgumentError("tensor not found in the network"))
    end
    deleteat!(tn.tensors, findfirst(x -> x === tensor, tn.tensors))
    return tn
end

# UnsafeScopeable implementation
TenetNext.get_unsafe_scope(tn::MockTensorNetwork) = tn.unsafe_scope[]
TenetNext.set_unsafe_scope!(tn::MockTensorNetwork, uc) = tn.unsafe_scope[] = uc

mock_tensors = [
    Tensor(rand(2, 3), Index.([:i, :j])), Tensor(rand(3, 4), Index.([:j, :k])), Tensor(rand(3, 4), Index.([:j, :k]))
]

mock_tn = MockTensorNetwork(mock_tensors)

@testset "all_tensors" begin
    @test issetequal(tensors(mock_tn), mock_tensors)
end

@testset "all_inds" begin
    @test issetequal(all_inds(mock_tn), Index.([:i, :j, :k]))
end

@testset "hastensor" begin
    for tensor in mock_tensors
        @test hastensor(mock_tn, tensor)
    end

    # test a tensor with different data and indices
    @test !hastensor(mock_tn, Tensor(rand(2, 2), Index.([:m, :n])))

    # test a tensor with different data but same indices
    @test !hastensor(mock_tn, Tensor(rand(2, 2), Index.([:i, :j])))

    # test with copy (objectid is different so it's not the same)
    # TODO this behavior may change
    @test !hastensor(mock_tn, copy(mock_tensors[1]))
end

@testset "hasind" begin
    for ind in Index.([:i, :j, :k])
        @test hasind(mock_tn, ind)
    end

    @test !hasind(mock_tn, Index(:not_index))
end

@testset "ntensors" begin
    @test ntensors(mock_tn) == length(mock_tensors)
end

@testset "ninds" begin
    @test ninds(mock_tn) == length(all_inds(mock_tn))
end

@testset "size_ind(s)" begin
    @test size_inds(mock_tn) == Dict{Index,Int}(Index(:i) => 2, Index(:j) => 3, Index(:k) => 4)

    @test size_ind(mock_tn, Index(:i)) == 2
    @test size_ind(mock_tn, Index(:j)) == 3
    @test size_ind(mock_tn, Index(:k)) == 4
end

@testset "tensors_with_inds" begin
    # `with_inds` returns the tensors with the exact same indices
    @test isempty(tensors_with_inds(mock_tn, [Index(:i)]))

    @test issetequal(tensors_with_inds(mock_tn, [Index(:i), Index(:j)]), [mock_tensors[1]])

    # order doesn't matter
    @test issetequal(tensors_with_inds(mock_tn, [Index(:j), Index(:i)]), [mock_tensors[1]])

    # there can be more than one tensor with the same indices
    @test issetequal(tensors_with_inds(mock_tn, [Index(:j), Index(:k)]), [mock_tensors[2], mock_tensors[3]])

    # and again, order doesn't matter
    @test issetequal(tensors_with_inds(mock_tn, [Index(:k), Index(:j)]), [mock_tensors[2], mock_tensors[3]])

    # returning nothing should be type-stable
    @test isempty(tensors_with_inds(mock_tn, [Index(:not_index)]))
    @test tensors_with_inds(mock_tn, [Index(:not_index)]) isa Vector{<:Tensor}
end

@testset "tensors_contain_inds" begin
    # `contain_inds` returns the tensors for which the given indices are a subset
    @test issetequal(tensors_contain_inds(mock_tn, [Index(:i)]), [mock_tensors[1]])

    @test issetequal(tensors_contain_inds(mock_tn, [Index(:j)]), mock_tensors)

    @test issetequal(tensors_contain_inds(mock_tn, [Index(:k)]), [mock_tensors[2], mock_tensors[3]])

    @test issetequal(tensors_contain_inds(mock_tn, [Index(:i), Index(:j)]), [mock_tensors[1]])

    @test issetequal(tensors_contain_inds(mock_tn, [Index(:j), Index(:k)]), [mock_tensors[2], mock_tensors[3]])

    @test isempty(tensors_contain_inds(mock_tn, [Index(:i), Index(:j), Index(:k)]))

    # order doesn't matter
    @test issetequal(tensors_contain_inds(mock_tn, [Index(:j), Index(:i)]), [mock_tensors[1]])

    # returning nothing should be type-stable
    @test isempty(tensors_contain_inds(mock_tn, [Index(:not_index)]))
    @test tensors_contain_inds(mock_tn, [Index(:not_index)]) isa Vector{<:Tensor}
end

@testset "tensors_intersect_inds" begin
    # `intersect_inds` returns the tensors for which the given indices intersect
    @test issetequal(tensors_intersect_inds(mock_tn, [Index(:i)]), [mock_tensors[1]])

    @test issetequal(tensors_intersect_inds(mock_tn, [Index(:j)]), mock_tensors)

    @test issetequal(tensors_intersect_inds(mock_tn, [Index(:j), Index(:i)]), mock_tensors)

    @test issetequal(tensors_intersect_inds(mock_tn, [Index(:k)]), [mock_tensors[2], mock_tensors[3]])

    @test issetequal(tensors_intersect_inds(mock_tn, [Index(:i), Index(:j), Index(:k)]), mock_tensors)

    # order doesn't matter
    @test issetequal(tensors_intersect_inds(mock_tn, [Index(:i), Index(:j)]), mock_tensors)

    # returning nothing should be type-stable
    @test isempty(tensors_intersect_inds(mock_tn, [Index(:not_index)]))
    @test tensors_intersect_inds(mock_tn, [Index(:not_index)]) isa Vector{<:Tensor}
end

@testset "inds_set" begin
    @test issetequal(inds_set(mock_tn, :all), all_inds(mock_tn))
    @test issetequal(inds_set(mock_tn, :open), [Index(:i)])
    @test issetequal(inds_set(mock_tn, :inner), [Index(:k)])
    @test issetequal(inds_set(mock_tn, :hyper), [Index(:j)])
end

@testset "inds_parallel_to" begin
    @test isempty(inds_parallel_to(mock_tn, Index(:i)))

    let mock_tn = MockTensorNetwork(Tensor[mock_tensors[2], mock_tensors[3]])
        @test issetequal(inds_parallel_to(mock_tn, Index(:j)), [Index(:k)])
        @test issetequal(inds_parallel_to(mock_tn, Index(:k)), [Index(:j)])
    end
end

@testset "addtensor!" begin
    @testset let mock_tn = copy(mock_tn)
        new_tensor = Tensor(rand(2, 2), Index.([:m, :i]))
        addtensor!(mock_tn, new_tensor)
        @test hastensor(mock_tn, new_tensor)
    end
end

@testset "rmtensor!" begin
    @testset let mock_tn = copy(mock_tn)
        tensor_to_remove = mock_tensors[1]
        rmtensor!(mock_tn, tensor_to_remove)
        @test !hastensor(mock_tn, tensor_to_remove)
    end
end

@testset "replace_tensor!" begin
    @testset let mock_tn = copy(mock_tn)
        tensor_to_replace = mock_tensors[1]
        new_tensor = Tensor(rand(2, 2), Index.([:j, :i]))
        replace_tensor!(mock_tn, tensor_to_replace, new_tensor)
        @test hastensor(mock_tn, new_tensor)
        @test !hastensor(mock_tn, tensor_to_replace)
    end

    # replacement with index change
    @testset let mock_tn = copy(mock_tn)
        tensor_to_replace = mock_tensors[1]
        new_tensor = Tensor(rand(2, 2), Index.([:m, :n]))

        # not allowed by default
        @test_throws ArgumentError replace_tensor!(mock_tn, tensor_to_replace, new_tensor)

        # but allowed if inside an unsafe scope
        @unsafe_region mock_tn replace_tensor!(mock_tn, tensor_to_replace, new_tensor)
        @test hastensor(mock_tn, new_tensor)
        @test !hastensor(mock_tn, tensor_to_replace)
    end
end

@testset "replace_ind!" begin
    @testset let mock_tn = copy(mock_tn)
        replace_ind!(mock_tn, Index(:i), Index(:x))
        @test !hasind(mock_tn, Index(:i))
        @test hasind(mock_tn, Index(:x))
    end
end
