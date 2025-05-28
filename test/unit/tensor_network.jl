using Test
using TenetCore
using DelegatorTraits
using Serialization

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

function TenetCore.addtensor!(tn::MockTensorNetwork, tensor::Tensor)
    hastensor(tn, tensor) && throw(ArgumentError("tensor already exists in the network"))
    push!(tn.tensors, tensor)
    return tn
end

function TenetCore.rmtensor!(tn::MockTensorNetwork, tensor::Tensor)
    !hastensor(tn, tensor) && throw(ArgumentError("tensor not found in the network"))
    deleteat!(tn.tensors, findfirst(x -> x === tensor, tn.tensors))
    return tn
end

function TenetCore.replace_tensor!(tn::MockTensorNetwork, old_tensor, new_tensor)
    old_tensor === new_tensor && return tn

    TenetCore.rmtensor!(tn, old_tensor)
    TenetCore.addtensor!(tn, new_tensor)
end

function TenetCore.replace_ind!(tn::MockTensorNetwork, old_ind, new_ind)
    old_ind == new_ind && return tn

    for old_tensor in tensors_contain_inds(tn, old_ind)
        new_tensor = replace(old_tensor, old_ind => new_ind)
        TenetCore.replace_tensor!(tn, old_tensor, new_tensor)
    end
end

function TenetCore.slice!(tn::MockTensorNetwork, ind, i)
    for old_tensor in tensors(tn; contain=ind)
        new_tensor = selectdim(old_tensor, ind, i)
        TenetCore.replace_tensor!(tn, old_tensor, new_tensor)
    end
    return tn
end

struct WrapperTensorNetwork{T} <: TenetCore.AbstractTensorNetwork
    tn::T
end

Base.copy(tn::WrapperTensorNetwork) = WrapperTensorNetwork(copy(tn.tn))
DelegatorTraits.DelegatorTrait(::TenetCore.UnsafeScopeable, ::WrapperTensorNetwork) = DelegateToField{:tn}()
DelegatorTraits.DelegatorTrait(::TenetCore.TensorNetwork, ::WrapperTensorNetwork) = DelegateToField{:tn}()

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

function test_mock_tensor_network(tn)
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
            tn isa SimpleTensorNetwork || tn isa WrapperTensorNetwork{SimpleTensorNetwork}
        @test tensors_with_inds(tn, [Index(:not_index)]) isa Vector{<:Tensor} broken =
            tn isa SimpleTensorNetwork || tn isa WrapperTensorNetwork{SimpleTensorNetwork}
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
            tn isa SimpleTensorNetwork || tn isa WrapperTensorNetwork{SimpleTensorNetwork}
        @test tensors_contain_inds(tn, [Index(:not_index)]) isa Vector{<:Tensor} broken =
            tn isa SimpleTensorNetwork || tn isa WrapperTensorNetwork{SimpleTensorNetwork}
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

        # throw error on wrong size of existing index
        # TODO broken for `MockTensorNetwork`
        if tn isa SimpleTensorNetwork || tn isa WrapperTensorNetwork{SimpleTensorNetwork}
            @testset let tn = copy(tn)
                new_tensor = Tensor(rand(2, 3), Index.([:m, :i]))
                @test_throws DimensionMismatch addtensor!(tn, new_tensor)
            end
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
        # WARN no longer allowed by default
        # @testset let tn = copy(tn)
        #     tensor_to_replace = test_tensors[1]
        #     new_tensor = Tensor(rand(2, 3), Index.([:m, :j]))

        #     # not allowed by default
        #     @test_throws ArgumentError replace_tensor!(tn, tensor_to_replace, new_tensor)

        #     # but allowed if inside an unsafe scope
        #     @unsafe_region tn replace_tensor!(tn, tensor_to_replace, new_tensor)
        #     @test hastensor(tn, new_tensor)
        #     @test !hastensor(tn, tensor_to_replace)
        # end
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

    @testset "Base.selectdim" begin
        @testset "i isa int" begin
            proj_tn = selectdim(tn, Index(:i), 1)
            @test !hasind(proj_tn, Index(:i))
            @test issetequal(
                tensors(proj_tn), map(x -> Index(:i) ∈ inds(x) ? selectdim(x, Index(:i), 1) : x, tensors(tn))
            )
        end

        @testset "i isa range" begin
            proj_tn = selectdim(tn, Index(:i), 1:1)
            @test hasind(proj_tn, Index(:i))
            @test size(proj_tn, Index(:i)) == 1
            @test issetequal(
                tensors(proj_tn), map(x -> Index(:i) ∈ inds(x) ? selectdim(x, Index(:i), 1:1) : x, tensors(tn))
            )
        end
    end

    @testset "Base.view" begin
        @testset let proj_tn = view(tn, Index(:i) => 1, Index(:j) => 1:2)
            @test !hasind(proj_tn, Index(:i))
            @test hasind(proj_tn, Index(:j))
            @test size(proj_tn, Index(:j)) == 2
            @test issetequal(
                tensors(proj_tn),
                map(tensors(tn)) do tensor
                    if Index(:i) ∈ inds(tensor) && Index(:j) ∈ inds(tensor)
                        selectdim(selectdim(tensor, Index(:i), 1), Index(:j), 1:2)
                    elseif Index(:i) ∈ inds(tensor)
                        selectdim(tensor, Index(:i), 1)
                    elseif Index(:j) ∈ inds(tensor)
                        selectdim(tensor, Index(:j), 1:2)
                    else
                        tensor
                    end
                end,
            )
        end
    end

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

@testset "MockTensorNetwork" begin
    tn = MockTensorNetwork(test_tensors)
    test_mock_tensor_network(tn)
end

@testset "SimpleTensorNetwork" begin
    tn = SimpleTensorNetwork(test_tensors)

    test_mock_tensor_network(tn)

    @testset "Serialization" begin
        # Serialize
        buffer = IOBuffer()
        serialize(buffer, tn)
        seekstart(buffer)
        content = read(buffer)

        # Deserialize
        read_buffer = IOBuffer(content)
        tn2 = deserialize(read_buffer)

        @test issetequal(all_tensors(tn2), all_tensors(tn))
    end
end

@testset "WrapperTensorNetwork{MockTensorNetwork}" begin
    tn = WrapperTensorNetwork(MockTensorNetwork(test_tensors))
    test_mock_tensor_network(tn)
end

@testset "WrapperTensorNetwork{SimpleTensorNetwork}" begin
    tn = WrapperTensorNetwork(SimpleTensorNetwork(test_tensors))
    test_mock_tensor_network(tn)
end
