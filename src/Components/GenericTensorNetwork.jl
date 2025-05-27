using QuantumTags
using Serialization
using Random
using Base: IdSet
using Bijections

const LinkBijection = Bijection{Link,Edge{UUID},Dict{Link,Edge{UUID}},Dict{Edge{UUID},Link}}
const SiteBijection = Bijection{Site,Vertex{UUID},Dict{Site,Vertex{UUID}},Dict{Vertex{UUID},Site}}

struct GenericTensorNetwork <: AbstractTensorNetwork
    tn::SimpleTensorNetwork
    sitemap::SiteBijection
    linkmap::LinkBijection
end

GenericTensorNetwork(; kwargs...) = GenericTensorNetwork(SimpleTensorNetwork(; kwargs...))
GenericTensorNetwork(tn::SimpleTensorNetwork) = GenericTensorNetwork(tn, SiteBijection(), LinkBijection())

# TODO Find a way to remove the `unsafe` keyword argument from the constructor
GenericTensorNetwork(tensors; kwargs...) = GenericTensorNetwork(SimpleTensorNetwork(tensors; kwargs...))

Base.copy(tn::GenericTensorNetwork) = GenericTensorNetwork(copy(tn.tn), copy(tn.sitemap), copy(tn.linkmap))

# delegation
DelegatorTrait(::Network, ::GenericTensorNetwork) = DelegateTo{:tn}()
DelegatorTrait(::UnsafeScopeable, ::GenericTensorNetwork) = DelegateTo{:tn}()
DelegatorTrait(::TensorNetwork, ::GenericTensorNetwork) = DelegateTo{:tn}()

## Taggable implementation
# DelegatorTrait(::Taggable, ::GenericTensorNetwork) = DelegateTo{:tags}()
ImplementorTrait(::Taggable, ::GenericTensorNetwork) = Implements()

all_sites(tn::GenericTensorNetwork) = collect(all_sites_iter(tn))
all_links(tn::GenericTensorNetwork) = collect(all_links_iter(tn))

all_sites_iter(tn::GenericTensorNetwork) = keys(tn.sitemap)
all_links_iter(tn::GenericTensorNetwork) = keys(tn.linkmap)

hassite(tn::GenericTensorNetwork, site) = haskey(tn.sitemap, site)
haslink(tn::GenericTensorNetwork, link) = haskey(tn.linkmap, link)

nsites(::@NamedTuple{}, tn::GenericTensorNetwork) = length(tn.sitemap)
nlinks(::@NamedTuple{}, tn::GenericTensorNetwork) = length(tn.linkmap)

site_vertex(tn::GenericTensorNetwork, site) = tn.sitemap[site]
link_edge(tn::GenericTensorNetwork, link) = tn.linkmap[link]

vertex_site(tn::GenericTensorNetwork, vertex) = inv(tn.sitemap)[vertex]
edge_link(tn::GenericTensorNetwork, edge) = inv(tn.linkmap)[edge]

## override to get tensor/index from this level and not from the mixin (which can't)
# tensor_at(tn::GenericTensorNetwork, tag) = tensor(tn; vertex=site_vertex(tn, tag))
# ind_at(tn::GenericTensorNetwork, tag) = ind(tn; edge=link_edge(tn, tag))

# site_at(tn::GenericTensorNetwork, tensor::Tensor) = vertex_site(tn, tensor_vertex(tn, tensor))
# link_at(tn::GenericTensorNetwork, ind::Index) = edge_link(tn, index_edge(tn, ind))

tag_inner!(tn::GenericTensorNetwork, vertex::Vertex, site::Site) = tn.sitemap[site] = vertex
tag_inner!(tn::GenericTensorNetwork, edge::Edge, link::Link) = tn.linkmap[link] = edge

untag_inner!(tn::GenericTensorNetwork, site::Site) = delete!(tn.sitemap, site)
untag_inner!(tn::GenericTensorNetwork, link::Link) = delete!(tn.linkmap, link)

## Pluggable implementation
ImplementorTrait(::Pluggable, ::GenericTensorNetwork) = Implements()
DelegatorTrait(::Pluggable, ::GenericTensorNetwork) = DelegateToInterface{Taggable}()

# effects
function handle!(tn::GenericTensorNetwork, @nospecialize(e::RemoveTensorEffect))
    # it can break the mapping, so untag if the removed tensor is tagged
    _vertex = tensor_vertex(tn, e.f)
    if hasvalue(tn.sitemap, _vertex)
        site_tag = site_at(tn, e.f)
        untag_inner!(tn, site_tag)
    end

    # propagate the effect
    handle!(tn.tn, e)
end

function handle!(tn::GenericTensorNetwork, e::SliceEffect{<:Integer})
    # it can break the mapping, so untag if the sliced index is tagged
    _edge = index_edge(tn, e.ind)
    if hasvalue(tn.linkmap, _edge)
        link_tag = link_at(tn, e.ind)
        untag_inner!(tn, link_tag)
    end

    # propagate the effect
    handle!(tn.tn, e)
end

# derived methods
Base.:(==)(a::GenericTensorNetwork, b::GenericTensorNetwork) = all(splat(==), zip(tensors(a), tensors(b)))
function Base.isapprox(a::GenericTensorNetwork, b::GenericTensorNetwork; kwargs...)
    return all(((x, y),) -> isapprox(x, y; kwargs...), zip(tensors(a), tensors(b)))
end

"""
    rand(TensorNetwork, n::Integer, regularity::Integer; out = 0, dim = 2:9, seed = nothing, globalind = false)

Generate a random tensor network.

# Arguments

  - `n` Number of tensors.
  - `regularity` Average number of indices per tensor.
  - `out` Number of open indices.
  - `dim` Range of dimension sizes.
  - `seed` If not `nothing`, seed random generator with this value.
  - `globalind` Add a global 'broadcast' dimension to every tensor.
"""
function Base.rand(
    rng::Random.AbstractRNG,
    ::Type{TensorNetwork},
    n::Integer,
    regularity::Integer;
    out=0,
    dim=2:9,
    seed=nothing,
    globalind=false,
    eltype=Float64,
)
    !isnothing(seed) && Random.seed!(rng, seed)

    inds = letter.(randperm(n * regularity ÷ 2 + out))
    size_dict = Dict(ind => rand(dim) for ind in inds)

    outer_inds = collect(Iterators.take(inds, out))
    inner_inds = collect(Iterators.drop(inds, out))

    candidate_inds = shuffle(
        collect(Iterators.flatten([outer_inds, Iterators.flatten(Iterators.repeated(inner_inds, 2))]))
    )

    inputs = map(x -> [x], Iterators.take(candidate_inds, n))

    for ind in Iterators.drop(candidate_inds, n)
        i = rand(1:n)
        while ind in inputs[i]
            i = rand(1:n)
        end

        push!(inputs[i], ind)
    end

    if globalind
        ninds = length(size_dict)
        ind = letter(ninds + 1)
        size_dict[ind] = rand(dim)
        push!(outer_inds, ind)
        push!.(inputs, (ind,))
    end

    tensors = Tensor[Tensor(rand(eltype, [size_dict[ind] for ind in input]...), tuple(input...)) for input in inputs]
    return GenericTensorNetwork(tensors)
end

function Base.rand(::Type{TensorNetwork}, n::Integer, regularity::Integer; kwargs...)
    return rand(Random.default_rng(), TensorNetwork, n, regularity; kwargs...)
end
