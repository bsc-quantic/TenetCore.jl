using QuantumTags
using Bijections
using Networks
using Random

const SiteBijection = Bijection{Site,Vertex{UUID},Dict{Site,Vertex{UUID}},Dict{Vertex{UUID},Site}}
const LinkBijection = Bijection{Link,Edge{UUID},Dict{Link,Edge{UUID}},Dict{Edge{UUID},Link}}
# const BondBijection = Bijection{Bond,Edge{UUID},Dict{Bond,Edge{UUID}},Dict{Edge{UUID},Bond}}
# const PlugBijection = Bijection{Plug,Edge{UUID},Dict{Plug,Edge{UUID}},Dict{Edge{UUID},Plug}}

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

# Network interface
DelegatorTrait(::Network, ::GenericTensorNetwork) = DelegateToField{:tn}()

Networks.vertex_at(tn::GenericTensorNetwork, site::Site) = tn.sitemap[site]
Networks.edge_at(tn::GenericTensorNetwork, link::Link) = tn.linkmap[link]

# UnsafeScopeable interface
DelegatorTrait(::UnsafeScopeable, ::GenericTensorNetwork) = DelegateToField{:tn}()

# TensorNetwork interface
DelegatorTrait(::TensorNetwork, ::GenericTensorNetwork) = DelegateToField{:tn}()

tensor_at(tn::GenericTensorNetwork, site::Site) = tensor_at(tn, vertex_at(tn, site))
ind_at(tn::GenericTensorNetwork, link::Link) = ind_at(tn, edge_at(tn, link))

function rmtensor!(tn::GenericTensorNetwork, tensor)
    # it can break the mapping, so untag if the removed tensor is tagged
    _vertex = vertex_at(tn, tensor)
    if hasvalue(tn.sitemap, _vertex)
        site_tag = tn.sitemap(_vertex)
        untag_vertex!(tn, site_tag)
    end

    # propagate the mutation
    rmtensor!(tn.tn, tensor)

    return tn
end

function slice!(tn::GenericTensorNetwork, ind, i::Integer)
    # it can break the mapping, so untag if the sliced index is tagged
    _edge = edge_at(tn, ind)
    if hasvalue(tn.linkmap, _edge)
        link_tag = tn.linkmap(_edge)
        untag_edge!(tn, link_tag)
    end

    # propagate the mutation
    slice!(tn.tn, ind, i)

    return tn
end

# Taggable implementation
ImplementorTrait(::Networks.Taggable, ::GenericTensorNetwork) = Implements()

Networks.vertex_tags(tn::GenericTensorNetwork) = collect(keys(tn.sitemap))
Networks.edge_tags(tn::GenericTensorNetwork) = collect(keys(tn.linkmap))
Networks.has_vertex_tag(tn::GenericTensorNetwork, tag) = haskey(tn.sitemap, tag)
Networks.has_edge_tag(tn::GenericTensorNetwork, tag) = haskey(tn.linkmap, tag)
Networks.tag_at_vertex(tn::GenericTensorNetwork, vertex::Vertex) = tn.sitemap(vertex)
Networks.tag_at_edge(tn::GenericTensorNetwork, edge::Edge) = tn.linkmap(edge)

function Networks.tag_vertex!(tn::GenericTensorNetwork, vertex, site)
    hasvertex(tn, vertex) || throw(ArgumentError("Vertex $vertex not found in tensor network."))
    has_vertex_tag(tn, site) && throw(ArgumentError("Vertex tag $site already tagged in tensor network."))
    tn.sitemap[site] = vertex
    return tn
end

function Networks.tag_edge!(tn::GenericTensorNetwork, edge, link)
    hasedge(tn, edge) || throw(ArgumentError("Edge $edge not found in tensor network."))
    has_edge_tag(tn, link) && throw(ArgumentError("Edge tag $link already tagged in tensor network."))
    tn.linkmap[link] = edge
    return tn
end

function Networks.untag_vertex!(tn::GenericTensorNetwork, site)
    has_vertex_tag(tn, site) || throw(ArgumentError("Vertex tag $site not found in tensor network."))
    delete!(tn.sitemap, site)
    return tn
end

function Networks.untag_edge!(tn::GenericTensorNetwork, link)
    has_edge_tag(tn, link) || throw(ArgumentError("Edge tag $link not found in tensor network."))
    delete!(tn.linkmap, link)
    return tn
end

function Networks.replace_vertex_tag!(tn::GenericTensorNetwork, oldtag, newtag)
    has_vertex_tag(tn, oldtag) || throw(ArgumentError("Vertex tag $oldtag not found in tensor network."))
    has_vertex_tag(tn, newtag) && throw(ArgumentError("Vertex tag $newtag already tagged in tensor network."))
    vertex = tn.sitemap[oldtag]
    delete!(tn.sitemap, oldtag)
    tn.sitemap[newtag] = vertex
    return tn
end

function Networks.replace_edge_tag!(tn::GenericTensorNetwork, oldtag, newtag)
    has_edge_tag(tn, oldtag) || throw(ArgumentError("Edge tag $oldtag not found in tensor network."))
    has_edge_tag(tn, newtag) && throw(ArgumentError("Edge tag $newtag already tagged in tensor network."))
    edge = tn.linkmap[oldtag]
    delete!(tn.linkmap, oldtag)
    tn.linkmap[newtag] = edge
    return tn
end

# Lattice implementation
ImplementorTrait(::Lattice, ::GenericTensorNetwork) = Implements()

all_sites(tn::GenericTensorNetwork) = collect(all_sites_iter(tn))
all_bonds(tn::GenericTensorNetwork) = filter!(isbond, edge_tags(tn))

all_sites_iter(tn::GenericTensorNetwork) = keys(tn.sitemap)
all_bond_iter(tn::GenericTensorNetwork) = Iterators.filter(isbond, keys(tn.linkmap))

hassite(tn::GenericTensorNetwork, site) = has_vertex_tag(tn, site)
hasbond(tn::GenericTensorNetwork, link) = has_edge_tag(tn.linkmap, link)

nsites(::@NamedTuple{}, tn::GenericTensorNetwork) = length(tn.sitemap)
nbonds(::@NamedTuple{}, tn::GenericTensorNetwork) = count(isplug, edge_tags(tn))

setsite!(tn::GenericTensorNetwork, vertex, site) = tag_vertex!(tn, vertex, site)
setbond!(tn::GenericTensorNetwork, edge, bond) = tag_edge!(tn, edge, bond)

unsetsite!(tn::GenericTensorNetwork, site) = untag_vertex!(tn, site)
unsetbond!(tn::GenericTensorNetwork, bond) = untag_edge!(tn, bond)

site_at(tn::GenericTensorNetwork, vertex::Vertex) = tag_at_vertex(tn, vertex)
site_at(tn::GenericTensorNetwork, tensor::Tensor) = site_at(tn, vertex_at(tn, tensor))

# TODO should we check that it's a bond?
bond_at(tn::GenericTensorNetwork, edge::Edge) = tag_at_edge(tn, edge)
bond_at(tn::GenericTensorNetwork, ind) = bond_at(tn, edge_at(tn, ind))

setsite!(tn::GenericTensorNetwork, tensor::Tensor, site) = tag_vertex!(tn, vertex_at(tn, tensor), site)
setbond!(tn::GenericTensorNetwork, ind::Index, bond) = tag_edge!(tn, edge_at(tn, ind), bond)

# Pluggable implementation
ImplementorTrait(::Pluggable, ::GenericTensorNetwork) = Implements()

all_plugs(tn::GenericTensorNetwork) = filter!(isplug, edge_tags(tn))
all_plugs_iter(tn::GenericTensorNetwork) = Iterators.filter(isplug, keys(tn.linkmap))

ind_at(tn::GenericTensorNetwork, plug::Plug) = ind_at(tn, edge_at(tn, plug))

function setplug!(tn::GenericTensorNetwork, edge::Edge, plug)
    @assert !hasplug(tn, plug) "Plug $plug already found."
    @assert hasedge(tn, edge) "Edge $edge not found."
    # TODO check that the edge is not already tagged
    tag_edge!(tn, edge, plug)
    return tn
end
setplug!(tn::GenericTensorNetwork, ind::Index, plug) = setplug!(tn, edge_at(tn, ind), plug)

unsetplug!(tn::GenericTensorNetwork, plug) = untag_edge!(tn, plug)

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
