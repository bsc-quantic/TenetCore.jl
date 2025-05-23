# from https://discourse.julialang.org/t/sort-keys-of-namedtuple/94630/3
@generated sort_nt(nt::NamedTuple{KS}) where {KS} = :(NamedTuple{$(Tuple(sort(collect(KS))))}(nt))
sort_nt(nt::@NamedTuple{}) = nt

mutable struct CachedField{T}
    isvalid::Bool
    value::T
end

CachedField{T}() where {T} = CachedField{T}(false, T())

invalidate!(cf::CachedField) = cf.isvalid = false
function Base.get!(f, cf::CachedField)
    !cf.isvalid && (cf.value = f())
    cf.isvalid = true
    return cf.value
end

function hist(x; init=Dict{eltype(x),Int}())
    for xi in x
        if haskey(init, xi)
            init[xi] += 1
        else
            init[xi] = 1
        end
    end
    return init
end

# gensymclean(i) = gensym(String(split(string(i), "#")[3]))
gensymclean(ind) = gensym(String(only(match(r"(?:##)*(\w+)(?:#\d+)*", string(ind)).captures)))

const NUM_UNICODE_LETTERS = VERSION >= v"1.9" ? 136104 : 131756

"""
    letter(i)

Return `i`-th printable Unicode letter.

# Examples

```jldoctest; setup = :(letter = Tenet.letter)
julia> letter(1)
:A

julia> letter(27)
:a

julia> letter(249)
:ƃ

julia> letter(20204)
:櫛
```
"""
letter(i) = Symbol(first(iterate(Iterators.drop(Iterators.filter(isletter, Iterators.map(Char, 1:(2^21 - 1))), i - 1))))

# NOTE from https://stackoverflow.com/q/54652787
function nonunique(x)
    uniqueindexes = indexin(unique(x), collect(x))
    nonuniqueindexes = setdiff(1:length(x), uniqueindexes)
    return Tuple(unique(x[nonuniqueindexes]))
end

struct IndexCounter
    counter::Threads.Atomic{Int}

    IndexCounter(init::Int=1) = new(Threads.Atomic{Int}(init))
end

currindex(gen::IndexCounter) = letter(gen.counter[])
function nextindex!(gen::IndexCounter)
    if gen.counter.value >= 135000
        throw(ErrorException("run-out of indices!"))
    end
    return letter(Threads.atomic_add!(gen.counter, 1))
end
resetinds!(gen::IndexCounter) = letter(Threads.atomic_xchg!(gen.counter, 1))
