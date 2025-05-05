# This file defines utilities to work with "effects"; i.e. mutation signals sent from low-level types to higher-level wrapping types

"""
    Effect

Abstract type for effects.
"""
abstract type Effect end

struct MissingEffectHandlerException{T,E} <: Core.Exception end
MissingEffectHandlerException(::T, ::E) where {T,E} = MissingEffectHandlerException{T,E}()
MissingEffectHandlerException(::Type{T}, ::Type{E}) where {T,E} = MissingEffectHandlerException{T,E}()
Base.showerror(io::IO, ::MissingEffectHandlerException{T,E}) where {T,E} = print(io, "$T cannot handle effect $E")

function checkeffect end

"""
    handle!(x, effect::Effect)

Handle the `effect` on `x`. By default, does nothing.
"""
function handle! end

# by default, ignore effects
function handle!(@nospecialize(x::X), @nospecialize(e)) where {X}
    @debug "ignored effect $e on $X"
    nothing
end

"""
    PushEffect{F} <: Effect

Represents the effect of pushing an object.
"""
struct PushEffect{F} <: Effect
    f::F
end

PushEffect(@nospecialize(f::Tensor)) = PushEffect{Tensor}(f)

"""
    DeleteEffect{F} <: Effect

Represents the effect of deleting an object.
"""
struct DeleteEffect{F} <: Effect
    f::F
end

DeleteEffect(@nospecialize(f::Tensor)) = DeleteEffect{Tensor}(f)

"""
    ReplaceEffect{F} <: Effect

Represents the effect of replacing an object with a new one.
"""
struct ReplaceEffect{O,N} <: Effect
    old::O
    new::N
end

ReplaceEffect(old::Tensor, new::Tensor) = ReplaceEffect{Tensor,Tensor}(old, new)
ReplaceEffect(old::Tensor, new::TN) where {TN} = ReplaceEffect{Tensor,TN}(old, new)
ReplaceEffect(f::Pair) = ReplaceEffect(f.first, f.second)

# Taggable interface
"""
    TagEffect{Tag,Obj} <: Effect

Represents the effect of setting a link or mapping between a `Tag` and an `Obj`ect.
"""
struct TagEffect{T,O} <: Effect
    tag::T
    obj::O
end

TagEffect(tag::T, @nospecialize(obj::Tensor)) where {T} = TagEffect{T,Tensor}(tag, obj)
TagEffect(tag::T, @nospecialize(obj::Index)) where {T} = TagEffect{T,Index}(tag, obj)

"""
    UntagEffect{Tag,Obj} <: Effect

Represents the effect of setting a link or mapping between a `Tag` and an `Obj`ect.
"""
struct UntagEffect{T} <: Effect
    tag::T
end

# Lattice interface
"""
    SetSiteEffect{Tag,Obj} <: Effect

Represents the effect of setting a mapping between a `Tag` and an `Obj`ect.
"""
struct SetSiteEffect{T,O} <: Effect
    site::T
    obj::O
end

SetSiteEffect(site::T, @nospecialize(obj::Tensor)) where {T} = SetSiteEffect{T,Tensor}(site, obj)
SetSiteEffect(site::T, @nospecialize(obj::Index)) where {T} = SetSiteEffect{T,Index}(site, obj)

"""
    SetBondEffect{Tag,Obj} <: Effect

Represents the effect of setting a mapping between a `Tag` and an `Obj`ect.
"""
struct SetBondEffect{T,O} <: Effect
    bond::T
    obj::O
end

SetBondEffect(bond::T, @nospecialize(obj::Tensor)) where {T} = SetBondEffect{T,Tensor}(bond, obj)
SetBondEffect(bond::T, @nospecialize(obj::Index)) where {T} = SetBondEffect{T,Index}(bond, obj)

"""
    UnsetSiteEffect{Tag} <: Effect

Represents the effect of unsetting a mapping of a `Site` `Tag`.
"""
struct UnsetSiteEffect{T} <: Effect
    site::T
end

"""
    UnsetBondEffect{Tag} <: Effect

Represents the effect of unsetting a mapping of a `Bond` `Tag`.
"""
struct UnsetBondEffect{T} <: Effect
    site::T
end

# Pluggable interface
"""
    SetPlugEffect{Tag,Obj} <: Effect

Represents the effect of setting a mapping between a `Plug` `Tag` and an `Obj`ect.
"""
struct SetPlugEffect{T,O} <: Effect
    plug::T
    obj::O
end

SetPlugEffect(plug::T, @nospecialize(obj::Index)) where {T} = SetPlugEffect{T,Index}(plug, obj)

"""
    UnsetPlugEffect{Tag} <: Effect

Represents the effect of unsetting a mapping of a `Plug` `Tag`.
"""
struct UnsetPlugEffect{T} <: Effect
    plug::T
end
