# This file defines utilities to work with "effects"; i.e. mutation signals sent from low-level types to higher-level wrapping types

"""
    Effect

Abstract type for effects.
"""
abstract type Effect end

"""
    canhandle(x, effect::Effect)

Returns `true` if `x` can handle the effect.
"""
canhandle(::T, ::E) where {T,E} = canhandle(T, E)
canhandle(::Type{T}, ::Type{E}) where {T,E} = false # hasmethod(handle!, Tuple{T,E})

struct MissingEffectHandlerException{T,E} <: Core.Exception end
MissingEffectHandlerException(::T, ::E) where {T,E} = MissingEffectHandlerException{T,E}()
MissingEffectHandlerException(::Type{T}, ::Type{E}) where {T,E} = MissingEffectHandlerException{T,E}()
Base.showerror(io::IO, ::MissingEffectHandlerException{T,E}) where {T,E} = print(io, "$T cannot handle effect $E")

function checkhandle(x::T, effect::E) where {T,E}
    if !canhandle(x, effect)
        throw(MissingEffectHandlerException(T, E))
    end
end

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
