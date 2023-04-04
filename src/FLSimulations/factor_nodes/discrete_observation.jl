using ForneyLab: SoftFactor, generateId, @ensureVariables, addNode!, associate!, removePrefix
import ForneyLab: slug, averageEnergy, requiresBreaker, breakerParameters, ApproximationMethod

export DiscreteObservation

abstract type EpistemicFactor <: SoftFactor end
abstract type Generalized <: ApproximationMethod end
abstract type Bethe <: ApproximationMethod end

"""
Description:

    Composite node for discrete observation model

    out ∈ {0, 1}^d where Σ_k out_k = 1
    A ∈ R^{m × n} observation matrix
    c ∈ R^m goal prior statistics

    f(y, s, A, c)

Interfaces:

    1. y (internal edge exposed)
    2. s
    3. A
    4. c

Construction:

    DiscreteObservation{T}(id=:some_id)
"""
mutable struct DiscreteObservation{T<:ApproximationMethod} <: EpistemicFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    n_factors::Int64 # Number of categories in s (for initialization)
    n_iterations::Union{Int64, Nothing} # Number of Newton iterations

    function DiscreteObservation{T}(y, s, A, c; 
                                    id=generateId(DiscreteObservation),
                                    n_factors=2,
                                    n_iterations=nothing) where T<:ApproximationMethod
        @ensureVariables(y, s, A, c)
        self = new(id, Array{Interface}(undef, 4), Dict{Symbol,Interface}(), n_factors, n_iterations)
        addNode!(currentGraph(), self)
        self.i[:y] = self.interfaces[1] = associate!(Interface(self), y)
        self.i[:s] = self.interfaces[2] = associate!(Interface(self), s)
        self.i[:A] = self.interfaces[3] = associate!(Interface(self), A)
        self.i[:c] = self.interfaces[4] = associate!(Interface(self), c)

        return self
    end
end

slug(::Type{DiscreteObservation{T}}) where T<:ApproximationMethod = "DO{$(removePrefix(T))}"

# A breaker message is required if interface is partnered with a DO node out interface
requiresBreaker(interface::Interface, partner_interface::Interface, partner_node::DiscreteObservation{Generalized}) = (partner_interface == partner_node.interfaces[2])

breakerParameters(interface::Interface, partner_interface::Interface, partner_node::DiscreteObservation{Generalized}) = (Message{Categorical, Univariate}, (partner_node.n_factors,)) # Defaults to two factors

# Average energy functionals
function averageEnergy(::Type{DiscreteObservation{Generalized}},
                       ::Distribution{Univariate, Categorical}, # Unconstrained observation (not used)
                       marg_s::Distribution{Univariate},
                       marg_A::Distribution{MatrixVariate},
                       marg_c::Distribution)

    s = unsafeMean(marg_s)
    A = unsafeMean(marg_A)
    amb_A = unsafeAmbMean(marg_A)
    log_c = unsafeLogMean(marg_c)

    s'*amb_A - (A*s)'*log_c # + (A*s)'*safelog.(A*s) # Entropy term is included by algorithm
end

function averageEnergy(::Type{DiscreteObservation{Bethe}},
                       marg_y::Distribution{Univariate, Categorical}, # Unconstrained observation
                       marg_s::Distribution{Univariate},
                       marg_A::Distribution{MatrixVariate},
                       marg_c::Distribution)

    y = unsafeMean(marg_y)
    s = unsafeMean(marg_s)
    log_A = unsafeLogMean(marg_A)
    log_c = unsafeLogMean(marg_c)

    -y'*(log_A*s + log_c) # + y'*log_y # Entropy term is included by algorithm
end

function averageEnergy(::Type{<:DiscreteObservation}, # Holds for Generalized and Bethe
                       marg_y::Distribution{Multivariate, PointMass}, # Constrained observation
                       marg_s::Distribution{Univariate},
                       marg_A::Distribution{MatrixVariate},
                       marg_c::Distribution)

    y_hat = unsafeMean(marg_y)
    s = unsafeMean(marg_s)
    log_A = unsafeLogMean(marg_A)
    log_c = unsafeLogMean(marg_c)

    -y_hat'*(log_A*s + log_c)
end
