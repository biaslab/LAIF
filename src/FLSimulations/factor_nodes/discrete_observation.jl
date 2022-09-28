using ForneyLab: SoftFactor, generateId, @ensureVariables, addNode!, associate!
import ForneyLab: slug, averageEnergy, requiresBreaker, breakerParameters

export DiscreteObservation

abstract type EpistemicFactor <: SoftFactor end

"""
Description:

    Composite node for discrete epistemic observation model

    out ∈ {0, 1}^d where Σ_k out_k = 1
    A ∈ R^{m × n}      observation matrix
    c ∈ R^m goal prior statistics

    f(out, A, c)

Interfaces:

    1. out
    2. A
    3. c

Construction:

    DiscreteObservation(id=:some_id)
"""
mutable struct DiscreteObservation <: EpistemicFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    n_factors::Int64 # Number of categories (for initialization)
    n_iterations::Union{Int64, Nothing} # Number of Newton iterations

    function DiscreteObservation(out, A, c; id=generateId(DiscreteObservation), n_factors=2, n_iterations=nothing)
        @ensureVariables(out, A, c)
        self = new(id, Array{Interface}(undef, 3), Dict{Symbol,Interface}(), n_factors, n_iterations)
        addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:A] = self.interfaces[2] = associate!(Interface(self), A)
        self.i[:c] = self.interfaces[3] = associate!(Interface(self), c)

        return self
    end
end

# Helper function to prevent log of 0
safelog(x) = log(clamp(x,tiny,Inf))

slug(::Type{DiscreteObservation}) = "DO"

# A breaker message is required if interface is partnered with a DO node
requiresBreaker(interface::Interface, partner_interface::Interface, partner_node::DiscreteObservation) = true

breakerParameters(interface::Interface, partner_interface::Interface, partner_node::DiscreteObservation) = (Message{Categorical, Univariate}, (partner_node.n_factors,)) # Defaults to two factors

# Average energy functional
function averageEnergy(::Type{DiscreteObservation},
                       marg_out::Distribution{Univariate},
                       marg_A::Distribution{MatrixVariate},
                       marg_c::Distribution{Multivariate})

    s = unsafeMean(marg_out)
    A = unsafeMean(marg_A)
    log_c = unsafeLogMean(marg_c)

    (A*s)'*safelog.(A*s) - s'*diag(A'*safelog.(A)) - (A*s)'*log_c
end
