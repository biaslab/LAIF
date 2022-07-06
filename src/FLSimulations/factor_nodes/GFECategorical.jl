export GFECategorical

"""
Description:

    Composite node for discrete GFE

    out ∈ {0, 1}^d where Σ_k out_k = 1
    A ∈ R^{m × n}      observation matrix
    c ∈ R^m goal prior statistics

    f(out, A, c)

Interfaces:

    1. out
    2. A
    3. c

Construction:

    GFECategorical(id=:some_id)
"""
mutable struct GFECategorical <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function GFECategorical(out, A, c; id=generateId(GFECategorical))
        @ensureVariables(out, A, c)
        self = new(id, Array{Interface}(undef, 3), Dict{Symbol,Interface}())
        addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:A] = self.interfaces[2] = associate!(Interface(self), A)
        self.i[:c] = self.interfaces[3] = associate!(Interface(self), c)

        return self
    end
end

slug(::Type{GFECategorical}) = "GFECat"

# Average energy functional
function averageEnergy(::Type{GFECategorical}, 
                       marg_out::Distribution{Univariate, Categorical}, 
                       marg_A::Distribution{MatrixVariate, PointMass},
                       marg_c::Distribution{Multivariate, PointMass})
    
    s = marg_out.params[:p]
    A = marg_A.params[:m]
    c = marg_c.params[:m]

    -s'*diag(A'*log.(A .+ tiny)) - (A*s)'*log.(c .+ tiny) + (A*s)'*log.(A*s .+ tiny)
end
