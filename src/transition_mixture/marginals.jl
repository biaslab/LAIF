using RxInfer,ReactiveMP,GraphPPL,Rocket, LinearAlgebra, OhMyREPL, Distributions;
enable_autocomplete_brackets(false),colorscheme!("GruvboxDark");
import Distributions: mean, entropy
import StatsBase: xlogx # This makes entropy calculations consistent with Distributions.jl

const Tensorvariate = ArrayLikeVariate{3}
const DiscreteTensorvariateDistribution   = Distribution{Tensorvariate,  Discrete}

struct ContingencyTensor{T<: Real, P <: AbstractArray{T}} <: DiscreteTensorvariateDistribution
    p::P
end

# Only use normalised tensors for now! Or baby dies....
Distributions.mean(dist::ContingencyTensor) = dist.p

# Clamplog means
mean(::typeof(ReactiveMP.clamplog), dist::MatrixDirichlet) = digamma.(ReactiveMP.clamplog.(dist.a)) .- digamma.(sum(ReactiveMP.clamplog.(dist.a)); dims = 1)
#mean(::typeof(ReactiveMP.clamplog), dist::PointMass) = ReactiveMP.clamplog.(mean(dist))


Distributions.entropy(dist::ContingencyTensor) = -sum(xlogx.(dist.p))

@marginalrule TransitionMixture(:out_in_z) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_z::DiscreteNonParametric, q_B1::PointMass, q_B2::PointMass, q_B3::PointMass, q_B4::PointMass, ) = begin

    I_ = length(probvec(m_out))
    J  = length(probvec(m_in))
    K  = length(probvec(m_z))
    μ_out = probvec(m_out)
    μ_in = probvec(m_in)
    μ_z = probvec(m_z)

    # Need to make this generic
    A_tilde = [mean(ReactiveMP.clamplog,q_B1);;; mean(ReactiveMP.clamplog,q_B2);;; mean(ReactiveMP.clamplog,q_B3);;; mean(ReactiveMP.clamplog,q_B4)]

    B = cat(map( x -> x * μ_out * μ_in', μ_z)..., dims=3) .* A_tilde
    return ContingencyTensor(B ./ sum(B))
end


#@marginalrule TransitionMixture(:out_in_z) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_z::DiscreteNonParametric, q_B1::PointMass, q_B2::PointMass, q_B3::PointMass, q_B4::PointMass, ) = begin
#
#    I_ = length(m_out)
#    J = length(m_in)
#    K = length(m_z)
#    μ_out = mean(m_out)
#    μ_in = mean(m_in)
#    μ_z = mean(m_z)
#
#    A_tilde = [mean(ReactiveMP.clamplog,q_B1);;; mean(ReactiveMP.clamplog,q_B2);;; mean(ReactiveMP.clamplog,q_B3);;; mean(ReactiveMP.clamplog,q_B4)]
#    B = zeros(I_,J,K)
#    for i in 1:I_
#        for j in 1:J
#            for k in 1:K
#                B[i,j,k] = μ_out[i]*μ_in[j]*μ_z[k] * A_tilde[i,j,k]
#            end
#        end
#    end
#    return ContingencyTensor(B ./ sum(B))
#end
