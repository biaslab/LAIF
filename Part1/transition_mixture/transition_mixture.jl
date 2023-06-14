# ContingencyTensor is defined here
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


Distributions.entropy(dist::ContingencyTensor) = -sum(xlogx.(dist.p))

struct TransitionMixture end

@node TransitionMixture Stochastic [out,in,z,B1,B2,B3,B4,]

@average_energy TransitionMixture (q_out_in_z::ContingencyTensor, q_B1::PointMass, q_B2::PointMass, q_B3::PointMass, q_B4::PointMass) = begin

    # Need to make this generic
    log_A_bar = [mean(ReactiveMP.clamplog,q_B1);;; mean(ReactiveMP.clamplog,q_B2);;; mean(ReactiveMP.clamplog,q_B3);;; mean(ReactiveMP.clamplog,q_B4)]

    B = mean(q_out_in_z)

    sum(-tr.(transpose.(eachslice(B,dims=3)) .* eachslice(log_A_bar,dims=3)))
end

# Used when input state is clamped
@average_energy TransitionMixture (q_out_z::ContingencyTensor,q_in::PointMass, q_B1::PointMass, q_B2::PointMass, q_B3::PointMass, q_B4::PointMass) = begin

    # Need to make this generic
    log_A_bar = [mean(ReactiveMP.clamplog,q_B1);;; mean(ReactiveMP.clamplog,q_B2);;; mean(ReactiveMP.clamplog,q_B3);;; mean(ReactiveMP.clamplog,q_B4)]

    B = mean(q_out_z)

    sum(-tr.(transpose.(eachslice(B,dims=3)) .* eachslice(log_A_bar,dims=3)))
end
