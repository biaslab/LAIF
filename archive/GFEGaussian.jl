using ForwardDiff: jacobian
using Plots
using ReactiveMP,GraphPPL,Rocket, LinearAlgebra, OhMyREPL, Distributions;
enable_autocomplete_brackets(false);
unicodeplots()

###
TODO:
How to deal with the ratio of Gaussian distributions?
Approximate with moment matching? LaPlace? Something?
###

# Helper functions
# We don't want log(0) to happen
safelog(x) = log(clamp(x,tiny,Inf))
normalize(x) = x ./ sum(x)
softmax(x) = exp.(x) ./ sum(exp.(x))

struct ForwardOnlyMeta end

struct GFEGaussian end
@node GFEGaussian Stochastic [out,in,Σ]

@average_energy GFEGaussian (q_out::PointMass, q_in::MvNormalMeanVariance,q_A::PointMass,) = begin
    s = probvec(q_in)
    A = mean(q_A)
    c = probvec(q_out)

    -s' * diag(A' * safelog.(A)) + (A*s)'* safelog.(A*s) - (A*s)'*safelog.(c)
end


@rule GFEGaussian(:in, Marginalisation) (m_out::MvNormalMeanPrecision, q_out::MvNormalMeanPrecision,m_in::MvNormalMeanPrecision, q_in::MvNormalMeanPrecision,  m_Σ::PointMass,) = begin

    # We use the goal prior on an edge here
    ξ_c,Λ_c = weightedmean_invcov(q_out)

    # incoming message
    ξ_b,Λ_b = weightedmean_invcov(m_in)

    # Incoming marginal
    ξ_s,Λ_s = weightedmean_invcov(q_in)


    return MvNormalMeanVariance(ρ ./ sum(ρ))
end

# Block backwards rule
@rule GFEGaussian(:in, Marginalisation) (m_out::PointMass, q_out::PointMass,m_in::DiscreteNonParametric, q_in::DiscreteNonParametric, m_A::PointMass, q_A::PointMass, meta::ForwardOnlyMeta) = begin
    return missing
end


using LinearAlgebra

function posdef(n)
    bob = rand(n)
    bob * bob' + I(n)
end

input = MvNormalMeanCovariance(zeros(3), posdef(3))
#
μ_s,Σ_s = mean_invcov(input)
dims = size(Σ_s)


cov(MvNormalMeanCovariance(zeros(3),inv(Σ_s -posdef(3))))




g(s) = inv(S_c)*μ_c + inv(S_b + S_s)* =
S_s =


unvec(bob)
bob = MvNormalMeanCovariance(ones(2),diageye(2)*2)
weightedmean_invcov(bob)
mean_invcov(bob)

x_1 = MvNormalMeanCovariance(-rand(2)*10,posdef(2)*20)
x_2 = MvNormalMeanCovariance(rand(2)*4,posdef(2))


n_samples = 3000
storage = rand(x_1,n_samples) ./ rand(x_2,n_samples);
histogram(storage[1,:])
histogram(storage[2,:])

plot(storage[1,:])
plot(storage[2,:])

cov(storage[1,:])
mean(storage[2,:])
cov(storage[2,:])
