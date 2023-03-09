using ForwardDiff: jacobian
using DomainSets: FullSpace

import Base: prod

# Helper functions
# We don't want log(0) to happen
safelog(x) = log(clamp(x,tiny,Inf))
#normalize(x) = x ./ sum(x)
softmax(x) = exp.(x) ./ sum(exp.(x))

# Meta object to replicate the classic EFE schedule
struct ForwardOnlyMeta end

struct GFECategorical end
@node GFECategorical Stochastic [out,in,A]

@average_energy GFECategorical (q_out::PointMass, q_in::Categorical,q_A::PointMass,) = begin
    s = probvec(q_in)
    A = mean(q_A)
    c = probvec(q_out)

    -s' * diag(A' * safelog.(A)) + (A*s)'* safelog.(A*s) - (A*s)'*safelog.(c)
end


@rule GFECategorical(:in, Marginalisation) (m_out::PointMass, q_out::PointMass,m_in::DiscreteNonParametric, q_in::DiscreteNonParametric, m_A::PointMass, q_A::PointMass,) = begin

    z = probvec(q_in)
    d = probvec(m_in)
    A = mean(q_A)

    # We use the goal prior on an edge here
    C = probvec(q_out)

    # Newton iterations for stability
    g(s) = s - softmax(safelog.(d) + diag(A' * safelog.(A)) + A' *(safelog.(C) - safelog.(A * s)))

    z_min = z
    for i in 1:20 # TODO make this user specified
        z_k = z_min - inv(jacobian(g,z_min)) * g(z_min)
        z_min = z_k
    end
    z_k = z_min

    ρ = z_k ./ (d .+ tiny)
    return Categorical(ρ ./ sum(ρ))
end

@rule GFECategorical(:in, Marginalisation) (m_out::PointMass, q_out::PointMass,m_in::DiscreteNonParametric, q_in::DiscreteNonParametric, m_A::MatrixDirichlet, q_A::MatrixDirichlet,) = begin
    @call_rule GFECategorical(:in, Marginalisation) (m_out = m_out, q_out = q_out, m_in = m_in, q_in = q_in, m_A = PointMass(mean(m_A)), q_A = PointMass(mean(q_A)),)
end

# Block backwards rule if using ForwardOnlyMeta. Replicates standard EFE schedule
@rule GFECategorical(:in, Marginalisation) (m_out::PointMass, q_out::PointMass,m_in::DiscreteNonParametric, q_in::DiscreteNonParametric, m_A::PointMass, q_A::PointMass, meta::ForwardOnlyMeta) = begin
    return missing
end



# Message towards A
import Distributions.rand



@rule GFECategorical(:A,Marginalisation) (m_out::PointMass, q_out::PointMass,m_in::DiscreteNonParametric, q_in::DiscreteNonParametric, m_A::MatrixDirichlet, q_A::MatrixDirichlet,) = begin
    A_bar = mean(q_A)
    c = probvec(m_out)
    s = probvec(m_in)

    # LogPdf
    logpdf(A) = s' *( diag(A'*safelog.(A)) + A'*(safelog.(c) - safelog.(A_bar*s)))
    return ContinuousMatrixvariateLogPdf(FullSpace(),logpdf)

end


# Draw sample from a Matrixvariate Dirichlet distribution with independent rows
# TODO: Find a way to not create a bunch of intermediate Dirichlet distributions
function rand(dist::MatrixDirichlet)
    α = clamp.(dist.a,tiny,Inf)
    # Sample from independent rowwise Dirichlet distributions and replace NaN's with 0.0.
    # We replace NaN's to allow for rows of 0's which are common in AIF modelling
    replace!(reduce(hcat, Distributions.rand.(Dirichlet.(eachrow(α))))', NaN => 0.0)
end


# We need a product of MatrixVariate Logpdf's and MatrixDirichlet to compute the marginal over the transition matrix. We approximate it using EVMP (Add citation to Semihs paper)
# TODO: Is this really ProdAnalytical?
# TODO: Check that the DomainSpace is right. Replace with "Unspecified"
import Base: prod
prod(::ProdAnalytical, left::MatrixDirichlet{Float64, Matrix{Float64}}, right::ContinuousMatrixvariateLogPdf{FullSpace{Float64}}) = begin
    _logpdf = right.logpdf

    # Draw 50 samples
    weights = []
    samples = []
    for n in 1:50
        A_hat = rand(left)
        ρ_n = exp(_logpdf(A_hat))

        push!(samples, A_hat)
        push!(weights, ρ_n)
    end
    #Z = sum(weights)
    #return MatrixDirichlet(sum(samples .* weights) / Z)
    return SampleList(samples,weights)
end