using DomainSets: Domain
using StatsFuns: gammainvcdf, loggamma
using ReactiveMP: AbstractContinuousGenericLogPdf, GenericLogPdfVectorisedProduct, UnspecifiedDomain, approximate_prod_with_sample_list
using RxInfer: AutoProposal, SampleListFormConstraint
using Random

import Base: prod, rand, eltype, size
import Distributions: logpdf, mean
import Random: rand, rand!
import ReactiveMP: getdomain, getlogpdf
import RxInfer: __approximate


h(A) = -diag(A'*safelog.(A))

mean_h(d::PointMass) = (d.point, h(d.point))


#------------------------------
# ContinuousMatrixvariateLogPdf
#------------------------------

struct ContinuousMatrixvariateLogPdf{D <: Domain, F} <: AbstractContinuousGenericLogPdf
    domain::D
    logpdf::F
end

ContinuousMatrixvariateLogPdf(f::Function) = ContinuousMatrixvariateLogPdf(UnspecifiedDomain(), f)

getdomain(d::ContinuousMatrixvariateLogPdf) = d.domain
getlogpdf(d::ContinuousMatrixvariateLogPdf) = d.logpdf


#-----------
# SampleList
#-----------

function __approximate(constraint::SampleListFormConstraint{N, R, S, M}, left::ContinuousMatrixvariateLogPdf, right) where {N, R, S <: AutoProposal, M}
    return approximate_prod_with_sample_list(constraint.rng, constraint.method, right, left, N)
end

function __approximate(constraint::SampleListFormConstraint{N, R, S, M}, left, right::ContinuousMatrixvariateLogPdf) where {N, R, S <: AutoProposal, M}
    return approximate_prod_with_sample_list(constraint.rng, constraint.method, left, right, N)
end

function __approximate(constraint::SampleListFormConstraint{N, R, S, M}, left::GenericLogPdfVectorisedProduct, right) where {N, R, S <: AutoProposal, M}
    return approximate_prod_with_sample_list(constraint.rng, constraint.method, right, left, N)
end

function __approximate(constraint::SampleListFormConstraint{N, R, S, M}, left, right::GenericLogPdfVectorisedProduct) where {N, R, S <: AutoProposal, M}
    return approximate_prod_with_sample_list(constraint.rng, constraint.method, left, right, N)
end

# These are hacks to make _rand! work with matrix variate logpfds
eltype(::GenericLogPdfVectorisedProduct) = Float64
eltype(::ContinuousMatrixvariateLogPdf) = Float64

function mean_h(d::SampleList)
    s = get_samples(d)
    w = get_weights(d)

    return (sum(s.*w), sum(h.(s).*w))
end


#----------------
# MatrixDirichlet
#----------------

size(d::MatrixDirichlet) = size(d.a)

function logpdf(d::MatrixDirichlet, x::AbstractMatrix)
    return sum(sum((d.a.-1).*log.(x),dims=1) - sum(loggamma.(d.a), dims=1) + loggamma.(sum(d.a,dims=1)))
end

# Average energy definition for SampleList marginal
@average_energy MatrixDirichlet (q_out::SampleList, q_a::PointMass) = begin
    H = mapreduce(+, zip(eachcol(mean(q_a)), eachcol(mean(log, q_out)))) do (q_a_column, logmean_q_out_column)
        return -loggamma(sum(q_a_column)) + sum(loggamma.(q_a_column)) - sum((q_a_column .- 1.0) .* logmean_q_out_column)
    end
    return H
end

# In-place operations for sampling
function rand!(rng::AbstractRNG, d::MatrixDirichlet, container::Array{Float64, 3})
    s = size(d)
    for i in 1:size(container, 3)
        M = view(container, :, :, i)
        sample = rand(rng, d)
        copyto!(M, sample)
    end

    return container
end

# Custom sampling implementation
function rand(rng::AbstractRNG, d::MatrixDirichlet)
    U = rand(rng, size(d.a)...)
    S = gammainvcdf.(d.a, 1.0, U)
    return S./sum(S, dims=1) # Normalize columns
end

function mean_h(d::MatrixDirichlet)
    n_samples = 20 # Fixed number of samples
    s = [rand(d) for i=1:n_samples]

    return (sum(s)./n_samples, sum(h.(s))./n_samples)
end