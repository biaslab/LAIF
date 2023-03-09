#Barebones implementation of matrixvariate logpdf for learning parameters with GFECategoricals
export ContinuousMatrixvariateLogPdf

using Distributions
using ReactiveMP
#
import DomainSets
using ReactiveMP: AbstractContinuousGenericLogPdf

struct ContinuousMatrixvariateLogPdf{D <: DomainSets.Domain, F} <: AbstractContinuousGenericLogPdf
    domain::D
    logpdf::F

    #ContinuousMatrixvariateLogPdf(domain::D, logpdf::F) where {D, F} = begin
    #    @assert DomainSets.dimension(domain) == 1 "Cannot create ContinuousMatrixvariateLogPdf. Dimension of domain = $(domain) is not equal to 1."
    #    return new{D, F}(domain, logpdf)
    #end
end

variate_form(::Type{<:ContinuousMatrixvariateLogPdf}) = Matrixvariate
variate_form(::ContinuousMatrixvariateLogPdf)         = Matrixvariate

promote_variate_type(::Type{Matrixvariate}, ::Type{AbstractContinuousGenericLogPdf}) = ContinuousMatrixvariateLogPdf

getdomain(dist::ContinuousMatrixvariateLogPdf) = dist.domain
getlogpdf(dist::ContinuousMatrixvariateLogPdf) = dist.logpdf

ContinuousMatrixvariateLogPdf(f::Function) = ContinuousMatrixvariateLogPdf(DomainSets.FullSpace(), f)

Base.show(io::IO, dist::ContinuousMatrixvariateLogPdf) = print(io, "ContinuousMatrixvariateLogPdf(", getdomain(dist), ")")
Base.show(io::IO, ::Type{<:ContinuousMatrixvariateLogPdf{D}}) where {D} = print(io, "ContinuousMatrixvariateLogPdf{", D, "}")

Distributions.support(dist::ContinuousMatrixvariateLogPdf) = Distributions.RealInterval(DomainSets.infimum(getdomain(dist)), DomainSets.supremum(getdomain(dist)))

# Fallback for various optimisation packages which may pass arguments as vectors
function Distributions.logpdf(dist::ContinuousMatrixvariateLogPdf, x::AbstractVector{<:Real})
    #@assert length(x) === 1 "`ContinuousMatrixvariateLogPdf` expects either float or a vector of a single float as an input for the `logpdf` function."
    return logpdf(dist, first(x))
end

Base.convert(::Type{<:ContinuousMatrixvariateLogPdf}, domain::D, logpdf::F) where {D <: DomainSets.Domain, F} = ContinuousMatrixvariateLogPdf(domain, logpdf)

convert_eltype(::Type{ContinuousMatrixvariateLogPdf}, ::Type{T}, dist::ContinuousMatrixvariateLogPdf) where {T <: Real} = convert(ContinuousMatrixvariateLogPdf, dist.domain, dist.logpdf)

#vague(::Type{<:ContinuousMatrixvariateLogPdf}) = ContinuousMatrixvariateLogPdf(DomainSets.FullSpace(), (x) -> 1.0)

# We do not check typeof of a different functions because in most of the cases lambdas have different types, but they can still be the same
function is_typeof_equal(::ContinuousMatrixvariateLogPdf{D, F1}, ::ContinuousMatrixvariateLogPdf{D, F2}) where {D, F1 <: Function, F2 <: Function}
    return true
end
