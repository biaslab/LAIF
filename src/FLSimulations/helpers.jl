using StatsFuns: gammainvcdf, loggamma
import ForneyLab: ruleSPEqualityDirichlet, sample, logPdf, sampleWeightsAndEntropy, sample, unsafeMean, unsafeLogMean, VariateType, differentialEntropy, softmax, tiny

differentialEntropy(::Distribution{<:VariateType, PointMass}) = 0.0 # Define entropy of pointmass as zero

# Edit: add tiny to x
logPdf(dist::Distribution{MatrixVariate, Dirichlet}, x) = sum(sum((dist.params[:a].-1).*log.(x .+ tiny),dims=1) - sum(loggamma.(dist.params[:a]), dims=1) + loggamma.(sum(dist.params[:a],dims=1)))

# Custom update that outputs a Function message as result of Dirichlet-Function message product
ruleSPEqualityDirichlet(msg_1::Message{<:Function}, msg_2::Message{<:Dirichlet}, msg_3::Nothing) = Message(prodDirFn!(msg_1.dist, msg_2.dist))
ruleSPEqualityDirichlet(msg_1::Message{<:Function}, msg_2::Nothing, msg_3::Message{<:Dirichlet}) = Message(prodDirFn!(msg_1.dist, msg_3.dist))
ruleSPEqualityDirichlet(msg_1::Nothing, msg_2::Message{<:Function}, msg_3::Message{<:Dirichlet}) = Message(prodDirFn!(msg_2.dist, msg_3.dist))
ruleSPEqualityDirichlet(msg_1::Message{<:Dirichlet}, msg_2::Message{<:Function}, msg_3::Nothing) = Message(prodDirFn!(msg_2.dist, msg_1.dist))
ruleSPEqualityDirichlet(msg_1::Message{<:Dirichlet}, msg_2::Nothing, msg_3::Message{<:Function}) = Message(prodDirFn!(msg_3.dist, msg_1.dist))
ruleSPEqualityDirichlet(msg_1::Nothing, msg_2::Message{<:Dirichlet}, msg_3::Message{<:Function}) = Message(prodDirFn!(msg_3.dist, msg_2.dist))

prodDirFn!(dist_fn::Distribution{MatrixVariate, Function}, dist_dir::Distribution{MatrixVariate, Dirichlet}) =
    Distribution(MatrixVariate, Function, log_pdf=(A)->logPdf(dist_dir, A)+dist_fn.params[:log_pdf](A))

ruleSPEqualityDirichlet(msg_1::Message{<:Function}, msg_2::Message{<:Function}, msg_3::Nothing) = Message(prod!(msg_1.dist, msg_2.dist))
ruleSPEqualityDirichlet(msg_1::Message{<:Function}, msg_2::Nothing, msg_3::Message{<:Function}) = Message(prod!(msg_1.dist, msg_3.dist))
ruleSPEqualityDirichlet(msg_1::Nothing, msg_2::Message{<:Function}, msg_3::Message{<:Function}) = Message(prod!(msg_2.dist, msg_3.dist))

# Edit number of default samples
function sampleWeightsAndEntropy(x::Distribution, y::Distribution{<:VariateType, <:Function})
    n_samples = 10 # Number of samples is fixed
    samples = sample(x, n_samples)

    # Apply log-pdf functions to the samples
    log_samples_x = logPdf.([x], samples)
    log_samples_y = logPdf.([y], samples)

    # Extract the sample weights
    w_raw = exp.(log_samples_y) # Unnormalized weights
    w_sum = sum(w_raw)
    weights = w_raw./w_sum # Normalize the raw weights

    # Compute the separate contributions to the entropy
    H_y = log(w_sum) - log(n_samples)
    H_x = -sum( weights.*(log_samples_x + log_samples_y) )
    entropy = H_x + H_y

    # Inform next step about the proposal and integrand to be used in entropy calculation in smoothing
    logproposal = (samples) -> logPdf.([x], samples)
    logintegrand = (samples) -> logPdf.([y], samples)

    return (samples, weights, w_raw, logproposal, logintegrand, entropy)
end

# Helper function to prevent log of 0
safelog(x) = log(clamp(x,tiny,Inf))

# Ambiguity weight vector
amb(A) = -diag(A'*safelog.(A))

function sample(dist::Distribution{MatrixVariate, Dirichlet})
    A = dist.params[:a]
    U = rand(size(A)...)
    S = gammainvcdf.(A, 1.0, U)

    return S./sum(S, dims=1) # Normalize columns
end

function unsafeLogMean(dist::Distribution{MatrixVariate, SampleList})
    sum = zeros(size(dist.params[:s][1]))
    for i=1:length(dist.params[:s])
        sum = sum .+ log.(dist.params[:s][i] .+ tiny).*dist.params[:w][i]
    end
    return sum
end

unsafeAmbMean(dist::Distribution{MatrixVariate, PointMass}) = amb(dist.params[:m])

function unsafeAmbMean(dist::Distribution{MatrixVariate, Dirichlet})
    n_samples = 10 # Number of samples is fixed
    s = sample(dist, n_samples)
    sum(amb.(s))./n_samples
end

unsafeAmbMean(dist::Distribution{MatrixVariate, SampleList}) = sum(amb.(dist.params[:s]).*dist.params[:w])

function softmax(v::Vector)
    r = v .- maximum(v)
    clamp!(r, -100.0, 0.0)
    exp.(r)./sum(exp.(r))
end

# Symmetry breaking for initial statistics
function asym(n::Int64)
    p = ones(n) .+ 1e-3*rand(n)
    return p./sum(p)
end

asym(A::Matrix) = A + 1e-2*rand(size(A)...)