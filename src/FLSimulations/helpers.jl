using StatsFuns: gammainvcdf, loggamma
import ForneyLab: ruleSPEqualityFnFactor, sample, logPdf, sampleWeightsAndEntropy, sample, unsafeMean, unsafeLogMean

# Edit: add tiny to x
logPdf(dist::Distribution{MatrixVariate, Dirichlet}, x) = sum(sum((dist.params[:a].-1).*log.(x .+ tiny),dims=1) - sum(loggamma.(dist.params[:a]), dims=1) + loggamma.(sum(dist.params[:a],dims=1)))

# Custom update that outputs a Function message as result o6f Dirichlet-Function message product (instead of SampleList)
ruleSPEqualityFnFactor(msg_1::Message{<:Function}, msg_2::Message{<:Dirichlet}, msg_3::Nothing) = Message(prodDirFn!(msg_1.dist, msg_2.dist))
ruleSPEqualityFnFactor(msg_1::Message{<:Function}, msg_2::Nothing, msg_3::Message{<:Dirichlet}) = Message(prodDirFn!(msg_1.dist, msg_3.dist))
ruleSPEqualityFnFactor(msg_1::Nothing, msg_2::Message{<:Function}, msg_3::Message{<:Dirichlet}) = Message(prodDirFn!(msg_2.dist, msg_3.dist))
ruleSPEqualityFnFactor(msg_1::Message{<:Dirichlet}, msg_2::Message{<:Function}, msg_3::Nothing) = Message(prodDirFn!(msg_2.dist, msg_1.dist))
ruleSPEqualityFnFactor(msg_1::Message{<:Dirichlet}, msg_2::Nothing, msg_3::Message{<:Function}) = Message(prodDirFn!(msg_3.dist, msg_1.dist))
ruleSPEqualityFnFactor(msg_1::Nothing, msg_2::Message{<:Dirichlet}, msg_3::Message{<:Function}) = Message(prodDirFn!(msg_3.dist, msg_2.dist))

prodDirFn!(dist_fn::Distribution{MatrixVariate, Function}, dist_dir::Distribution{MatrixVariate, Dirichlet}) =
    Distribution(MatrixVariate, Function, log_pdf=(A)->logPdf(dist_dir, A)+dist_fn.params[:log_pdf](A))

# Edit number of default samples
function sampleWeightsAndEntropy(x::Distribution, y::Distribution)
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

function sample(dist::Distribution{MatrixVariate, Dirichlet})
    A = similar(dist.params[:a])
    for i = 1:size(A)[2]
        A[:,i] = sample(Distribution(Multivariate, Dirichlet, a=dist.params[:a][:,i]))
    end
    return A
end

function unsafeMean(dist::Distribution{MatrixVariate, SampleList})
    sum = zeros(size(dist.params[:s][1]))
    for i=1:length(dist.params[:s])
        sum = sum .+ dist.params[:s][i].*dist.params[:w][i]
    end
    return sum
end

function unsafeLogMean(dist::Distribution{MatrixVariate, SampleList})
    sum = zeros(size(dist.params[:s][1]))
    for i=1:length(dist.params[:s])
        sum = sum .+ log.(dist.params[:s][i] .+ tiny).*dist.params[:w][i]
    end
    return sum
end

import ForneyLab: softmax, tiny

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