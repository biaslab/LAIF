# Alias for safe logarithm
const safelog = ReactiveMP.clamplog

# Softmax that plays nice with automatic differentiation
function softmax(v::AbstractVector)
    r = clamp.(v .- maximum(v), -100, 0.0)
    exp.(r)./sum(exp.(r))
end

# Symmetry breaking for vague Categorical statistics
function asym(n::Int64)
    p = ones(n) .+ 1e-3*rand(n)
    return p./sum(p)
end

# Symmetry breaking for matrix Dirichlet statistics
asym(A::Matrix) = A + 1e-2*rand(size(A)...)
