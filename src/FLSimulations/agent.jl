using ForneyLab: softmax #, unsafeMean, unsafeMode

function evaluatePolicies(A, B, C, D)
    #C_t = [C, C]
    C_t = [ones(16)/16, C]

    n_its = 20

    # Evaluate all policies
    F = zeros(4,4)
    for i in 1:4  # First move
        for j = 1:4  # Second move
            data = Dict(:u       => [B[i], B[j]],
                        :A       => A,
                        :C       => C_t,
                        :D_t_min => D)

            marginals = Dict{Symbol, ProbabilityDistribution}(
                :x_1 => ProbabilityDistribution(Univariate, Categorical, p=ones(8)./8),
                :x_2 => ProbabilityDistribution(Univariate, Categorical, p=ones(8)./8))
        
            for i=1:n_its
                step!(data, marginals)
            end
        
            F[i, j] = freeEnergy(data, marginals)
        end
    end

    return F./log(2) # Convert to bits
end
