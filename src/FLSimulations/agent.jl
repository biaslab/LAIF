import ForneyLab: softmax, tiny

function softmax(v::Vector)
    r = v .- maximum(v)
    clamp!(r, -100.0, 0.0)
    exp.(r)./sum(exp.(r))
end

function evaluatePoliciesGBFE(A, B, C, D,n_its)
    # Evaluate all policies
    F = zeros(4,4)
    for i in 1:4  # First move
        for j = 1:4  # Second move
            data = Dict(:u       => [B[i], B[j]],
                        :A       => A,
                        :C       => [C, C],
                        :D_t_min => D)

            marginals = Dict{Symbol, ProbabilityDistribution}(
                :x_1 => ProbabilityDistribution(Univariate, Categorical, p=ones(8)./8),
                :x_2 => ProbabilityDistribution(Univariate, Categorical, p=ones(8)./8))
        
            messages = init()

            #n_its = 10
            for k=1:n_its
                step!(data, marginals, messages)
            end
        
            F[i, j] = freeEnergy(data, marginals)
        end
    end

    return F./log(2) # Convert to bits
end

function evaluatePoliciesEFE(A, B, C, D)
    # Construct priors
    C_t = [C, C]
    D_t_min = D
    
    # Evaluate all policies
    Q = zeros(4,4)
    for i in 1:4 # First move
        x_t_hat = B[i]*D_t_min # Expected state
        y_t_hat = A*x_t_hat # Expected outcome

        # We follow Eq. D.2 in da Costa (2021) "Active inference on discrete state-spaces: a synthesis"
        predicted_uncertainty_t = diag(A' * log.(A .+ tiny))' * x_t_hat # Friston (for reference): ones(16)' * (A.*log.(A .+ tiny)) * x_t_hat
        predicted_divergence_t = transpose( log.(y_t_hat .+ tiny) - log.(C_t[1] .+ tiny) ) * y_t_hat
        Q_t = predicted_uncertainty_t - predicted_divergence_t

        for j in 1:4 # Second move
            x_t_plus_hat = B[j]*x_t_hat # Expected state
            y_t_plus_hat = A*x_t_plus_hat # Expected outcome

            predicted_uncertainty_t_plus = diag(A' * log.(A .+ tiny))' * x_t_plus_hat
            predicted_divergence_t_plus = transpose( log.(y_t_plus_hat .+ tiny) - log.(C_t[2] .+ tiny) ) * y_t_plus_hat
            Q_t_plus = predicted_uncertainty_t_plus - predicted_divergence_t_plus

            Q[i, j] = Q_t + Q_t_plus
        end
    end

    return -Q./log(2) # Return expected free energy per policy in bits
end

function evaluatePoliciesGMFE(A, B, C, D)
    # Evaluate all policies
    F = zeros(4,4)
    for i in 1:4  # First move
        for j = 1:4  # Second move
            data = Dict(:u       => [B[i], B[j]],
                        :A       => A,
                        :C       => [C, C],
                        :D_t_min => D)

            marginals = Dict{Symbol, ProbabilityDistribution}(
                :x_1 => ProbabilityDistribution(Univariate, Categorical, p=ones(8)./8),
                :x_2 => ProbabilityDistribution(Univariate, Categorical, p=ones(8)./8))
        
            n_its = 10
            for k=1:n_its
                stepX0!(data, marginals)
                stepX1!(data, marginals)
                stepX2!(data, marginals)
            end
        
            F[i, j] = freeEnergy(data, marginals)
        end
    end

    return F./log(2) # Convert to bits
end

function evaluatePoliciesBFE(A, B, C, D)
    # Evaluate all policies
    F = zeros(4,4)
    for i in 1:4  # First move
        for j = 1:4  # Second move
            data = Dict(:u       => [B[i], B[j]],
                        :A       => A,
                        :C       => [C, C],
                        :D_t_min => D)

            marginals = step!(data)

            F[i, j] = freeEnergy(data, marginals)
        end
    end

    return F./log(2) # Convert to bits
end
