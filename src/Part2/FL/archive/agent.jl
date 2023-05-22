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

function evaluatePoliciesGBFE(A, B, C_t, D; n_its=10)
    # Evaluate all policies
    G = zeros(4,4)
    for i in 1:4  # First move
        for j = 1:4  # Second move
            data = Dict(:u       => [B[i], B[j]],
                        :A       => A,
                        :C       => C_t,
                        :D_t_min => D)

            marginals = Dict{Symbol, ProbabilityDistribution}(
                :x_t_min => ProbabilityDistribution(Univariate, Categorical, p=D),
                :x_1 => ProbabilityDistribution(Univariate, Categorical, p=asym(8)),
                :x_2 => ProbabilityDistribution(Univariate, Categorical, p=asym(8)))
        
            messages = init()

            for k=1:n_its
                step!(data, marginals, messages)
            end
        
            G[i, j] = freeEnergy(data, marginals)
        end
    end

    return G./log(2) # Convert to bits
end

# Evaluation includes parameter estimate
function evaluatePoliciesFullGBFE(A, B, C, D; n_its=10)
    # Evaluate all policies
    G = zeros(4,4)
    for i in 1:4  # First move
        for j = 1:4  # Second move
            data = Dict(:u       => [B[i], B[j]],
                        :A       => A,
                        :C       => [C, C],
                        :D_t_min => D)

            marginals = Dict{Symbol, ProbabilityDistribution}(
                :x_t_min => ProbabilityDistribution(Univariate, Categorical, p=D),
                :x_1 => ProbabilityDistribution(Univariate, Categorical, p=asym(8)),
                :x_2 => ProbabilityDistribution(Univariate, Categorical, p=asym(8)),
                :A_t => ProbabilityDistribution(MatrixVariate, Dirichlet, a=A_0),
                :c_t => ProbabilityDistribution(Multivariate, Dirichlet, a=C_0))
            
            messages = initX()
            
            Gs = zeros(n_its)
            for k=1:n_its
                stepX!(data, marginals, messages)
                stepC!(data, marginals, messages)
                stepA!(data, marginals, messages)

                Gs[k] = freeEnergy(data, marginals)
            end
                                
            G[i, j] = mean(Gs[5:n_its])
        end
    end

    return G./log(2) # Convert to bits
end

function evaluatePoliciesEFE(A, B, C_t, D)
    # Construct priors
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

function evaluatePoliciesGMFE(A, B, C_t, D; n_its=10)
    # Evaluate all policies
    G = zeros(4,4)
    for i in 1:4  # First move
        for j = 1:4  # Second move
            data = Dict(:u       => [B[i], B[j]],
                        :A       => A,
                        :C       => C_t,
                        :D_t_min => D)

            marginals = Dict{Symbol, ProbabilityDistribution}(
                :x_t_min => ProbabilityDistribution(Univariate, Categorical, p=D),
                :x_1 => ProbabilityDistribution(Univariate, Categorical, p=asym(8)),
                :x_2 => ProbabilityDistribution(Univariate, Categorical, p=asym(8)))
        
            for k=1:n_its
                stepX0!(data, marginals)
                stepX1!(data, marginals)
                stepX2!(data, marginals)
            end
        
            G[i, j] = freeEnergy(data, marginals)
        end
    end

    return G./log(2) # Convert to bits
end

function evaluatePoliciesBFE(A, B, C_t, D)
    # Evaluate all policies
    F = zeros(4,4)
    for i in 1:4  # First move
        for j = 1:4  # Second move
            data = Dict(:u       => [B[i], B[j]],
                        :A       => A,
                        :C       => C_t,
                        :D_t_min => D)

            marginals = step!(data)

            F[i, j] = freeEnergy(data, marginals)
        end
    end

    return F./log(2) # Convert to bits
end

function initializeAgent(A, B, C, D)
    n_its = 10
    function plan()
        # Evaluate all policies
        G = zeros(4,4)
        for i in 1:4  # First move
            for j = 1:4  # Second move
                data = Dict(:u       => [B[i], B[j]],
                            :A       => A,
                            :C       => C_t,
                            :D_t_min => D_t_min)

                marginals = Dict{Symbol, ProbabilityDistribution}(
                    :x_t_min => ProbabilityDistribution(Univariate, Categorical, p=D),
                    :x_1 => ProbabilityDistribution(Univariate, Categorical, p=asym(8)),
                    :x_2 => ProbabilityDistribution(Univariate, Categorical, p=asym(8)))
                
                messages = initPlan()
                                        
                Gs = zeros(n_its)
                for k=1:n_its
                    stepPlan!(data, marginals, messages)
                    Gs[k] = freeEnergyPlan(data, marginals)
                end
                Gs = Gs./log(2) # Convert to bits                

                G[i, j] = mean(Gs[5:n_its]) # Average to smooth fluctuations
            end
        end

        return G./log(2) # Return free energy in bits
    end

    function act(G::Matrix{Float64})
        # We include policy selection in the act function for clearer code; procedurally, policy selection belongs in the plan step
        p = softmax(vec(-100*G)) # Determine policy probabilities with high precision (max selection)
        S = reshape(sample(ProbabilityDistribution(Categorical, p=p)), 4, 4) # Reshaped policy sample
        (_, pol) = findmax(S)

        return pol[1] # Return first action of policy
    end

    D_t_min = D
    C_t = [C, C]
    function slide(a_t::Int64, o_t::Vector{Float64})
        # Estimate state
        data = Dict(:B_t     => B[a_t],
                    :A       => A,
                    :o_t     => o_t,
                    :D_t_min => D_t_min)
        marginals = stepSlide!(data)
        D_t_min = ForneyLab.unsafeMean(marginals[:x_t]) # Reset prior state statistics
        
        # Shift goals for next move
        C_t = circshift(C_t, -1)
        C_t[end] = C
    end

    return (plan, act, slide)
end