function constructPriors()
    eps = 0.1
    
    # Strong prior on position 1
    A_0_1 = [10.0 10.0;
             10.0 10.0;
             eps  eps;
             eps  eps]

    A_0 = eps*ones(16, 8) # Vague prior on everything else

    A_0[1:4, 1:2] = A_0_1

    # Agent knows it starts at position 1
    D_0 = zeros(8)
    D_0[1:2] = [0.5, 0.5]

    return (A_0, D_0)
end

function initializeAgent(A_0, B, C, D_0)
    n_its = 50 # Iterations of variational algorithm
    A_s = deepcopy(A_0)
    D_s = deepcopy(D_0)
    function infer(t::Int64, a::Vector, o::Vector)
        # Define possible policies
        G = Matrix{Union{Float64, Missing}}(undef, 4, 4)
        if t == 0
            pols = [(1,1), (1,2), (1,3), (1,4), (2,1), (3,1), (4,1), (4,2), (4,3), (4,4)]
        elseif t == 1
            a1 = a[1] # Register first move
            if a1 in [2, 3]
                pols = [(a1,1)] # Mandatory move to 1
            else
                pols = [(a1,1), (a1,2), (a1,3), (a1,4)]
            end
        elseif t == 2
            a1 = a[1] # Register both moves
            a2 = a[2]
            pols = [(a1, a2)]
        end

        for (i, j) in pols
            data = Dict(:u   => [B[i], B[j]],
                        :A_s => A_s,
                        :C   => C,
                        :D_s => D_s)

            marginals = Dict{Symbol, ProbabilityDistribution}(
                :x_0 => ProbabilityDistribution(Univariate, Categorical, p=asym(8)),
                :x_1 => ProbabilityDistribution(Univariate, Categorical, p=asym(8)),
                :x_2 => ProbabilityDistribution(Univariate, Categorical, p=asym(8)),
                :A => ProbabilityDistribution(MatrixVariate, Dirichlet, a=asym(A_s)))
            
            # Define (un)observed marginals
            for k=1:2
                if isassigned(o, k)
                    # Observed
                    marginals[:y_*k] = Distribution(Multivariate, PointMass, m=o[k])
                else
                    # Unobserved
                    marginals[:y_*k] = Distribution(Univariate, Categorical, p=asym(16))
                end
            end

            messages = initX()
                                    
            Gis = zeros(n_its)
            for i=1:n_its
                stepX!(data, marginals, messages)
                stepA!(data, marginals)
                stepY!(data, marginals)
                Gis[i] = freeEnergy(data, marginals)
            end

            G[i, j] = mean(Gis[10:n_its])./log(2) # Average to smooth fluctuations and convert to bits
            if t == 2 # Update posterior statistics after learning
                A_s = deepcopy(marginals[:A].params[:a])
            end
        end

        return (G, A_s)
    end

    function act(t, G)
        # We include policy selection in the act function for clearer code; procedurally, policy selection belongs in the plan step
        idx = findall((!).(ismissing.(G))) # Find coordinates of non-missing entries
        Gvec = G[idx] # Convert to vector of valid entries
        p = softmax(-100.0*Gvec)
        s = sample(ProbabilityDistribution(Categorical, p=p)) # Sample a 1-of-K representation
        c = first(idx[s.==1.0]) # Select coordinate (policy) by sample
        
        return c[t+1] # Return current action
    end

    return (infer, act)
end