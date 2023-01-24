function constructAPrior()
    eps = 0.01
    
    # A_0_X = [1.5 0.5;
    #          0.5 1.5;
    #          1.0 1.0;
    #          1.0 1.0] # Sum of probabilities in A
             
    A_0_4 = [10.0 eps;
             eps  10.0;
             eps  eps;
             eps  eps]

    A_0 = eps*ones(16, 8)

    # A_0[1:4, 1:2] = A_0_X
    # A_0[5:8, 3:4] = A_0_X
    # A_0[9:12, 5:6] = A_0_X
    A_0[13:16, 7:8] = A_0_4

    return A_0
end

function initializeAgent(A_0, B, C, D)
    D_t_min = deepcopy(D)
    n_its = 10
    A_s = deepcopy(A_0)
    function infer(t::Int64, a::Vector, o::Vector)
        # Evaluate all policies
        if t == 1
            G = Matrix{Union{Float64, Missing}}(undef, 4, 4)
            risk = Matrix{Union{Float64, Missing}}(undef, 4, 4)
            amb = Matrix{Union{Float64, Missing}}(undef, 4, 4)
            nov = Matrix{Union{Float64, Missing}}(undef, 4, 4)
            pols = [(1,1), (1,2), (1,3), (1,4), (2,1), (3,1), (4,1), (4,2), (4,3), (4,4)]
            for (i, j) in pols
                data = Dict(:u       => [B[i], B[j]],
                            :A_s     => A_s,
                            :C       => C,
                            :D_t_min => D_t_min)

                marginals = Dict{Symbol, ProbabilityDistribution}(
                    :x_t_min => ProbabilityDistribution(Univariate, Categorical, p=D),
                    :x_1 => ProbabilityDistribution(Univariate, Categorical, p=asym(8)),
                    :x_2 => ProbabilityDistribution(Univariate, Categorical, p=asym(8)),
                    :A => ProbabilityDistribution(MatrixVariate, Dirichlet, a=asym(A_s)))
                
                messages = initt1X()
                                        
                Gks = zeros(n_its)
                riskks = zeros(n_its)
                ambks = zeros(n_its)
                novks = zeros(n_its)
                for k=1:n_its
                    stept1X!(data, marginals, messages)
                    stept1A!(data, marginals)
                    Gks[k] = freeEnergyt1(data, marginals)/log(2) # Convert to bits
                    (riskks[k], ambks[k], novks[k]) = freeEnergyDecompt1(data, marginals) # Already returned in bits
                end

                G[i, j] = mean(Gks[5:n_its]) # Average to smooth fluctuations
                risk[i, j] = mean(riskks[5:n_its])
                amb[i, j] = mean(ambks[5:n_its])
                nov[i, j] = mean(novks[5:n_its])
            end

            return (G, risk, amb, nov) # Return free energy
        elseif t == 2
            G = Vector{Union{Float64, Missing}}(undef, 4)
            if a[1] in [2, 3]
                pols = [1] # Mandatory move to 1
            else
                pols = [1, 2, 3, 4]
            end

            for j in pols # Second move
                data = Dict(:u       => [B[a[1]], B[j]],
                            :y       => o,
                            :A_s     => A_s,
                            :C       => C,
                            :D_t_min => D_t_min)

                marginals = Dict{Symbol, ProbabilityDistribution}(
                    :x_t_min => ProbabilityDistribution(Univariate, Categorical, p=D),
                    :x_1 => ProbabilityDistribution(Univariate, Categorical, p=asym(8)),
                    :x_2 => ProbabilityDistribution(Univariate, Categorical, p=asym(8)),
                    :A => ProbabilityDistribution(MatrixVariate, Dirichlet, a=asym(A_s)))
                
                messages = initt2X()
                                        
                Gs = zeros(n_its)
                for k=1:n_its
                    stept2X!(data, marginals, messages)
                    stept2A!(data, marginals)
                    Gs[k] = freeEnergyt2(data, marginals)/log(2) # Convert to bits
                    Gs[k] += averageEnergy(Categorical, 
                                           Distribution(Multivariate, PointMass, m=o[1]),
                                           Distribution(Multivariate, PointMass, m=C))/log(2)
                end

                G[j] = mean(Gs[5:n_its]) # Average to smooth fluctuations
            end

            return G
        elseif t == 3
            data = Dict(:u       => [B[a[1]], B[a[2]]],
                        :y       => o,
                        :A_s     => A_s,
                        :C       => C,
                        :D_t_min => D_t_min)

            marginals = Dict{Symbol, ProbabilityDistribution}(
                :x_t_min => ProbabilityDistribution(Univariate, Categorical, p=D),
                :x_1 => ProbabilityDistribution(Univariate, Categorical, p=asym(8)),
                :x_2 => ProbabilityDistribution(Univariate, Categorical, p=asym(8)),
                :A => ProbabilityDistribution(MatrixVariate, Dirichlet, a=asym(A_s)))

            for k=1:n_its
                stept3X!(data, marginals)
                stept3A!(data, marginals)
            end
            G = freeEnergyt3(data, marginals)/log(2) # Convert to bits
            G += averageEnergy(Categorical, 
                               Distribution(Multivariate, PointMass, m=o[1]),
                               Distribution(Multivariate, PointMass, m=C))/log(2)
            G += averageEnergy(Categorical, 
                               Distribution(Multivariate, PointMass, m=o[2]),
                               Distribution(Multivariate, PointMass, m=C))/log(2)
            A_s = deepcopy(marginals[:A].params[:a]) # Reset for next simulation

            return (G, A_s) # Return free energy and posterior statistics
        end
    end

    function act(G)
        # We include policy selection in the act function for clearer code; procedurally, policy selection belongs in the plan step
        idx = findall((!).(ismissing.(G))) # Find coordinates of non-missing entries
        Gvec = G[idx] # Convert to vector of valid entries
        p = softmax(-100.0*Gvec)
        s = sample(ProbabilityDistribution(Categorical, p=p)) # Sample a 1-of-K representation
        c = first(idx[s.==1.0]) # Select coordinate (policy) by sample
        
        return (c[1], c) # Return first action from policy
    end

    return (infer, act)
end