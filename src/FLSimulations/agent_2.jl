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

function constructAPrior()
    eps = 0.1

    A_0_4 = [1.0 0.0;
             0.0 1.0;
             0.0 0.0;
             0.0 0.0]

    A_0 = eps*ones(16, 8)
    A_0[13:16, 7:8] = A_0_4 .+ eps # Hint that position 4 resolves information

    return A_0
end

function initializeAgent(A_0, B, C, D)
    D_t_min = deepcopy(D)
    n_its = 10
    A_s = deepcopy(A_0)
    function infer(t::Int64, a::Vector, o::Vector)
        # Evaluate all policies
        if t == 1
            G = zeros(4,4)
            for i in 1:4  # First move
                for j = 1:4  # Second move
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
                                            
                    Gs = zeros(n_its)
                    for k=1:n_its
                        stept1X!(data, marginals, messages)
                        stept1A!(data, marginals)
                        Gs[k] = freeEnergyt1(data, marginals)
                    end
                    Gs = Gs./log(2) # Convert to bits                

                    G[i, j] = mean(Gs[5:n_its]) # Average to smooth fluctuations
                end
            end

            # Disallow illegal moves
            G[2,2] = G[2,3] = G[2,4] = 100.0
            G[3,2] = G[3,3] = G[3,4] = 100.0

            return G./log(2) # Return free energy in bits
        elseif t == 2
            G = zeros(4)
            for j = 1:4  # Second move
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
                    Gs[k] = freeEnergyt2(data, marginals)
                end
                Gs = Gs./log(2) # Convert to bits                

                G[j] = mean(Gs[5:n_its]) # Average to smooth fluctuations
            end

            # Disallow illegal move
            if a[1] in [2,3]
                G[2] = G[3] = G[4] = 100.0
            end

            return G./log(2) # Return free energy in bits
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
            A_s = deepcopy(marginals[:A].params[:a]) # Reset for next simulation

            return A_s # Return posterior statistics
        end
    end

    function act(G)
        # We include policy selection in the act function for clearer code; procedurally, policy selection belongs in the plan step
        p = softmax(vec(-100*G)) # Determine policy probabilities with high precision (max selection)
        S = reshape(sample(ProbabilityDistribution(Categorical, p=p)), size(G)) # Reshaped policy sample
        (_, pol) = findmax(S)

        return pol[1] # Return first action of policy
    end

    return (infer, act)
end