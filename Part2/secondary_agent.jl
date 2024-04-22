function constructSecondaryPriors()
    eps = 0.1
    B_0 = eps*ones(2, L)
end

function initializeSecondaryAgent(B_0)
    iterations = 50 # Iterations of variational algorithm

    B_s = deepcopy(B_0)
    function infer_prime(t::Int64, a::Union{Int64, Missing}, o::Union{Vector, Missing})
        # Define possible policies
        G = Vector{Float64}(undef, L)
        if t === 1
            pols = 1:L
        elseif t === 2
            pols = a # Register move
        end
        
        # Define (un)observed data for meta objects
        x = deepcopy(o)
    
        for i in pols
            α = αs[i]
            u = zeros(L)
            u[i] = 1.0 # One-hot policy

            # Define model
            model = t_maze_secondary(B_s, x, u)

            # Utility depends on offer
            C = softmax((1-α)*[c, -c])
            
            data = (c = C,)
    
            constraints = structured(t<2)

            initmarginals = (B = MatrixDirichlet(asym(B_s)),)
    
            res = inference(model         = model,
                            data          = data,
                            constraints   = constraints,
                            initmarginals = initmarginals,
                            iterations    = iterations,
                            free_energy   = true)
                        
            G[i] = mean(res.free_energy[10:iterations])./log(2) # Average to smooth fluctuations and convert to bits
            # G[i] = res.free_energy[end]/log(2) # Convert to bits
            if t === 2 # Return posterior statistics after learning
                B_s = res.posteriors[:B][end].a
            end
        end
    
        return (G, B_s)
    end

    function act_prime(G)
        p = softmax(-10.0*G) # Sharpen for minimum selection (fixed precision)
        pol = rand(Categorical(p)) # Select a policy
        
        return pol # Select from possible actions
    end

    return (infer_prime, act_prime)
end
;