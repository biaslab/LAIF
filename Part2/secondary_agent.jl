function initializeSecondaryAgent(B_0, C, αs)
    iterations = 50 # Iterations of variational algorithm
    L = length(αs)

    B_s = deepcopy(B_0)
    function infer(t::Int64, a::Union{Int64, Missing}, o::Union{Vector, Missing})
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

            # Observation matrix depends on offer
            eps = 1e-4
            #    CV  NC
            A = [1-α eps; # RW
                 α   1-eps] # NR

            # Define model
            model = t_maze_secondary(A, B_s, x, u)

            data = (c = C,)
    
            constraints = structured()

            initmarginals = (B = MatrixDirichlet(asym(B_s)),
                             z = vague(Categorical, 2))
    
            res = inference(model         = model,
                            data          = data,
                            constraints   = constraints,
                            initmarginals = initmarginals,
                            iterations    = iterations,
                            free_energy   = true)
                        
            # G[i] = mean(res.free_energy[10:iterations])./log(2) # Average to smooth fluctuations and convert to bits
            G[i] = res.free_energy[end]/log(2) # Convert to bits
            if t === 2 # Return posterior statistics after learning
                println(round.(res.free_energy, digits=2))
                B_s = res.posteriors[:B][end].a
            end
        end
    
        return (G, B_s)
    end

    function act(G)
        p = softmax(-10.0*G) # Sharpen for minimum selection (fixed precision)
        pol = rand(Categorical(p)) # Select a policy
        
        return pol # Select from possible actions
    end

    return (infer, act)
end