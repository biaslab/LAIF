function constructPrimaryABCD(α::Float64, c::Float64)
    # Observation model
    #      P1
    #      RL       RR
    #      CV  NC   CV  NC
    A_1 = [0.5 0.5  0.5 0.5;
           0.5 0.5  0.5 0.5;
           0.0 0.0  0.0 0.0;
           0.0 0.0  0.0 0.0]

    #      P2
    #      RL       RR
    #      CV  NC   CV  NC
    A_2 = [0.0 0.0  0.0 0.0;
           0.0 0.0  0.0 0.0;
           α   1.0  1-α 0.0;
           1-α 0.0  α   1.0]

    #      P3
    #      RL       RR
    #      CV  NC   CV  NC
    A_3 = [0.0 0.0  0.0 0.0;
           0.0 0.0  0.0 0.0;
           1-α 0.0  α   1.0;
           α   1.0  1-α 0.0]

    #      P4
    #      RL       RR
    #      CV  NC   CV  NC
    A_4 = [1.0 1.0  0.0 0.0;
           0.0 0.0  1.0 1.0;
           0.0 0.0  0.0 0.0;
           0.0 0.0  0.0 0.0]

    A = zeros(16, 16)
    A[1:4, 1:4]     = A_1
    A[5:8, 5:8]     = A_2
    A[9:12, 9:12]   = A_3
    A[13:16, 13:16] = A_4

    # Transition model (with forced move back after reward-arm visit)
    B_1 = kron([1 1 1 1; # Row: can I move to 1?
                0 0 0 0;
                0 0 0 0;
                0 0 0 0], I(4))

    B_2 = kron([0 1 1 0; 
                1 0 0 1; # Row: can I move to 2?
                0 0 0 0;
                0 0 0 0], I(4))

    B_3 = kron([0 1 1 0;
                0 0 0 0;
                1 0 0 1; # Row: can I move to 3?
                0 0 0 0], I(4))

    #      P1           P2 ...
    #      RL    RR        
    #      CV NC CV NC
    B_4 = [0  0  0  0   1  0  0  0   1  0  0  0   0  0  0  0;
           0  0  0  0   0  1  0  0   0  1  0  0   0  0  0  0;
           0  0  0  0   0  0  1  0   0  0  1  0   0  0  0  0;
           0  0  0  0   0  0  0  1   0  0  0  1   0  0  0  0;

           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0;
           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0;
           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0;
           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0;

           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0;
           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0;
           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0;
           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0;

           1  1  0  0   0  0  0  0   0  0  0  0   1  1  0  0; # Move to World with visited cue
           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0;
           0  0  1  1   0  0  0  0   0  0  0  0   0  0  1  1; # Move to World with visited cue
           0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0]

    B = [B_1, B_2, B_3, B_4]

    # Goal prior
    C = softmax(kron(ones(4), [0.0, 0.0, c, -c]))

    # Initial state prior
    #    P1                   P2 ...
    #    RL        RR        
    #    CV   NC   CV   NC   
    D = [0.0, 0.5, 0.0, 0.5,  0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0]

    return (A, B, C, D)
end

function initializePrimaryAgent(A, B, C, D)
    iterations = 50 # Iterations of variational algorithm
    function infer(t::Int64, a::Vector, o::Vector)
        # Define possible policies
        G = Matrix{Union{Float64, Missing}}(missing, 4, 4)
        if t === 1
            pols = [(1,1), (1,2), (1,3), (1,4), (2,1), (3,1), (4,1), (4,2), (4,3), (4,4)]
        elseif t === 2
            a1 = a[1] # Register first move
            if a1 in [2, 3]
                pols = [(a1,1)] # Mandatory move to 1
            else
                pols = [(a1,1), (a1,2), (a1,3), (a1,4)]
            end
        elseif t === 3
            a1 = a[1] # Register both moves
            a2 = a[2]
            pols = [(a1, a2)]
        end
    
        # Define (un)observed data for meta objects
        x = Vector{Union{Vector{Float64}, Missing}}(missing, 2)
        for k=1:2
            if isassigned(o, k) # Observed datapoint
                x[k] = o[k]
            end
        end
    
        # Define model
        model = t_maze_primary(A_s, D, x) # Observation matrix depends on offer
        
        for (i, j) in pols
            data = (u = [B[i], B[j]],
                    c = [C, C])
    
            initmarginals = (z_0 = Categorical(asym(16)),
                             z   = [Categorical(asym(16)),
                                    Categorical(asym(16))])
    
            res = inference(model         = model,
                            data          = data,
                            initmarginals = initmarginals,
                            iterations    = iterations,
                            free_energy   = true)
            
            G[i, j] = res.free_energy[end]/log(2) # Convert to bits
        end
    
        return G
    end
    
    function act(t, G)
        # We include policy selection in the act function for clearer code; procedurally, policy selection belongs in the plan step
        idx = findall((!).(ismissing.(G))) # Find coordinates of non-missing entries
        Gvec = G[idx] # Convert to vector of valid entries
        p = softmax(-10.0*Gvec) # Sharpen for minimum selection
        pol = rand(Categorical(p)) # Select a policy
        
        return idx[pol][t] # Select current action from policy
    end

    A_s = deepcopy(A)
    function execute(a::Int64) # Processes action (offer) from secondary agent
        αs = [0.85, 0.9, 0.95]

        # Set A matrix of primary agent
        (A_s, _, _, _) = initializePrimaryABCD(αs[a], 2.0)

        return A_s
    end

    function observe(a::Vector{Int64}) # Accepts executed action sequence by primary agent
        # Secondary agent observes CV if primary agent has tried to visit the cue position (NC otherwise)
        if 4 in a # Offer is made when action to cue is proposed
                  # CV   NC
            return [1.0, 0.0]
        else
            return [0.0, 1.0]
        end
    end

    return (infer, act)
end