function initializeSecondaryWorld()
    # Secondary execute plays interaction of primary agent with environment
    function execute_prime(s::Int64, a_prime::Int64)
        as = Vector{Int64}(undef, 2) # Primary actions per time
        os = Vector{Vector}(undef, 2) # Primary observations (one-hot) per time
        reset(s, a_prime) 
        for t=1:2
              G_t = infer(t, as, os, a_prime)
            as[t] = act(t, G_t)
                    execute(as[t])
            os[t] = observe()
        end

        a_s = deepcopy(as)
    end

    a_s = Vector{Int64}(undef, 2)
    function observe_prime() # Accepts executed action sequence by primary agent
        # Secondary agent observes CV if primary agent has tried to visit the cue position (NC otherwise)
        if 4 in a_s # Offer is made when action to cue is proposed
                  # CV   NC
            return [1.0, 0.0]
        else
            return [0.0, 1.0]
        end
    end

    return (execute_prime, observe_prime)
end
;