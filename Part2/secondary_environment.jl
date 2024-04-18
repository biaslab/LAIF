# Secondary World offers a simplified representation of the primary agent
function initializeSecondaryWorld(αs)
    theta = 0.91 # Offer-acceptance threshold
    function execute(a_s::Int64) # Determine acceptance of offer a_s
        if αs[a_s] > theta
            #       CV NC
            x_s = [1, 0]
        else
            x_s = [0, 1]
        end
    end

    #      CV NC
    x_s = [0, 1]
    observe() = x_s

    return (execute, observe)
end