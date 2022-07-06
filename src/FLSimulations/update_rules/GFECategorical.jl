export ruleVBGFECategoricalOut

@naiveVariationalRule(:node_type     => GFECategorical,
                      :outbound_type => Message{Categorical},
                      :inbound_types => (Nothing, Distribution),
                      :name          => VBGFECategoricalOut)

function ruleVBGFECategoricalOut(marg_out::Distribution{Univariate, Categorical}, 
                                 marg_A::Distribution{MatrixVariate, PointMass},
                                 marg_c::Distribution{Multivariate, PointMass})

    s = marg_out.params[:p]
    A = marg_A.params[:m]
    c = marg_c.params[:m]

    rho = diag(A'*log.(A .+ tiny)) + A'*log.(c .+ tiny) - A'*log.(A*s .+ tiny)

    Message(Univariate, Categorical, p=rho./sum(rho))
end

function collectNaiveVariationalNodeInbounds(::GFECategorical, entry::ScheduleEntry)
    target_to_marginal_entry = current_inference_algorithm.target_to_marginal_entry

    inbounds = Any[]
    for node_interface in entry.interface.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        if isClamped(inbound_interface)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, assembleClamp!(inbound_interface.node, Distribution))
        else
            # Collect entry from marginal schedule
            push!(inbounds, target_to_marginal_entry[node_interface.edge.variable])
        end
    end

    return inbounds
end


