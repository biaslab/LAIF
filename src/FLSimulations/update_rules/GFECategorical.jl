import ForneyLab: collectSumProductNodeInbounds

using ForneyLab: isClamped, assembleClamp!

export ruleVBGFECategoricalOut

@sumProductRule(:node_type     => GFECategorical,
                :outbound_type => Message{Categorical},
                :inbound_types => (Nothing, Message{PointMass}, Message{PointMass}),
                :name          => SPGFECategoricalOutDPP)

function ruleSPGFECategoricalOutDPP(marg_out::Distribution{Univariate, Categorical}, 
                                    msg_A::Message{PointMass, MatrixVariate},
                                    msg_c::Message{PointMass, Multivariate})

    s = marg_out.params[:p]
    A = msg_A.dist.params[:m]
    c = msg_c.dist.params[:m]

    rho = diag(A'*log.(A .+ tiny)) + A'*log.(c .+ tiny) - A'*log.(A*s .+ tiny)

    Message(Univariate, Categorical, p=rho./sum(rho))
end

function collectSumProductNodeInbounds(::GFECategorical, entry::ScheduleEntry)
    algo = currentInferenceAlgorithm()
    interface_to_schedule_entry = algo.interface_to_schedule_entry
    target_to_marginal_entry = algo.target_to_marginal_entry

    inbounds = Any[]
    for node_interface in entry.interface.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        if node_interface === entry.interface
            # Collect entry from marginal schedule
            push!(inbounds, target_to_marginal_entry[node_interface.edge.variable])
        elseif isClamped(inbound_interface)
            # Hard-code outbound message of constant node in schedule
            push!(inbounds, assembleClamp!(inbound_interface.node, Message))
        else
            # Collect message from previous result
            push!(inbounds, interface_to_schedule_entry[inbound_interface])
        end
    end

    return inbounds
end