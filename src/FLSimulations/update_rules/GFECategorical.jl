import ForneyLab: collectSumProductNodeInbounds, collectNaiveVariationalNodeInbounds

using ForneyLab: isClamped, assembleClamp!
using ForwardDiff: jacobian

export ruleVBGFECategoricalOut, ruleVBGFECategoricalOut

@sumProductRule(:node_type     => GFECategorical,
                :outbound_type => Message{Categorical},
                :inbound_types => (Nothing, Message{PointMass}, Message{PointMass}),
                :name          => SPGFECategoricalOutDPP)

function ruleSPGFECategoricalOutDPP(msg_out::Message{Categorical, Univariate},
                                    marg_out::Distribution{Univariate, Categorical},
                                    msg_A::Message{PointMass, MatrixVariate},
                                    msg_c::Message{PointMass, Multivariate};
                                    n_iterations=10)
    d = msg_out.dist.params[:p]
    s_0 = marg_out.params[:p]
    A = msg_A.dist.params[:m]
    c = msg_c.dist.params[:m]

    rho = msgGFECategoricalOut(d, s_0, A, c, n_iterations)

    Message(Univariate, Categorical, p=rho)
end

function collectSumProductNodeInbounds(node::GFECategorical, entry::ScheduleEntry)
    algo = currentInferenceAlgorithm()
    interface_to_schedule_entry = algo.interface_to_schedule_entry
    target_to_marginal_entry = algo.target_to_marginal_entry

    inbounds = Any[]
    for node_interface in entry.interface.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        if node_interface === entry.interface
            # Collect inbound message
            push!(inbounds, interface_to_schedule_entry[inbound_interface])
            # Collect inbound marginal
            push!(inbounds, target_to_marginal_entry[node_interface.edge.variable])
        elseif isClamped(inbound_interface)
            # Hard-code outbound message of constant node in schedule
            push!(inbounds, assembleClamp!(inbound_interface.node, Message))
        else
            # Collect message from previous result
            push!(inbounds, interface_to_schedule_entry[inbound_interface])
        end
    end

    # Push custom arguments if defined
    if (node.n_iterations !== nothing)
        push!(inbounds, Dict{Symbol, Any}(:n_iterations => node.n_iterations,
                                          :keyword      => true))
    end
    
    return inbounds
end

@naiveVariationalRule(:node_type     => GFECategorical,
                      :outbound_type => Message{Categorical},
                      :inbound_types => (Nothing, Distribution, Distribution),
                      :name          => VBGFECategoricalOut)

function ruleVBGFECategoricalOut(msg_out::Message{Categorical, Univariate},
                                 marg_out::Distribution{Univariate, Categorical},
                                 marg_A::Distribution{MatrixVariate, PointMass},
                                 marg_c::Distribution{Multivariate, PointMass};
                                 n_iterations=10)
    d = msg_out.dist.params[:p]
    s_0 = marg_out.params[:p]
    A = marg_A.params[:m]
    c = marg_c.params[:m]

    rho = msgGFECategoricalOut(d, s_0, A, c, n_iterations)

    Message(Univariate, Categorical, p=rho)
end

function collectNaiveVariationalNodeInbounds(node::GFECategorical, entry::ScheduleEntry)
    algo = currentInferenceAlgorithm()
    interface_to_schedule_entry = algo.interface_to_schedule_entry
    target_to_marginal_entry = algo.target_to_marginal_entry

    inbounds = Any[]
    for node_interface in entry.interface.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        if node_interface === entry.interface
            # Collect inbound message
            push!(inbounds, interface_to_schedule_entry[inbound_interface])
            # Collect inbound marginal
            push!(inbounds, target_to_marginal_entry[node_interface.edge.variable])
        elseif isClamped(inbound_interface)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, assembleClamp!(inbound_interface.node, Distribution))
        else
            # Collect entry from marginal schedule
            push!(inbounds, target_to_marginal_entry[node_interface.edge.variable])
        end
    end

    # Push custom arguments if defined
    if (node.n_iterations !== nothing)
        push!(inbounds, Dict{Symbol, Any}(:n_iterations => node.n_iterations,
                                          :keyword      => true))
    end
    
    return inbounds
end

function msgGFECategoricalOut(d::Vector, s_0::Vector, A::Matrix, c::Vector, n_iterations::Int64)
    # Root-finding problem for marginal statistics
    g(s) = s - softmax(log.(d .+ tiny) + diag(A'*log.(A .+ tiny)) + A'*log.(c .+ tiny) - A'*log.(A*s .+ tiny))

    s_k_min = s_0
    for k=1:n_iterations
        s_k = s_k_min - inv(jacobian(g, s_k_min))*g(s_k_min) # Newton step for multivariate root finding    
        s_k_min = s_k
    end
    s_k = s_k_min

    # Compute outbound message statistics
    rho = s_k./(d .+ tiny)
    return rho./sum(rho)
end