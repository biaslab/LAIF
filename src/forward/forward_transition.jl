"""
Contains modified rules for the transition node to block backwards flow of messages.
Useful for replicating the standard AIF algorithm in Reactive
"""

struct ForwardOnlyMeta end

# Incoming edges
@rule Transition(:in, Marginalisation) (q_out::Any, q_a::MatrixDirichlet, meta::ForwardOnlyMeta) = begin
    return missing
end

@rule Transition(:in, Marginalisation) (m_out::Categorical, q_a::MatrixDirichlet, meta::ForwardOnlyMeta) = begin
    return missing
end

@rule Transition(:in, Marginalisation) (m_out::Categorical, q_a::PointMass, meta::ForwardOnlyMeta) = begin
    return missing
end

@rule Transition(:in, Marginalisation) (q_out::PointMass, q_a::PointMass, meta::ForwardOnlyMeta) = begin
    return missing
end

@rule Transition(:in, Marginalisation) (m_out::Categorical, m_a::PointMass, meta::ForwardOnlyMeta) = begin
    return missing
end

@rule Transition(:in, Marginalisation) (m_out::Missing, q_a::PointMass, meta::ForwardOnlyMeta) = begin
    return missing
end

@marginalrule Transition(:out_in) (m_out::Missing, m_in::DiscreteNonParametric, q_a::PointMass, meta::ForwardOnlyMeta) =
 begin
    return missing
end


# Outgoing edges
@rule Transition(:out, Marginalisation) (q_in::Categorical, q_a::MatrixDirichlet, meta::ForwardOnlyMeta) = begin
    return @call_rule Transition(:out, Marginalisation) (q_in = q_in, q_a = q_a)
end

@rule Transition(:out, Marginalisation) (m_in::Categorical, q_a::MatrixDirichlet, meta::ForwardOnlyMeta) = begin
    return @call_rule Transition(:out, Marginalisation) (m_in = m_in, q_a = q_a)
end

@rule Transition(:out, Marginalisation) (m_in::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    return @call_rule Transition(:out, Marginalisation) (m_in = m_in, m_a = q_a, meta = meta)
end

@rule Transition(:out, Marginalisation) (q_in::PointMass, q_a::PointMass, meta::ForwardOnlyMeta) = begin
   return @call_rule Transition(:out, Marginalisation) (q_in = q_in, q_a = q_a)
end

@rule Transition(:out, Marginalisation) (m_in::Categorical, m_a::PointMass, meta::ForwardOnlyMeta) = begin
    return @call_rule Transition(:out, Marginalisation) (m_in = m_in, m_a = m_a)
end


# Edge towards transition matrix
@rule Transition(:a, Marginalisation) (q_out::Any, q_in::Categorical, meta::ForwardOnlyMeta) = begin
    return @call_rule Transition(:a, Marginalisation) (q_out = q_out, q_in = q_in)
end

# TODO: Fix this rule
#@rule Transition(:a, Marginalisation) (q_out_in::Contingency, meta::ForwardOnlyMeta) = begin
#    return @call_rule Transition(:a, Marginalisation) (q_out_in = q_out_in)
#end

# Marginals
@marginalrule Transition(:out_in) (m_out::Categorical, m_in::Categorical, q_a::MatrixDirichlet, meta::ForwardOnlyMeta) = begin
    return @call_marginalrule Transition(:out_in) (m_out = m_out, m_in = m_in, q_a = q_a)
end

@marginalrule Transition(:out_in) (m_out::Categorical, m_in::Categorical, q_a::PointMass, meta::ForwardOnlyMeta) = begin
    return @call_marginalrule Transition(:out_in) (m_out = m_out, m_in = m_in, q_a = q_a)
end

@marginalrule Transition(:out_in_a) (m_out::Categorical, m_in::Categorical, m_a::PointMass, meta::ForwardOnlyMeta) = begin
    return @call_marginalrule Transition(:out_in_a) (m_out = m_out, m_in = m_in, m_a = m_a)
end
