
import ReactiveMP: AbstractFormConstraint

struct EpistemicProduct <: AbstractFormConstraint end

ReactiveMP.make_form_constraint(::Type{EpistemicProduct}) = EpistemicProduct()

ReactiveMP.is_point_mass_form_constraint(::EpistemicProduct) = false
ReactiveMP.default_form_check_strategy(::EpistemicProduct)   = FormConstraintCheckLast()
ReactiveMP.default_prod_constraint(::EpistemicProduct)       = ProdGeneric()

function ReactiveMP.constrain_form(::EpistemicProduct, distribution)
    error("Unexpected")
end

function ReactiveMP.constrain_form(::EpistemicProduct, distribution::DistProduct)
    return distribution.left[:μ]::Categorical
end


