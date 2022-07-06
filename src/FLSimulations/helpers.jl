using LaTeXStrings

function plotResults(F; dpi=100, clim=(4.0,8.0), title="", highlight=nothing)
    p = heatmap(F,
            dpi=dpi,
            color=:grays, 
            aspect_ratio=:equal, 
            xlim=(0.5,4.5), 
            ylim=(0.5,4.5), 
            xlabel=L"\mathrm{Second\,Move\,} (\hat{u}_2)",
            ylabel=L"\mathrm{First\,Move\,} (\hat{u}_1)",
            title=title,
            clim=clim,
            xtickfontsize=12,
            ytickfontsize=12,
            xguidefontsize=14,
            yguidefontsize=14,
            size=(500,430))

    F_round = round.(F, digits=2)
    if highlight != nothing
        extremum = highlight(F_round)
    else
        extremum = NaN
    end

    for i=1:4
        for j=1:4
            # Annotate number
            if F[i,j] >= clim[2]-0.3
                colour = :black
            else
                colour = :white
            end

            if extremum == F_round[i,j]
                ann = (j, i, text("$(F_round[i,j])*", 15, :red, :center)) # Annotate extremum
            else
                ann = (j, i, text(F_round[i,j], 15, colour, :center))
            end
            annotate!(ann, linecolor=colour)
        end
    end

    return p
end

function plotReward(alphas, cs, R; dpi=100, clim=(0.0,1.0), title="")
    R_mean = mean.(R)

    p = plot(cs,
             alphas, 
             R_mean,
             st=:contour,
             fill=true,
             dpi=dpi,
             xlabel="c",
             ylabel=L"\alpha",
             title=title,
             clim=clim,
             xtickfontsize=12,
             ytickfontsize=12,
             xguidefontsize=14,
             yguidefontsize=14,
             size=(500,450))

    return p
end

function annotateActions(p, alphas, cs, P; dpi=100, title="")
    P_unique = unique.(P)
    
    J = length(alphas)
    K = length(cs)

    for j=1:J
        for k=1:K
            ann = (cs[k], alphas[j], text(join(P_unique[j,k], "\n"), 5, :red, :center)) # Annotate policies
            annotate!(ann, linecolor=:red)
        end
    end

    return p
end