using LaTeXStrings

function plotFreeEnergies(Gt::Vector, at::Vector, ot::Vector, r::Int64)
    gtsvec = skipmissing([vec(Gts[s][1]); Gts[s][2]])
    min = floor(minimum(gtsvec))
    max = ceil(maximum(gtsvec))

    p1 = plotG1(Gts[s][1], at, ot, r, clim=(min,max+1))
    p2 = plotG2(Gts[s][2], clim=(min,max+1))

    plot(p1, p2, layout=grid(2,1,heights=[0.8,0.2]), size=(300, 400))
end

function plotG1(F::Matrix, at::Vector, ot::Vector, r::Int64; 
                dpi=100, clim=(4.0,8.0), title="", highlight=minimum)
    p = heatmap(F,
            dpi=dpi,
            color=:grays, 
            aspect_ratio=:equal,
            colorbar=false,
            xlim=(0.5,4.5), 
            ylim=(0.5,4.5), 
            ylabel=L"\mathrm{First\,Policy\,} (\pi_1)",
            title=title,
            clim=clim,
            xticks=false,
            xtickfontsize=12,
            ytickfontsize=12,
            xguidefontsize=14,
            yguidefontsize=14)

    F_round = round.(F, digits=1)
    if highlight !== nothing
        extremum = highlight(skipmissing(F_round))
    else
        extremum = NaN
    end

    for i=1:4
        for j=1:4
            ismissing(F[i,j]) && continue

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

    correct = (at .== r) # Correctly visited
    correct_dict = Dict{Bool, String}(true  => "●",
                                      false => "○")
    txt = join([correct_dict[c_t] for c_t in correct], " ")
    ann = (3, 2.5, text(txt, 18, :gold, :center))
    annotate!(ann, linecolor=:gold)

    ann = (3, 2.45, text(join(at, "   "), 13, :black, :center)) # Executed actions
    annotate!(ann, linecolor=:black)

    ct = [Int(ot[t][13:14]'*[2, 3]) for t=1:2] # Received cues
    cue_dict = Dict{Int, String}(0 => " ",
                                 2 => "←",
                                 3 => "→")
    ann = (3, 2.1, text(join([cue_dict[ct[t]] for t=1:2], "  "), 18, :gold, :center))
    annotate!(ann, linecolor=:gold)

    return p
end

function plotG2(F::Vector; dpi=100, clim=(4.0,8.0), title="", highlight=minimum)
    F = reshape(F,1,4)
    p = heatmap(F,
            dpi=dpi,
            color=:grays, 
            aspect_ratio=:equal,
            colorbar=false,
            xlim=(0.5,4.5), 
            ylim=(0.5,1.5), 
            xlabel=L"\mathrm{Second\,Policy\,} (\pi_2)",
            title=title,
            clim=clim,
            yticks=false,
            xtickfontsize=12,
            ytickfontsize=12,
            xguidefontsize=14,
            yguidefontsize=14)

    F_round = round.(F, digits=1)
    if highlight !== nothing
        extremum = highlight(skipmissing(F_round))
    else
        extremum = NaN
    end

    for j=1:4
        ismissing(F[j]) && continue

        # Annotate number
        if F[j] >= clim[2]-0.3
            colour = :black
        else
            colour = :white
        end
        
        if extremum == F_round[j]
            ann = (j, 1, text("$(F_round[j])*", 15, :red, :center)) # Annotate extremum
        else
            ann = (j, 1, text(F_round[j], 15, colour, :center))
        end
        annotate!(ann, linecolor=colour)
    end

    return p
end
