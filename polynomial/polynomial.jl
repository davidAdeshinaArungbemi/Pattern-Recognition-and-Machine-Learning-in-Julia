using Revise, StatsPlots, Random, Distributions

# Random.seed!(24)

𝐱 = collect(range(start=Float32(0), stop=Float32(1), length=10)); #vector of x values

𝐲 = sin.(2 * 𝐱 .* pi) + rand(Normal(0, 0.2), length(𝐱)); #vector of actual output, add noise for more realism

degree = 3

𝐰 = rand(Normal(0, 30), degree + 1); # +1 for bias

function polyfit(𝐱, 𝐭)
    global 𝐰 #to have access to global variable
    η = 0.2 #learning rate

    for (x, t) in zip(𝐱, 𝐭)
        y = sum([w .* x^(j - 1) for (j, w) in enumerate(𝐰)]) # prediction

        𝚫E_𝚫y = y - t #derivative of error with respect to output y
        𝚫y_𝚫W = [x^(j - 1) for (j, _) in enumerate(𝐰)]#derivative of output y with respect to weights
        𝚫E_𝚫W = 𝚫E_𝚫y .* 𝚫y_𝚫W #derivative of error with respect to weights
        𝐰 -= η .* 𝚫E_𝚫W #weight update
    end

end

# polyfit(𝐱, 𝐲)

polypred = (𝐱, 𝐰) -> [sum(w .* (x^(j - 1)) for (j, w) in enumerate(𝐰)) for x in 𝐱]

anim = @gif for t in 1:500
    polyfit(𝐱, 𝐲)

    scatter(𝐱, 𝐲,
        title="Polynomial Curve Fitting",
        markersize=5,
        label="Actual",
    )

    plot!(
        𝐱,
        polypred(𝐱, 𝐰),
        label="Degree = $(degree)",
        legend=true
    )

    annotate!(0.8, 0.5, "Epoch: $(t)")
end

# gif(anim, "animation.gif", fps=15)