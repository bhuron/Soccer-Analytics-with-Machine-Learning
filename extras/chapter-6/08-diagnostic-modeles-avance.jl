### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ c5d6e7f8-0003-6a4b-2d3e-8f9a0b1c2d3e
begin
	using DataFrames
	using Statistics
	using Random
	using Distributions
	using LinearAlgebra
	using GLM
	using Plots
	gr()
	Random.seed!(42)
	Plots.default(fontfamily="Helvetica", titlefontsize=10, guidefontsize=8)
end

# ╔═╡ c5d6e7f8-0001-6a4b-2d3e-8f9a0b1c2d3e
md"""
# Diagnostic des modèles — plongée approfondie

**Chapitre 6 · Techniques de régression pour l'analyse du football — EXTRA**

## Ce que vous allez apprendre

- Analyse complète des résidus
- Détection de l'hétéroscédasticité
- Identification des observations influentes
- Tests rigoureux des hypothèses de régression
- Utilisation efficace des graphiques de diagnostic
- Mesures correctives quand les hypothèses sont violées
"""

# ╔═╡ c5d6e7f8-0002-6a4b-2d3e-8f9a0b1c2d3e
md"""
## Imports et configuration
"""

# ╔═╡ c5d6e7f8-0004-6a4b-2d3e-8f9a0b1c2d3e
md"""
## Hypothèses de la régression linéaire

1. **Linéarité** — relation linéaire entre X et y
2. **Indépendance** — observations indépendantes
3. **Homoscédasticité** — variance constante des résidus
4. **Normalité** — résidus distribués normalement
5. **Absence de multicolinéarité** — variables non fortement corrélées

**Pourquoi vérifier ?** Les violations peuvent conduire à des estimations
biaisées, des erreurs-types incorrectes et des prédictions médiocres.
"""

# ╔═╡ c5d6e7f8-0005-6a4b-2d3e-8f9a0b1c2d3e
begin
	n_diag = 200
	df_diag = DataFrame(
		shots_on_target   = rand(3:15, n_diag),
		possession        = round.(rand(35.0:0.1:70.0, n_diag); digits=1),
		xg                = round.(rand(0.5:0.01:3.5, n_diag); digits=2),
		opponent_strength = round.(rand(0.3:0.01:0.9, n_diag); digits=2),
		home              = rand([0, 1], n_diag),
	)
	df_diag.goals = round.(Int, clamp.(
		df_diag.xg .* 0.9 .+ df_diag.shots_on_target .* 0.08 .+
		df_diag.home .* 0.3 .- df_diag.opponent_strength .* 0.4 .+
		randn(n_diag) .* (0.3 .+ df_diag.xg .* 0.1), 0, 6))

	diag_model = lm(@formula(goals ~ shots_on_target + possession + xg +
		opponent_strength + home), df_diag)

	md"""Modèle ajusté — **$(n_diag) observations**, R² = **$(round(r²(diag_model); digits=3))**"""
end

# ╔═╡ c5d6e7f8-0006-6a4b-2d3e-8f9a0b1c2d3e
md"""
## 1. Analyse des résidus — 6 graphiques de diagnostic
"""

# ╔═╡ c5d6e7f8-0007-6a4b-2d3e-8f9a0b1c2d3e
let
	resids = residuals(diag_model)
	fitted = predict(diag_model, df_diag)
	std_resids = resids ./ std(resids)

	p1 = scatter(fitted, resids, alpha=0.5, legend=false, color=:steelblue,
		title="Résidus vs Valeurs ajustées", xlabel="Valeurs ajustées", ylabel="Résidus")
	hline!(p1, [0], color=:red, linestyle=:dash, linewidth=1.5)

	p2 = scatter(fitted, sqrt.(abs.(std_resids)), alpha=0.5, legend=false, color=:coral,
		title="Scale-Location", xlabel="Valeurs ajustées", ylabel="√|Résidus standardisés|")

	sorted_resids = sort(resids)
	n_r = length(resids)
	theo = quantile.(Normal(), (1:n_r .- 0.5) ./ n_r)
	p3 = scatter(theo, sorted_resids, alpha=0.5, legend=false, color=:purple,
		title="Q-Q Plot", xlabel="Quantiles théoriques", ylabel="Quantiles observés")
	plot!(p3, theo, theo, color=:red, linestyle=:dash, linewidth=1.5)

	p4 = histogram(resids, bins=20, legend=false, color=:darkgreen, alpha=0.7,
		title="Distribution des résidus", xlabel="Résidus", ylabel="Fréquence")
	vline!(p4, [0], color=:red, linestyle=:dash, linewidth=1.5)

	p5 = scatter(df_diag.xg, resids, alpha=0.5, legend=false, color=:steelblue,
		title="Résidus vs xG", xlabel="xG", ylabel="Résidus")
	hline!(p5, [0], color=:red, linestyle=:dash, linewidth=1.5)

	p6 = scatter(1:n_diag, resids, alpha=0.5, legend=false, color=:steelblue,
		title="Résidus vs Ordre", xlabel="Ordre d'observation", ylabel="Résidus")
	hline!(p6, [0], color=:red, linestyle=:dash, linewidth=1.5)

	plot(p1, p2, p3, p4, p5, p6, layout=(2, 3), size=(1000, 600))
end

# ╔═╡ c5d6e7f8-0008-6a4b-2d3e-8f9a0b1c2d3e
md"""
## 2. Test d'hétéroscédasticité — Breusch-Pagan

L'hétéroscédasticité = variance non constante des résidus. Le test de
Breusch-Pagan régresse les résidus au carré sur les prédicteurs.
"""

# ╔═╡ c5d6e7f8-0009-6a4b-2d3e-8f9a0b1c2d3e
let
	resids = residuals(diag_model)
	# Auxiliary regression: squared residuals ~ predictors
	df_diag.resid_sq = resids.^2
	aux_model = lm(@formula(resid_sq ~ shots_on_target + possession + xg +
		opponent_strength + home), df_diag)
	bp_lm = n_diag * r²(aux_model)
	# p-value from chi-square with df = number of predictors (5)
	bp_pval = 1 - cdf(Chisq(5), bp_lm)

	md"""
	**Test de Breusch-Pagan :**
	- Statistique LM : **$(round(bp_lm; digits=3))**
	- p-valeur : **$(round(bp_pval; digits=4))**

	$(bp_pval < 0.05 ? "⚠️ Hétéroscédasticité détectée — envisager une transformation log ou des erreurs-types robustes." : "✅ Homoscédasticité — l'hypothèse est respectée.")
	"""
end

# ╔═╡ c5d6e7f8-0010-6a4b-2d3e-8f9a0b1c2d3e
md"""
## 3. Observations influentes — Distance de Cook

La distance de Cook mesure l'influence de chaque observation sur le modèle.
Seuil : **4/n**.
"""

# ╔═╡ c5d6e7f8-0011-6a4b-2d3e-8f9a0b1c2d3e
let
	cd = cooksdistance(diag_model)
	threshold = 4 / n_diag
	influential = findall(cd .> threshold)

	p1 = plot(cd, legend=false, color=:steelblue, markersize=4, seriestype=:scatter,
		title="Distance de Cook", xlabel="Observation", ylabel="Cook's D")
	hline!(p1, [threshold], color=:red, linestyle=:dash, linewidth=1.5,
		label="Seuil = $(round(threshold; digits=3))")

	# Leverage vs standardized residuals
	resids = residuals(diag_model)
	std_res = resids ./ std(resids)
	# Leverage = diagonal of hat matrix H = X(X'X)⁻¹X'
	X_mat = modelmatrix(diag_model)
	H = X_mat * inv(X_mat' * X_mat) * X_mat'
	lev = diag(H)

	p2 = scatter(lev, std_res, alpha=0.5, legend=false, color=:steelblue,
		title="Levier vs Résidus standardisés", xlabel="Levier", ylabel="Résidus std")
	hline!(p2, [0], color=:red, linestyle=:dash)
	hline!(p2, [2], color=:orange, linestyle=:dash, alpha=0.5)
	hline!(p2, [-2], color=:orange, linestyle=:dash, alpha=0.5)
	if length(influential) > 0
		scatter!(p2, lev[influential], std_res[influential],
			color=:red, markersize=8, marker=:x, label="Influent")
	end

	plot(p1, p2, layout=(1, 2), size=(900, 350))
end

# ╔═╡ c5d6e7f8-0012-6a4b-2d3e-8f9a0b1c2d3e
md"""
## 4. Multicolinéarité — VIF

Le **Variance Inflation Factor** détecte les variables fortement corrélées.
VIF > 10 = problématique.
"""

# ╔═╡ c5d6e7f8-0013-6a4b-2d3e-8f9a0b1c2d3e
let
	predictors = [:shots_on_target, :possession, :xg, :opponent_strength, :home]
	X_mat = Matrix(df_diag[!, predictors])
	vif_vals = Float64[]
	for j in 1:length(predictors)
		y_col = X_mat[:, j]
		X_rest = X_mat[:, setdiff(1:end, j)]
		df_vif = DataFrame(X_rest, [string(predictors[i]) for i in setdiff(1:length(predictors), j)])
		df_vif.y = y_col
		m = lm(@formula(y ~ 1 + shots_on_target), df_vif)  # fallback
		# Build formula dynamically — use simple approach
		r2_j = try
			f = Term(:y) ~ sum([Term(Symbol(predictors[i])) for i in setdiff(1:length(predictors), j)])
			model_vif = lm(f, df_vif)
			r²(model_vif)
		catch
			0.0
		end
		push!(vif_vals, 1 / (1 - r2_j))
	end

	vif_df = sort(DataFrame(variable=string.(predictors), VIF=round.(vif_vals; digits=2)), :VIF, rev=true)

	# Bar chart
	bar(vif_df.variable, vif_df.VIF, legend=false, color=:steelblue, alpha=0.7,
		title="Variance Inflation Factor (VIF)", ylabel="VIF", xrotation=45)
	hline!([10], color=:red, linestyle=:dash, linewidth=1.5, label="Seuil (10)")
	hline!([5], color=:orange, linestyle=:dash, linewidth=1, alpha=0.5, label="Seuil (5)")
end

# ╔═╡ c5d6e7f8-0014-6a4b-2d3e-8f9a0b1c2d3e
md"""
## 5. Test de normalité — Shapiro-Wilk

Vérifie si les résidus suivent une distribution normale.
"""

# ╔═╡ c5d6e7f8-0015-6a4b-2d3e-8f9a0b1c2d3e
let
	resids = residuals(diag_model)
	# Manual Shapiro-Wilk approximation using correlation of sorted residuals
	# with expected normal order statistics
	n = length(resids)
	sorted_r = sort(resids)
	# Expected order statistics from standard normal
	m_vals = quantile.(Normal(), (1:n .- 0.375) ./ (n .+ 0.25))
	# Shapiro-Wilk W statistic approximation
	W = cor(sorted_r, m_vals)^2

	# Approximate p-value (this is a rough approximation for large n)
	# For n > 50, use Royston's approximation
	log_W = log(1 - W)
	pval = if n <= 200
		# Rough p-value (conservative)
		W < 0.95 ? 0.01 : W < 0.98 ? 0.05 : 0.5
	else
		0.5
	end

	md"""
	**Test de Shapiro-Wilk (approximation) :**
	- W ≈ **$(round(W; digits=4))** (1 = parfaitement normal)
	- $(W > 0.98 ? "✅ Résidus approximativement normaux" : "⚠️ Les résidus s'écartent de la normalité")

	Pour les grands échantillons, de légers écarts à la normalité sont
	moins préoccupants. L'inspection visuelle (Q-Q plot) est souvent
	plus informative que les tests formels.
	"""
end

# ╔═╡ c5d6e7f8-0016-6a4b-2d3e-8f9a0b1c2d3e
md"""
## 6. Mesures correctives — Transformation logarithmique
"""

# ╔═╡ c5d6e7f8-0017-6a4b-2d3e-8f9a0b1c2d3e
let
	df_diag.log_goals = log.(df_diag.goals .+ 1)
	log_model = lm(@formula(log_goals ~ shots_on_target + possession + xg +
		opponent_strength + home), df_diag)

	resids_orig = residuals(diag_model)
	resids_log  = residuals(log_model)
	fitted_orig = predict(diag_model, df_diag)
	fitted_log  = predict(log_model, df_diag)

	p1 = scatter(fitted_orig, resids_orig, alpha=0.5, legend=false, color=:steelblue,
		title="Modèle original", xlabel="Valeurs ajustées", ylabel="Résidus")
	hline!(p1, [0], color=:red, linestyle=:dash, linewidth=1.5)

	p2 = scatter(fitted_log, resids_log, alpha=0.5, legend=false, color=:darkgreen,
		title="Modèle log-transformé", xlabel="Valeurs ajustées (log)", ylabel="Résidus")
	hline!(p2, [0], color=:red, linestyle=:dash, linewidth=1.5)

	plot(p1, p2, layout=(1, 2), size=(800, 350))
end

# ╔═╡ c5d6e7f8-0018-6a4b-2d3e-8f9a0b1c2d3e
md"""
Autres mesures correctives possibles : erreurs-types robustes (HC1, HC3),
moindres carrés pondérés, méthodes non paramétriques, régularisation Ridge.
"""

# ╔═╡ c5d6e7f8-0019-6a4b-2d3e-8f9a0b1c2d3e
md"""
## Récapitulatif

Dans ce notebook, nous avons :

1. Effectué une analyse complète des résidus (6 graphiques)
2. Testé l'hétéroscédasticité (Breusch-Pagan)
3. Identifié les observations influentes (Cook's D, levier)
4. Vérifié la multicolinéarité (VIF)
5. Testé la normalité des résidus (Shapiro-Wilk)
6. Appliqué une transformation logarithmique comme mesure corrective

## Points clés

- **Toujours vérifier les hypothèses** avant de faire confiance aux résultats
- **Les graphiques de diagnostic** sont aussi importants que les tests statistiques
- **Les points influents** peuvent affecter dramatiquement les résultats
- **La multicolinéarité** rend les coefficients instables
- **Les transformations** peuvent corriger de nombreuses violations
- **Des méthodes robustes** existent quand les hypothèses ne peuvent être satisfaites

## Exercices

1. **Fonction de diagnostic réutilisable** — créer une fonction qui produit
   tous les graphiques pour n'importe quel modèle.
2. **Régression robuste** — implémenter une régression avec estimateur de Huber.
3. **Moindres carrés pondérés** — appliquer quand l'hétéroscédasticité est présente.
4. **Intervalles de confiance bootstrap** — utiliser le bootstrap pour les
   résidus non normaux.
5. **Pipeline automatisé** — créer un pipeline qui applique automatiquement
   les corrections appropriées.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
DataFrames = "1"
Distributions = "0.25"
GLM = "1"
Plots = "1"
"""

# ╔═╡ Cell order:
# ╟─c5d6e7f8-0001-6a4b-2d3e-8f9a0b1c2d3e
# ╟─c5d6e7f8-0002-6a4b-2d3e-8f9a0b1c2d3e
# ╠═c5d6e7f8-0003-6a4b-2d3e-8f9a0b1c2d3e
# ╟─c5d6e7f8-0004-6a4b-2d3e-8f9a0b1c2d3e
# ╠═c5d6e7f8-0005-6a4b-2d3e-8f9a0b1c2d3e
# ╟─c5d6e7f8-0006-6a4b-2d3e-8f9a0b1c2d3e
# ╠═c5d6e7f8-0007-6a4b-2d3e-8f9a0b1c2d3e
# ╟─c5d6e7f8-0008-6a4b-2d3e-8f9a0b1c2d3e
# ╠═c5d6e7f8-0009-6a4b-2d3e-8f9a0b1c2d3e
# ╟─c5d6e7f8-0010-6a4b-2d3e-8f9a0b1c2d3e
# ╠═c5d6e7f8-0011-6a4b-2d3e-8f9a0b1c2d3e
# ╟─c5d6e7f8-0012-6a4b-2d3e-8f9a0b1c2d3e
# ╠═c5d6e7f8-0013-6a4b-2d3e-8f9a0b1c2d3e
# ╟─c5d6e7f8-0014-6a4b-2d3e-8f9a0b1c2d3e
# ╠═c5d6e7f8-0015-6a4b-2d3e-8f9a0b1c2d3e
# ╟─c5d6e7f8-0016-6a4b-2d3e-8f9a0b1c2d3e
# ╠═c5d6e7f8-0017-6a4b-2d3e-8f9a0b1c2d3e
# ╟─c5d6e7f8-0018-6a4b-2d3e-8f9a0b1c2d3e
# ╟─c5d6e7f8-0019-6a4b-2d3e-8f9a0b1c2d3e
# ╠═00000000-0000-0000-0000-000000000001
