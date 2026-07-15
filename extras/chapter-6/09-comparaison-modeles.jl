### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ d6e7f8a9-0003-7b5c-3e4f-9a0b1c2d3e4f
begin
	using DataFrames
	using Statistics
	using Random
	using LinearAlgebra
	using GLM
	using Distributions
	using Plots
	gr()
	Random.seed!(42)
	Plots.default(fontfamily="Helvetica", titlefontsize=10, guidefontsize=9)
end

# ╔═╡ d6e7f8a9-0001-7b5c-3e4f-9a0b1c2d3e4f
md"""
# Comparaison et sélection avancée de modèles

**Chapitre 6 · Techniques de régression pour l'analyse du football — EXTRA**

## Ce que vous allez apprendre

- Maîtriser les cadres systématiques de comparaison de modèles
- Utiliser les critères d'information (AIC, BIC)
- Comparer plusieurs types de modèles de régression
- Construire des ensembles de modèles
- Comprendre le compromis biais-variance
- Sélectionner les modèles selon les objectifs métier
"""

# ╔═╡ d6e7f8a9-0002-7b5c-3e4f-9a0b1c2d3e4f
md"""
## Imports et configuration
"""

# ╔═╡ d6e7f8a9-0004-7b5c-3e4f-9a0b1c2d3e4f
md"""
## Le problème de la sélection de modèle

**Question :** comment choisir le meilleur modèle ?

**Considérations :**
1. **Précision prédictive** — dans quelle mesure généralise-t-il ?
2. **Interprétabilité** — peut-on expliquer les prédictions ?
3. **Coût de calcul** — rapidité d'entraînement/prédiction ?
4. **Robustesse** — sensibilité aux valeurs aberrantes ?
5. **Contraintes métier** — ce qui importe aux parties prenantes

**Pas de solution universelle :** aucun modèle n'est le meilleur pour tous
les problèmes !
"""

# ╔═╡ d6e7f8a9-0005-7b5c-3e4f-9a0b1c2d3e4f
begin
	n_cmp = 500
	df_cmp = DataFrame(
		shots             = rand(5:20, n_cmp),
		shots_on_target   = rand(2:12, n_cmp),
		possession        = round.(rand(35.0:0.1:70.0, n_cmp); digits=1),
		pass_accuracy     = round.(rand(70.0:0.1:92.0, n_cmp); digits=1),
		xg                = round.(rand(0.5:0.01:3.5, n_cmp); digits=2),
		opponent_xg       = round.(rand(0.5:0.01:3.5, n_cmp); digits=2),
		home              = rand([0, 1], n_cmp),
		opponent_strength = round.(rand(0.3:0.01:0.9, n_cmp); digits=2),
	)
	df_cmp.goals = round.(Int, clamp.(
		df_cmp.xg .* 0.75 .+ df_cmp.shots_on_target .* 0.08 .+
		df_cmp.home .* 0.3 .- df_cmp.opponent_strength .* 0.4 .+
		(df_cmp.possession .- 50) .* 0.01 .+ randn(n_cmp) .* 0.5, 0, 6))

	# Train/test split
	n_train_cmp = Int(round(0.75 * n_cmp))
	train_ix = shuffle(1:n_cmp)[1:n_train_cmp]
	test_ix  = setdiff(1:n_cmp, train_ix)
	X_all_cmp = Matrix(df_cmp[!, [:shots, :shots_on_target, :possession,
		:pass_accuracy, :xg, :opponent_xg, :home, :opponent_strength]])
	y_all_cmp = Vector(df_cmp[!, :goals])
	X_tr_cmp = X_all_cmp[train_ix, :]
	y_tr_cmp = y_all_cmp[train_ix]
	X_te_cmp = X_all_cmp[test_ix, :]
	y_te_cmp = y_all_cmp[test_ix]

	md"""Données chargées — **$(n_cmp) matchs**, entraînement $(n_train_cmp) / test $(n_cmp - n_train_cmp)"""
end

# ╔═╡ d6e7f8a9-0006-7b5c-3e4f-9a0b1c2d3e4f
md"""
## 1. Critères d'information — AIC et BIC

**AIC (Akaike) :** équilibre ajustement et complexité, pénalise le nombre
de paramètres. Plus bas = meilleur.

**BIC (Bayésien) :** similaire à l'AIC mais pénalise plus fortement la
complexité. Préfère les modèles plus simples.
"""

# ╔═╡ d6e7f8a9-0007-7b5c-3e4f-9a0b1c2d3e4f
let
	df_tr = DataFrame(X_tr_cmp, [:shots, :shots_on_target, :possession,
		:pass_accuracy, :xg, :opponent_xg, :home, :opponent_strength])
	df_tr.goals = y_tr_cmp

	basic    = lm(@formula(goals ~ xg + shots_on_target + home), df_tr)
	extended = lm(@formula(goals ~ xg + shots_on_target + home + opponent_strength + possession), df_tr)
	full     = lm(@formula(goals ~ shots + shots_on_target + possession + pass_accuracy +
		xg + opponent_xg + home + opponent_strength), df_tr)

	DataFrame(
		modèle   = ["Simple", "Étendu", "Complet"],
		variables = [3, 5, 8],
		R²       = round.([r²(basic), r²(extended), r²(full)]; digits=3),
		AIC      = round.([aic(basic), aic(extended), aic(full)]; digits=1),
		BIC      = round.([bic(basic), bic(extended), bic(full)]; digits=1),
	)
end

# ╔═╡ d6e7f8a9-0008-7b5c-3e4f-9a0b1c2d3e4f
md"""
Le BIC préfère généralement des modèles plus simples que l'AIC.
"""

# ╔═╡ d6e7f8a9-0009-7b5c-3e4f-9a0b1c2d3e4f
md"""
## 2. Validation croisée — K-Fold manuel
"""

# ╔═╡ d6e7f8a9-0010-7b5c-3e4f-9a0b1c2d3e4f
let
	k_folds = 5
	indices = shuffle(1:n_cmp)
	fold_size = n_cmp ÷ k_folds
	r2_cv = Float64[]

	for fold in 1:k_folds
		te = indices[(fold-1)*fold_size+1:min(fold*fold_size, n_cmp)]
		tr = setdiff(indices, te)
		df_cv = DataFrame(X_all_cmp[tr, :], [:shots, :shots_on_target, :possession,
			:pass_accuracy, :xg, :opponent_xg, :home, :opponent_strength])
		df_cv.goals = y_all_cmp[tr]
		m = lm(@formula(goals ~ xg + shots_on_target + home + opponent_strength + possession), df_cv)
		df_te = DataFrame(X_all_cmp[te, :], [:shots, :shots_on_target, :possession,
			:pass_accuracy, :xg, :opponent_xg, :home, :opponent_strength])
		yp = predict(m, df_te)
		push!(r2_cv, 1 - sum((y_all_cmp[te] .- yp).^2) / sum((y_all_cmp[te] .- mean(y_all_cmp[te])).^2))
	end

	md"""
	**Validation croisée $(k_folds)-Fold :**
	- R² moyen : **$(round(mean(r2_cv); digits=3))** (± $(round(std(r2_cv); digits=3)))
	"""
end

# ╔═╡ d6e7f8a9-0011-7b5c-3e4f-9a0b1c2d3e4f
md"""
## 3. Comparaison de 5 modèles
"""

# ╔═╡ d6e7f8a9-0012-7b5c-3e4f-9a0b1c2d3e4f
function knn_predict(X_tr, y_tr, X_te, k)
	dists = [sqrt(sum((X_te[i, :] .- X_tr[j, :]).^2)) for i in 1:size(X_te,1), j in 1:size(X_tr,1)]
	preds = zeros(size(X_te, 1))
	for i in 1:size(X_te, 1)
		preds[i] = mean(y_tr[sortperm(dists[i, :])[1:k]])
	end
	return preds
end

# ╔═╡ d6e7f8a9-0013-7b5c-3e4f-9a0b1c2d3e4f
let
	# Standardize for regularized models
	X_mean = mean(X_tr_cmp; dims=1)
	X_std  = std(X_tr_cmp; dims=1)
	X_tr_s = (X_tr_cmp .- X_mean) ./ X_std
	X_te_s = (X_te_cmp .- X_mean) ./ X_std

	df_tr = DataFrame(X_tr_cmp, [:shots, :shots_on_target, :possession,
		:pass_accuracy, :xg, :opponent_xg, :home, :opponent_strength])
	df_tr.goals = y_tr_cmp
	df_te = DataFrame(X_te_cmp, [:shots, :shots_on_target, :possession,
		:pass_accuracy, :xg, :opponent_xg, :home, :opponent_strength])

	results = DataFrame(model=String[], train_r2=Float64[], test_r2=Float64[],
		test_rmse=Float64[], test_mae=Float64[])

	# 1. Linear
	lin = lm(@formula(goals ~ shots + shots_on_target + possession + pass_accuracy +
		xg + opponent_xg + home + opponent_strength), df_tr)
	lp = predict(lin, df_te)
	push!(results, metrics_row("Linéaire", y_tr_cmp, predict(lin, df_tr), y_te_cmp, lp))

	# 2. Ridge (λ=1.0)
	λ = 1.0
	X_aug = hcat(ones(size(X_tr_s,1)), X_tr_s)
	ridge_b = (X_aug' * X_aug + λ * I) \ (X_aug' * y_tr_cmp)
	ridge_pred = hcat(ones(size(X_te_s,1)), X_te_s) * ridge_b
	push!(results, metrics_row("Ridge (λ=1)", y_tr_cmp,
		hcat(ones(size(X_tr_s,1)), X_tr_s) * ridge_b, y_te_cmp, ridge_pred))

	# 3. KNN (K=7)
	knn_pred = knn_predict(X_tr_s, y_tr_cmp, X_te_s, 7)
	knn_train = knn_predict(X_tr_s, y_tr_cmp, X_tr_s, 7)
	push!(results, metrics_row("KNN (K=7)", y_tr_cmp, knn_train, y_te_cmp, knn_pred))

	# 4. Mean baseline
	mean_pred = fill(mean(y_tr_cmp), length(y_te_cmp))
	mean_train = fill(mean(y_tr_cmp), length(y_tr_cmp))
	push!(results, metrics_row("Moyenne", y_tr_cmp, mean_train, y_te_cmp, mean_pred))

	# 5. Simple (xg + home)
	simple = lm(@formula(goals ~ xg + home), df_tr)
	sp = predict(simple, df_te)
	push!(results, metrics_row("Simple (xG+dom.)", y_tr_cmp, predict(simple, df_tr), y_te_cmp, sp))

	sort!(results, :test_r2, rev=true)

	global cmp_results = results

	# 2×2 visualization
	p1 = scatter(results.train_r2, results.test_r2, legend=false, markersize=6, color=:steelblue,
		title="Entraînement vs Test", xlabel="R² entraînement", ylabel="R² test")
	plot!(p1, [0, 1], [0, 1], color=:red, linestyle=:dash, linewidth=1.5)

	p2 = bar(results.model, results.test_r2, legend=false, color=:steelblue, alpha=0.7,
		title="R² test par modèle", xrotation=45, ylabel="R² test")

	p3 = bar(results.model, results.test_rmse, legend=false, color=:coral, alpha=0.7,
		title="RMSE test (plus bas = meilleur)", xrotation=45, ylabel="RMSE")

	gap = results.train_r2 .- results.test_r2
	colors = [g < 0.05 ? :green : g < 0.1 ? :orange : :red for g in gap]
	p4 = bar(results.model, gap, legend=false, color=colors, alpha=0.7,
		title="Écart de surapprentissage", xrotation=45, ylabel="R² entraînement − test")

	plot(p1, p2, p3, p4, layout=(2, 2), size=(900, 650))
end

# ╔═╡ d6e7f8a9-0014-7b5c-3e4f-9a0b1c2d3e4f
function metrics_row(name, y_tr, yp_tr, y_te, yp_te)
	r2_tr = 1 - sum((y_tr .- yp_tr).^2) / sum((y_tr .- mean(y_tr)).^2)
	r2_te = 1 - sum((y_te .- yp_te).^2) / sum((y_te .- mean(y_te)).^2)
	(name, r2_tr, r2_te, sqrt(mean((y_te .- yp_te).^2)), mean(abs.(y_te .- yp_te)))
end

# ╔═╡ d6e7f8a9-0015-7b5c-3e4f-9a0b1c2d3e4f
md"""
## 4. Ensembles de modèles

Combiner plusieurs modèles pour de meilleures prédictions.
"""

# ╔═╡ d6e7f8a9-0016-7b5c-3e4f-9a0b1c2d3e4f
let
	X_mean = mean(X_tr_cmp; dims=1)
	X_std  = std(X_tr_cmp; dims=1)
	X_tr_s = (X_tr_cmp .- X_mean) ./ X_std
	X_te_s = (X_te_cmp .- X_mean) ./ X_std

	# Train individual models
	df_tr = DataFrame(X_tr_cmp, [:shots, :shots_on_target, :possession,
		:pass_accuracy, :xg, :opponent_xg, :home, :opponent_strength])
	df_tr.goals = y_tr_cmp
	df_te = DataFrame(X_te_cmp, [:shots, :shots_on_target, :possession,
		:pass_accuracy, :xg, :opponent_xg, :home, :opponent_strength])

	lin = lm(@formula(goals ~ xg + shots_on_target + home + opponent_strength + possession), df_tr)
	pred_lin = predict(lin, df_te)

	X_aug = hcat(ones(size(X_tr_s,1)), X_tr_s)
	ridge_b = (X_aug' * X_aug + 1.0*I) \ (X_aug' * y_tr_cmp)
	pred_ridge = hcat(ones(size(X_te_s,1)), X_te_s) * ridge_b

	pred_knn = knn_predict(X_tr_s, y_tr_cmp, X_te_s, 7)

	preds = [pred_lin pred_ridge pred_knn]

	# Simple average ensemble
	ens_simple = mean(preds; dims=2)[:]
	# Weighted average (by test R²)
	w = [1-sum((y_te_cmp.-p).^2)/sum((y_te_cmp.-mean(y_te_cmp)).^2) for p in [pred_lin, pred_ridge, pred_knn]]
	w = w / sum(w)
	ens_weighted = preds * w

	r2_indiv = [round(1 - sum((y_te_cmp.-p).^2)/sum((y_te_cmp.-mean(y_te_cmp)).^2); digits=3) for p in [pred_lin, pred_ridge, pred_knn]]
	r2_simple = round(1 - sum((y_te_cmp.-ens_simple).^2)/sum((y_te_cmp.-mean(y_te_cmp)).^2); digits=3)
	r2_weighted = round(1 - sum((y_te_cmp.-ens_weighted).^2)/sum((y_te_cmp.-mean(y_te_cmp)).^2); digits=3)

	md"""
	**R² individuels :** Linéaire = $(r2_indiv[1]), Ridge = $(r2_indiv[2]), KNN = $(r2_indiv[3])

	**Ensembles :**
	- Moyenne simple : R² = **$(r2_simple)**
	- Moyenne pondérée : R² = **$(r2_weighted)**

	$(r2_weighted > maximum(r2_indiv) ? "✅ L'ensemble surpasse le meilleur modèle individuel !" : "⚠️ Le meilleur modèle individuel reste plus performant.")
	"""
end

# ╔═╡ d6e7f8a9-0017-7b5c-3e4f-9a0b1c2d3e4f
md"""
## 5. Compromis biais-variance
"""

# ╔═╡ d6e7f8a9-0018-7b5c-3e4f-9a0b1c2d3e4f
let
	# Use single feature (xG) with polynomial degrees 1-8
	x_simple = df_cmp.xg
	y_simple = df_cmp.goals
	n_s = length(x_simple)
	train_n = Int(round(0.75 * n_s))
	t_ix = shuffle(1:n_s)[1:train_n]
	e_ix = setdiff(1:n_s, t_ix)

	degrees = 1:8
	train_r2_poly = Float64[]
	test_r2_poly  = Float64[]

	for d in degrees
		X_poly = hcat([x_simple.^p for p in 1:d]...)
		X_tr_p = [ones(length(t_ix)) X_poly[t_ix, :]]
		X_te_p = [ones(length(e_ix)) X_poly[e_ix, :]]
		b = X_tr_p \ y_simple[t_ix]
		yp_tr = X_tr_p * b
		yp_te = X_te_p * b
		push!(train_r2_poly, 1 - sum((y_simple[t_ix].-yp_tr).^2)/sum((y_simple[t_ix].-mean(y_simple[t_ix])).^2))
		push!(test_r2_poly,  1 - sum((y_simple[e_ix].-yp_te).^2)/sum((y_simple[e_ix].-mean(y_simple[e_ix])).^2))
	end

	best_d = degrees[argmax(test_r2_poly)]

	plot(collect(degrees), train_r2_poly, label="Entraînement", marker=:circle, color=:steelblue,
		title="Compromis biais-variance", xlabel="Degré polynomial (complexité)", ylabel="R²")
	plot!(collect(degrees), test_r2_poly, label="Test", marker=:square, color=:coral)
	vline!([best_d], color=:green, linestyle=:dash, linewidth=1.5, label="Optimal (d=$best_d)")
end

# ╔═╡ d6e7f8a9-0019-7b5c-3e4f-9a0b1c2d3e4f
md"""
- **Degré faible :** biais élevé (sous-apprentissage) — les deux scores sont bas
- **Degré optimal :** meilleure performance de test
- **Degré élevé :** variance élevée (surapprentissage) — entraînement haut, test bas
"""

# ╔═╡ d6e7f8a9-0020-7b5c-3e4f-9a0b1c2d3e4f
md"""
## Récapitulatif

Dans ce notebook, nous avons :

1. Utilisé les critères d'information (AIC, BIC) pour la sélection
2. Implémenté la validation croisée K-Fold
3. Comparé 5 modèles de régression systématiquement
4. Construit des ensembles de modèles
5. Visualisé le compromis biais-variance
6. Discuté le cadre de sélection basé sur les objectifs métier

## Points clés

- **Pas de modèle universel** — tout dépend du contexte et des priorités
- **AIC/BIC** aident à comparer des modèles de complexité différente
- **La validation croisée** fournit des estimations robustes
- **Les ensembles** surpassent souvent les modèles individuels
- **Le compromis biais-variance** guide la complexité du modèle
- **Les objectifs métier** doivent guider la sélection
- **L'interprétabilité** compte pour l'adhésion des parties prenantes

## Exercices

1. **CV imbriquée** — implémenter une CV imbriquée pour le réglage des hyperparamètres
2. **Plus de modèles** — ajouter Lasso, ElasticNet, Gradient Boosting
3. **Sélection bayésienne** — utiliser des méthodes bayésiennes pour comparer
4. **Optimisation multi-objectif** — équilibrer précision, équité, rapidité
5. **Apprentissage sensible au coût** — intégrer les coûts de prédiction
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
# ╟─d6e7f8a9-0001-7b5c-3e4f-9a0b1c2d3e4f
# ╟─d6e7f8a9-0002-7b5c-3e4f-9a0b1c2d3e4f
# ╠═d6e7f8a9-0003-7b5c-3e4f-9a0b1c2d3e4f
# ╟─d6e7f8a9-0004-7b5c-3e4f-9a0b1c2d3e4f
# ╠═d6e7f8a9-0005-7b5c-3e4f-9a0b1c2d3e4f
# ╟─d6e7f8a9-0006-7b5c-3e4f-9a0b1c2d3e4f
# ╠═d6e7f8a9-0007-7b5c-3e4f-9a0b1c2d3e4f
# ╟─d6e7f8a9-0008-7b5c-3e4f-9a0b1c2d3e4f
# ╟─d6e7f8a9-0009-7b5c-3e4f-9a0b1c2d3e4f
# ╠═d6e7f8a9-0010-7b5c-3e4f-9a0b1c2d3e4f
# ╟─d6e7f8a9-0011-7b5c-3e4f-9a0b1c2d3e4f
# ╠═d6e7f8a9-0012-7b5c-3e4f-9a0b1c2d3e4f
# ╠═d6e7f8a9-0013-7b5c-3e4f-9a0b1c2d3e4f
# ╠═d6e7f8a9-0014-7b5c-3e4f-9a0b1c2d3e4f
# ╟─d6e7f8a9-0015-7b5c-3e4f-9a0b1c2d3e4f
# ╠═d6e7f8a9-0016-7b5c-3e4f-9a0b1c2d3e4f
# ╟─d6e7f8a9-0017-7b5c-3e4f-9a0b1c2d3e4f
# ╠═d6e7f8a9-0018-7b5c-3e4f-9a0b1c2d3e4f
# ╟─d6e7f8a9-0019-7b5c-3e4f-9a0b1c2d3e4f
# ╟─d6e7f8a9-0020-7b5c-3e4f-9a0b1c2d3e4f
# ╠═00000000-0000-0000-0000-000000000001
