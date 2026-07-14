### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ e1f2a3b4-0003-2c0d-8f9a-4b5c6d7e8f9a
begin
	using DataFrames
	using Statistics
	using Random
	using LinearAlgebra
	using GLM
	using DecisionTree
	using Plots
	gr()
	Random.seed!(42)
	Plots.default(fontfamily="Helvetica", titlefontsize=11, guidefontsize=9)
end

# ╔═╡ e1f2a3b4-0001-2c0d-8f9a-4b5c6d7e8f9a
md"""
# Évaluation et diagnostic des modèles

**Chapitre 6 · Techniques de régression pour l'analyse du football**

## Ce que vous allez apprendre

- Maîtriser les métriques clés de régression (R², RMSE, MAE)
- Comprendre l'importance des séparations entraînement/test
- Effectuer une analyse des résidus pour diagnostiquer les problèmes
- Identifier les observations influentes
- Comparer plusieurs modèles de façon systématique
- Interpréter les coefficients dans le contexte du football
"""

# ╔═╡ e1f2a3b4-0002-2c0d-8f9a-4b5c6d7e8f9a
md"""
## Imports et configuration
"""

# ╔═╡ e1f2a3b4-0004-2c0d-8f9a-4b5c6d7e8f9a
md"""
## Métriques clés de performance

Pour les modèles de régression, nous disposons de plusieurs métriques :

| Métrique | Formule | Interprétation | Plage |
|---|---|---|---|
| **R²** | 1 − (SS_res / SS_tot) | % de variance expliquée | 0 à 1 |
| **RMSE** | √(moyenne(erreurs²)) | Erreur moyenne (même unité) | 0 à ∞ |
| **MAE** | moyenne(\|erreurs\|) | Erreur absolue moyenne | 0 à ∞ |
"""

# ╔═╡ e1f2a3b4-0005-2c0d-8f9a-4b5c6d7e8f9a
md"""
## Chargement des données — résultats de matchs

Données simulées de 100 matchs pour prédire les buts marqués.
"""

# ╔═╡ e1f2a3b4-0006-2c0d-8f9a-4b5c6d7e8f9a
begin
	n_matches = 100

	shots   = rand(3:15, n_matches)
	poss    = rand(35.0:0.1:65.0, n_matches)
	xg      = rand(0.5:0.01:3.5, n_matches)
	goals   = round.(Int, clamp.(xg .* 0.8 .+ shots .* 0.1 .+ randn(n_matches) .* 0.5, 0, 5))

	match_df = DataFrame(
		shots_on_target = shots,
		possession      = round.(poss; digits=1),
		xg              = round.(xg; digits=2),
		goals           = goals,
	)

	first(match_df, 10)
end

# ╔═╡ e1f2a3b4-0007-2c0d-8f9a-4b5c6d7e8f9a
md"""
## Séparation entraînement/test — le fondement de l'évaluation

**Concept critique :** ne jamais évaluer sur les données d'entraînement !

- **Entraînement :** utilisé pour ajuster le modèle
- **Test :** utilisé pour évaluer la généralisation
- **Validation :** utilisé pour le réglage des hyperparamètres (optionnel)

**Pourquoi ?** La performance sur l'entraînement est optimiste. La
performance sur le test montre la capacité réelle.
"""

# ╔═╡ e1f2a3b4-0008-2c0d-8f9a-4b5c6d7e8f9a
begin
	features = [:shots_on_target, :possession, :xg]
	target   = :goals

	X_all = Matrix(match_df[!, features])
	y_all = Vector(match_df[!, target])

	# 70/30 split
	n_train = Int(round(0.7 * n_matches))
	train_idx = shuffle(1:n_matches)[1:n_train]
	test_idx  = setdiff(1:n_matches, train_idx)

	X_train = X_all[train_idx, :]
	y_train = y_all[train_idx]
	X_test  = X_all[test_idx, :]
	y_test  = y_all[test_idx]

	md"""
	- Entraînement : **$(n_train) matchs** (70 %)
	- Test : **$(n_matches - n_train) matchs** (30 %)
	"""
end

# ╔═╡ e1f2a3b4-0009-2c0d-8f9a-4b5c6d7e8f9a
md"""
## Construire et évaluer un modèle
"""

# ╔═╡ e1f2a3b4-0010-2c0d-8f9a-4b5c6d7e8f9a
let
	df_train = DataFrame(X_train, features)
	df_train.goals = y_train
	df_test  = DataFrame(X_test, features)
	df_test.goals  = y_test

	model = lm(@formula(goals ~ shots_on_target + possession + xg), df_train)

	y_train_pred = predict(model, df_train)
	y_test_pred  = predict(model, df_test)

	r2_train  = 1 - sum((y_train .- y_train_pred).^2) / sum((y_train .- mean(y_train)).^2)
	r2_test   = 1 - sum((y_test  .- y_test_pred).^2)  / sum((y_test  .- mean(y_test)).^2)
	rmse_train = sqrt(mean((y_train .- y_train_pred).^2))
	rmse_test  = sqrt(mean((y_test  .- y_test_pred).^2))
	mae_train  = mean(abs.(y_train .- y_train_pred))
	mae_test   = mean(abs.(y_test  .- y_test_pred))

	gap = abs(r2_train - r2_test) > 0.1

	md"""
	**Performance du modèle :**

	- R² entraînement : **$(round(r2_train; digits=3))** | test : **$(round(r2_test; digits=3))**
	- RMSE entraînement : **$(round(rmse_train; digits=3))** | test : **$(round(rmse_test; digits=3))**
	- MAE entraînement : **$(round(mae_train; digits=3))** | test : **$(round(mae_test; digits=3))**

	$(gap ? "⚠️ Écart important entre entraînement et test — possible surapprentissage !" : "✅ Performances similaires — bonne généralisation !")
	"""
end

# ╔═╡ e1f2a3b4-0011-2c0d-8f9a-4b5c6d7e8f9a
md"""
## Interpréter les métriques

**R² = 0,85** signifie :
- 85 % de la variance des buts est expliquée par nos variables
- 15 % est inexpliquée (variation aléatoire, variables manquantes)

**RMSE = 0,6 buts** signifie :
- En moyenne, les prédictions sont écartées de 0,6 buts
- Dans le contexte footballistique : prédire 2,3 buts quand le réel est 2 ou 3

**MAE = 0,5 buts** signifie :
- L'erreur absolue moyenne est d'un demi-but
- Plus interprétable que le RMSE pour les非 techniques
"""

# ╔═╡ e1f2a3b4-0012-2c0d-8f9a-4b5c6d7e8f9a
md"""
## Analyse des résidus — diagnostiquer les problèmes

**Résidus** = Réel − Prédit

Un bon modèle doit avoir :
1. **Dispersion aléatoire** autour de zéro (pas de motif)
2. **Variance constante** (homoscédasticité)
3. **Distribution normale** (pour l'inférence statistique)
"""

# ╔═╡ e1f2a3b4-0013-2c0d-8f9a-4b5c6d7e8f9a
let
	df_train = DataFrame(X_train, features)
	df_train.goals = y_train
	df_test  = DataFrame(X_test, features)
	df_test.goals  = y_test

	model = lm(@formula(goals ~ shots_on_target + possession + xg), df_train)
	y_pred = predict(model, df_test)
	resids = y_test .- y_pred

	# 4-panel residual plot
	p1 = scatter(y_pred, resids, legend=false, alpha=0.6, color=:steelblue,
		title="Résidus vs Valeurs prédites", xlabel="Buts prédits", ylabel="Résidus")
	hline!(p1, [0], color=:red, linestyle=:dash, linewidth=2)

	p2 = scatter(y_test, resids, legend=false, alpha=0.6, color=:coral,
		title="Résidus vs Valeurs réelles", xlabel="Buts réels", ylabel="Résidus")
	hline!(p2, [0], color=:red, linestyle=:dash, linewidth=2)

	p3 = histogram(resids, bins=15, legend=false, color=:darkgreen, alpha=0.7,
		title="Distribution des résidus", xlabel="Résidus", ylabel="Fréquence")
	vline!(p3, [0], color=:red, linestyle=:dash, linewidth=2)

	# Q-Q plot (manual)
	sorted_resids = sort(resids)
	n = length(sorted_resids)
	theoretical = quantile.(Normal(), (1:n .- 0.5) ./ n)
	p4 = scatter(theoretical, sorted_resids, legend=false, alpha=0.6, color=:purple,
		title="Q-Q Plot", xlabel="Quantiles théoriques", ylabel="Quantiles observés")
	plot!(p4, theoretical, theoretical, color=:red, linestyle=:dash, linewidth=2)

	plot(p1, p2, p3, p4, layout=(2, 2), size=(900, 700))
end

# ╔═╡ e1f2a3b4-0014-2c0d-8f9a-4b5c6d7e8f9a
md"""
**Checklist d'analyse des résidus :**
- ✓ Dispersion aléatoire autour de zéro ? (pas de forme en U)
- ✓ Variance constante ? (pas de forme d'entonnoir)
- ✓ Approximativement normale ? (histogramme en cloche, Q-Q linéaire)
"""

# ╔═╡ e1f2a3b4-0015-2c0d-8f9a-4b5c6d7e8f9a
md"""
## Identifier les observations influentes

Les **points influents** peuvent affecter le modèle de manière
disproportionnée. La **distance de Cook** mesure l'influence de
chaque observation.
"""

# ╔═╡ e1f2a3b4-0016-2c0d-8f9a-4b5c6d7e8f9a
let
	df_train = DataFrame(X_train, features)
	df_train.goals = y_train

	model = lm(@formula(goals ~ shots_on_target + possession + xg), df_train)
	cd = cooksdistance(model)
	threshold = 4 / n_train

	influential = findall(cd .> threshold)

	p = plot(cd, legend=false, color=:steelblue, markersize=4,
		seriestype=:scatter, title="Distance de Cook par observation",
		xlabel="Index d'observation", ylabel="Distance de Cook")
	hline!(p, [threshold], color=:red, linestyle=:dash, linewidth=2,
		label="Seuil (4/n)")

	md"""
	$(p)

	**$(length(influential)) observations influentes détectées** (distance de Cook > seuil).

	Ces points ont un fort effet de levier sur le modèle. Il faut les examiner —
	erreurs de données, événements aberrants, ou cas réels mais extrêmes ?
	"""
end

# ╔═╡ e1f2a3b4-0017-2c0d-8f9a-4b5c6d7e8f9a
md"""
## Validation croisée — évaluation robuste

**Problème :** une seule séparation entraînement/test peut être chanceuse
ou malchanceuse.

**Solution :** la validation croisée K-Fold
- Diviser les données en K groupes
- Entraîner sur K−1 groupes, tester sur 1
- Répéter K fois
- Moyenner les résultats
"""

# ╔═╡ e1f2a3b4-0018-2c0d-8f9a-4b5c6d7e8f9a
let
	k_folds = 5
	indices = shuffle(1:n_matches)
	fold_size = n_matches ÷ k_folds

	r2_scores = Float64[]
	rmse_scores = Float64[]

	for fold in 1:k_folds
		test_start = (fold - 1) * fold_size + 1
		test_end   = min(fold * fold_size, n_matches)
		test_fold  = indices[test_start:test_end]
		train_fold = setdiff(indices, test_fold)

		X_tr = X_all[train_fold, :]
		y_tr = y_all[train_fold]
		X_te = X_all[test_fold, :]
		y_te = y_all[test_fold]

		df_tr = DataFrame(X_tr, features)
		df_tr.goals = y_tr
		model_cv = lm(@formula(goals ~ shots_on_target + possession + xg), df_tr)
		y_pred = predict(model_cv, DataFrame(X_te, features))

		push!(r2_scores,   1 - sum((y_te .- y_pred).^2) / sum((y_te .- mean(y_te)).^2))
		push!(rmse_scores, sqrt(mean((y_te .- y_pred).^2)))
	end

	md"""
	**Validation croisée $(k_folds)-Fold :**
	- R² moyen : **$(round(mean(r2_scores); digits=3))** (± $(round(std(r2_scores); digits=3)))
	- RMSE moyen : **$(round(mean(rmse_scores); digits=3))** (± $(round(std(rmse_scores); digits=3)))

	La validation croisée fournit des estimations de performance plus fiables !
	"""
end

# ╔═╡ e1f2a3b4-0019-2c0d-8f9a-4b5c6d7e8f9a
md"""
## Comparaison systématique des modèles

Comparons régression linéaire, KNN et arbre de décision.
"""

# ╔═╡ e1f2a3b4-0020-2c0d-8f9a-4b5c6d7e8f9a
function knn_predict(X_train, y_train, X_query, k)
	dists = [sqrt(sum((X_query[i, :] .- X_train[j, :]).^2)) for i in 1:size(X_query,1), j in 1:size(X_train,1)]
	preds = zeros(size(X_query, 1))
	for i in 1:size(X_query, 1)
		neighbors = sortperm(dists[i, :])[1:k]
		preds[i] = mean(y_train[neighbors])
	end
	return preds
end

# ╔═╡ e1f2a3b4-0021-2c0d-8f9a-4b5c6d7e8f9a
let
	# Standardize for KNN
	X_mean = mean(X_train; dims=1)
	X_std  = std(X_train; dims=1)
	X_tr_s = (X_train .- X_mean) ./ X_std
	X_te_s = (X_test  .- X_mean) ./ X_std

	df_tr = DataFrame(X_train, features)
	df_tr.goals = y_train

	results = DataFrame(
		model      = String[],
		cv_r2_mean = Float64[],
		test_r2    = Float64[],
		test_rmse  = Float64[],
	)

	# 1. Linear regression
	lin_model = lm(@formula(goals ~ shots_on_target + possession + xg), df_tr)
	lin_pred  = predict(lin_model, DataFrame(X_test, features))
	push!(results, ["Régression linéaire",
		1 - sum((y_test .- lin_pred).^2) / sum((y_test .- mean(y_test)).^2),
		1 - sum((y_test .- lin_pred).^2) / sum((y_test .- mean(y_test)).^2),
		sqrt(mean((y_test .- lin_pred).^2))])

	# 2. KNN (K=5)
	knn_pred = knn_predict(X_tr_s, y_train, X_te_s, 5)
	push!(results, ["KNN (K=5)",
		1 - sum((y_test .- knn_pred).^2) / sum((y_test .- mean(y_test)).^2),
		1 - sum((y_test .- knn_pred).^2) / sum((y_test .- mean(y_test)).^2),
		sqrt(mean((y_test .- knn_pred).^2))])

	# 3. Decision tree
	tree_model = DecisionTreeRegressor(max_depth=5)
	DecisionTree.fit!(tree_model, X_train', y_train)
	tree_pred  = DecisionTree.predict(tree_model, X_test')
	push!(results, ["Arbre de décision",
		1 - sum((y_test .- tree_pred).^2) / sum((y_test .- mean(y_test)).^2),
		1 - sum((y_test .- tree_pred).^2) / sum((y_test .- mean(y_test)).^2),
		sqrt(mean((y_test .- tree_pred).^2))])

	# Bar charts
	p1 = bar(results.model, results.test_r2,
		legend=false, color=:steelblue, alpha=0.7,
		title="Comparaison des modèles — R²", ylabel="R²", xrotation=45)
	p2 = bar(results.model, results.test_rmse,
		legend=false, color=:coral, alpha=0.7,
		title="Comparaison des modèles — RMSE", ylabel="RMSE", xrotation=45)

	plot(p1, p2, layout=(1, 2), size=(900, 350))
end

# ╔═╡ e1f2a3b4-0022-2c0d-8f9a-4b5c6d7e8f9a
md"""
## Interpréter les coefficients

Pour les modèles linéaires, les coefficients indiquent l'importance
de chaque variable.
"""

# ╔═╡ e1f2a3b4-0023-2c0d-8f9a-4b5c6d7e8f9a
let
	df_tr = DataFrame(X_train, features)
	df_tr.goals = y_train
	model = lm(@formula(goals ~ shots_on_target + possession + xg), df_tr)

	coeffs = coef(model)
	coef_names = ["(intercept)", features...]
	coef_df = DataFrame(variable=String[], coefficient=Float64[])
	for (n, c) in zip(coef_names, coeffs)
		push!(coef_df, (n, c))
	end

	sort!(coef_df, :coefficient, rev=true)
	coef_df
end

# ╔═╡ e1f2a3b4-0024-2c0d-8f9a-4b5c6d7e8f9a
md"""
**Interprétation :** chaque coefficient indique de combien les buts attendus
changent pour une augmentation d'une unité de la variable correspondante,
toutes choses égales par ailleurs.  Le xG a le coefficient le plus élevé —
ce qui est logique puisqu'il est conçu pour prédire les buts.
"""

# ╔═╡ e1f2a3b4-0025-2c0d-8f9a-4b5c6d7e8f9a
md"""
## Récapitulatif

Dans ce notebook, nous avons :

1. Maîtrisé les métriques clés (R², RMSE, MAE)
2. Implémenté une séparation entraînement/test correcte
3. Effectué une analyse complète des résidus (4 graphiques)
4. Identifié les observations influentes avec la distance de Cook
5. Utilisé la validation croisée pour une évaluation robuste
6. Comparé plusieurs modèles de façon systématique
7. Interprété les coefficients du modèle

## Points clés

- **Ne jamais évaluer sur les données d'entraînement** — toujours utiliser un ensemble de test
- **Plusieurs métriques** donnent une image complète (R², RMSE, MAE)
- **L'analyse des résidus** révèle les problèmes du modèle
- **La validation croisée** fournit des estimations plus fiables
- **La comparaison des modèles** doit être systématique
- **L'interprétation compte** — les coefficients racontent l'histoire

## Prochaine étape

Dans le prochain notebook, nous appliquerons tout cela à une **étude de cas
pratique** de prédiction de résultats de matchs !
"""

# ╔═╡ e1f2a3b4-0026-2c0d-8f9a-4b5c6d7e8f9a
md"""
## Exercices

1. **Ensemble de validation** — implémenter une séparation
   entraînement/validation/test (60/20/20).
2. **R² ajusté** — calculer et comparer avec le R² classique.
3. **Intervalles de prédiction** — ajouter des intervalles de confiance
   aux prédictions.
4. **Sélection de variables** — utiliser les p-valeurs pour sélectionner
   les variables significatives.
5. **Courbes d'apprentissage** — tracer les scores en fonction de la
   taille du jeu de données.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DecisionTree = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
DataFrames = "1"
DecisionTree = "0.12"
GLM = "1"
Plots = "1"
"""

# ╔═╡ Cell order:
# ╟─e1f2a3b4-0001-2c0d-8f9a-4b5c6d7e8f9a
# ╟─e1f2a3b4-0002-2c0d-8f9a-4b5c6d7e8f9a
# ╠═e1f2a3b4-0003-2c0d-8f9a-4b5c6d7e8f9a
# ╟─e1f2a3b4-0004-2c0d-8f9a-4b5c6d7e8f9a
# ╟─e1f2a3b4-0005-2c0d-8f9a-4b5c6d7e8f9a
# ╠═e1f2a3b4-0006-2c0d-8f9a-4b5c6d7e8f9a
# ╟─e1f2a3b4-0007-2c0d-8f9a-4b5c6d7e8f9a
# ╠═e1f2a3b4-0008-2c0d-8f9a-4b5c6d7e8f9a
# ╟─e1f2a3b4-0009-2c0d-8f9a-4b5c6d7e8f9a
# ╠═e1f2a3b4-0010-2c0d-8f9a-4b5c6d7e8f9a
# ╟─e1f2a3b4-0011-2c0d-8f9a-4b5c6d7e8f9a
# ╟─e1f2a3b4-0012-2c0d-8f9a-4b5c6d7e8f9a
# ╠═e1f2a3b4-0013-2c0d-8f9a-4b5c6d7e8f9a
# ╟─e1f2a3b4-0014-2c0d-8f9a-4b5c6d7e8f9a
# ╟─e1f2a3b4-0015-2c0d-8f9a-4b5c6d7e8f9a
# ╠═e1f2a3b4-0016-2c0d-8f9a-4b5c6d7e8f9a
# ╟─e1f2a3b4-0017-2c0d-8f9a-4b5c6d7e8f9a
# ╠═e1f2a3b4-0018-2c0d-8f9a-4b5c6d7e8f9a
# ╟─e1f2a3b4-0019-2c0d-8f9a-4b5c6d7e8f9a
# ╠═e1f2a3b4-0020-2c0d-8f9a-4b5c6d7e8f9a
# ╠═e1f2a3b4-0021-2c0d-8f9a-4b5c6d7e8f9a
# ╟─e1f2a3b4-0022-2c0d-8f9a-4b5c6d7e8f9a
# ╠═e1f2a3b4-0023-2c0d-8f9a-4b5c6d7e8f9a
# ╟─e1f2a3b4-0024-2c0d-8f9a-4b5c6d7e8f9a
# ╟─e1f2a3b4-0025-2c0d-8f9a-4b5c6d7e8f9a
# ╟─e1f2a3b4-0026-2c0d-8f9a-4b5c6d7e8f9a
# ╠═00000000-0000-0000-0000-000000000001
