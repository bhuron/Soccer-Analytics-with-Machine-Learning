### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ a9b0c1d2-0003-0e8f-6b7c-2d3e4f5a6b7c
begin
	using JSON3
	using DataFrames
	using Statistics
	using Random
	using LinearAlgebra
	using GLM
	using Plots
	gr()
	Random.seed!(42)
	Plots.default(fontfamily="Helvetica", titlefontsize=11, guidefontsize=9)
end

# ╔═╡ a9b0c1d2-0001-0e8f-6b7c-2d3e4f5a6b7c
md"""
# Métriques d'évaluation des modèles

**Chapitre 4 · Prédire les résultats de matchs avec la classification**

## Ce que vous allez apprendre

- Comprendre et interpréter les matrices de confusion
- Calculer et interpréter précision, rappel et F1-score
- Utiliser les courbes ROC et l'AUC pour évaluer la performance
- Choisir les métriques appropriées selon le problème
"""

# ╔═╡ a9b0c1d2-0002-0e8f-6b7c-2d3e4f5a6b7c
md"""
## Imports et configuration
"""

# ╔═╡ a9b0c1d2-0004-0e8f-6b7c-2d3e4f5a6b7c
md"""
## Charger les données et entraîner le modèle

On reprend le modèle xG du notebook précédent.
"""

# ╔═╡ a9b0c1d2-0005-0e8f-6b7c-2d3e4f5a6b7c
begin
	DATA_DIR = joinpath(@__DIR__, "..", "..", "open-data", "data")

	function flatten_dict(d::AbstractDict; prefix="")
		flat = Dict{String,Any}()
		for (k, v) in d
			new_key = isempty(prefix) ? string(k) : "$prefix.$k"
			if v isa AbstractDict
				merge!(flat, flatten_dict(v; prefix=new_key))
			else
				flat[new_key] = v === nothing ? missing : v
			end
		end
		return flat
	end

	raw = JSON3.read(read(joinpath(DATA_DIR, "events", "22912.json"), String))
	dicts = flatten_dict.(raw)
	all_keys = union((keys(d) for d in dicts)...)
	rows = [let row = Dict{String,Any}()
		for k in all_keys; row[k] = get(d, k, missing); end; row
	end for d in dicts]
	events_eval = DataFrame(rows)

	# Extract shots with features
	shots_eval = subset(events_eval, "type.name" => ByRow(==("Shot")); skipmissing=true)
	shots_eval.goal = isequal.(shots_eval[!, "shot.outcome.name"], "Goal") .|> Int
	shots_eval.x = [(ismissing(loc) || loc === nothing) ? missing : loc[1] for loc in shots_eval.location]
	shots_eval.y = [(ismissing(loc) || loc === nothing) ? missing : loc[2] for loc in shots_eval.location]
	clean = dropmissing(shots_eval, [:x, :y])
	clean.distance = sqrt.((120 .- clean.x).^2 .+ (40 .- clean.y).^2)
	clean.angle = abs.(atand.(7.32 .* (120 .- clean.x) ./
		((120 .- clean.x).^2 .+ (40 .- clean.y).^2 .- (7.32/2)^2)))

	# Train/test split
	n_ev = nrow(clean)
	train_ix = shuffle(1:n_ev)[1:Int(round(0.8 * n_ev))]
	test_ix  = setdiff(1:n_ev, train_ix)
	df_tr_ev = clean[train_ix, :]
	df_te_ev = clean[test_ix, :]

	logit_ev = glm(@formula(goal ~ distance + angle), df_tr_ev, Binomial(), LogitLink())
	y_prob = predict(logit_ev, df_te_ev)
	y_pred = Int.(y_prob .>= 0.5)
	y_true = df_te_ev.goal

	global y_true, y_pred, y_prob
	md"""Modèle entraîné — **$(nrow(df_te_ev)) tirs** dans l'ensemble de test."""
end

# ╔═╡ a9b0c1d2-0006-0e8f-6b7c-2d3e4f5a6b7c
md"""
## 1. Matrice de confusion

Quatre composants :
- **Vrais positifs (TP)** : Prédit But, était But ✓
- **Vrais négatifs (TN)** : Prédit Pas but, était Pas but ✓
- **Faux positifs (FP)** : Prédit But, était Pas but ✗
- **Faux négatifs (FN)** : Prédit Pas but, était But ✗
"""

# ╔═╡ a9b0c1d2-0007-0e8f-6b7c-2d3e4f5a6b7c
let
	TP = sum((y_pred .== 1) .& (y_true .== 1))
	TN = sum((y_pred .== 0) .& (y_true .== 0))
	FP = sum((y_pred .== 1) .& (y_true .== 0))
	FN = sum((y_pred .== 0) .& (y_true .== 1))

	cm = [TN FP; FN TP]
	heatmap(["Pas but" "But"], ["Pas but", "But"], cm,
		color=:Blues, aspect_ratio=:equal,
		title="Matrice de confusion", xlabel="Prédit", ylabel="Réel")
	annotate!([(1, 1, Plots.text("TN=$TN", 10, :black, :center)),
	           (2, 1, Plots.text("FP=$FP", 10, :black, :center)),
	           (1, 2, Plots.text("FN=$FN", 10, :black, :center)),
	           (2, 2, Plots.text("TP=$TP", 10, :black, :center))])
end

# ╔═╡ a9b0c1d2-0008-0e8f-6b7c-2d3e4f5a6b7c
md"""
## 2. Accuracy (exactitude)

**Accuracy** = (TP + TN) / Total — pourcentage de prédictions correctes.

⚠️ Peut être trompeuse avec des classes déséquilibrées ! Si 90 % des tirs
ne sont pas des buts, un modèle prédisant toujours « Pas but » atteint
90 % d'exactitude mais ne sert à rien.
"""

# ╔═╡ a9b0c1d2-0009-0e8f-6b7c-2d3e4f5a6b7c
let
	TP = sum((y_pred .== 1) .& (y_true .== 1))
	TN = sum((y_pred .== 0) .& (y_true .== 0))
	acc = (TP + TN) / length(y_true)
	md"""**Accuracy = $(round(acc * 100; digits=1)) %**"""
end

# ╔═╡ a9b0c1d2-0010-0e8f-6b7c-2d3e4f5a6b7c
md"""
## 3. Précision (precision)

**Precision** = TP / (TP + FP) — parmi tous les tirs prédits comme but,
combien l'étaient vraiment ?

**Cas d'usage :** quand les faux positifs coûtent cher.
"""

# ╔═╡ a9b0c1d2-0011-0e8f-6b7c-2d3e4f5a6b7c
let
	TP = sum((y_pred .== 1) .& (y_true .== 1))
	FP = sum((y_pred .== 1) .& (y_true .== 0))
	prec = TP / (TP + FP)
	md"""**Precision = $(round(prec * 100; digits=1)) %** des buts prédits étaient de vrais buts."""
end

# ╔═╡ a9b0c1d2-0012-0e8f-6b7c-2d3e4f5a6b7c
md"""
## 4. Rappel (recall)

**Recall** = TP / (TP + FN) — parmi tous les vrais buts, combien avons-nous
correctement identifiés ?

**Cas d'usage :** quand les faux négatifs coûtent cher (ne pas rater une
occasion dangereuse).
"""

# ╔═╡ a9b0c1d2-0013-0e8f-6b7c-2d3e4f5a6b7c
let
	TP = sum((y_pred .== 1) .& (y_true .== 1))
	FN = sum((y_pred .== 0) .& (y_true .== 1))
	rec = TP / (TP + FN)
	md"""**Recall = $(round(rec * 100; digits=1)) %** des vrais buts ont été détectés."""
end

# ╔═╡ a9b0c1d2-0014-0e8f-6b7c-2d3e4f5a6b7c
md"""
## 5. F1-Score

**F1** = 2 × (Precision × Recall) / (Precision + Recall) — la moyenne
harmonique de la précision et du rappel.
"""

# ╔═╡ a9b0c1d2-0015-0e8f-6b7c-2d3e4f5a6b7c
let
	TP = sum((y_pred .== 1) .& (y_true .== 1))
	FP = sum((y_pred .== 1) .& (y_true .== 0))
	FN = sum((y_pred .== 0) .& (y_true .== 1))
	prec = TP / (TP + FP)
	rec  = TP / (TP + FN)
	f1   = 2 * prec * rec / (prec + rec)
	md"""**F1-Score = $(round(f1; digits=3))**"""
end

# ╔═╡ a9b0c1d2-0016-0e8f-6b7c-2d3e4f5a6b7c
md"""
## 6. Rapport de classification complet
"""

# ╔═╡ a9b0c1d2-0017-0e8f-6b7c-2d3e4f5a6b7c
let
	TP = sum((y_pred .== 1) .& (y_true .== 1))
	TN = sum((y_pred .== 0) .& (y_true .== 0))
	FP = sum((y_pred .== 1) .& (y_true .== 0))
	FN = sum((y_pred .== 0) .& (y_true .== 1))
	prec0 = TN / max(TN + FN, 1)
	prec1 = TP / max(TP + FP, 1)
	rec0  = TN / max(TN + FP, 1)
	rec1  = TP / max(TP + FN, 1)
	f1_0  = 2 * prec0 * rec0 / max(prec0 + rec0, 0.001)
	f1_1  = 2 * prec1 * rec1 / max(prec1 + rec1, 0.001)

	DataFrame(
		classe    = ["Pas but", "But"],
		précision = round.([prec0, prec1]; digits=3),
		rappel    = round.([rec0, rec1]; digits=3),
		f1_score  = round.([f1_0, f1_1]; digits=3),
		effectif  = [TN + FP, TP + FN],
	)
end

# ╔═╡ a9b0c1d2-0018-0e8f-6b7c-2d3e4f5a6b7c
md"""
## 7. Courbe ROC et AUC

La courbe ROC montre le compromis entre taux de vrais positifs (rappel)
et taux de faux positifs pour différents seuils de probabilité.

**AUC** = aire sous la courbe, de 0 à 1 :
- 0,5 = aléatoire, 1,0 = classifieur parfait
"""

# ╔═╡ a9b0c1d2-0019-0e8f-6b7c-2d3e4f5a6b7c
let
	# Compute ROC curve manually
	thresholds = sort(unique(y_prob); rev=true)
	tpr_vals = Float64[]
	fpr_vals = Float64[]
	for th in thresholds
		preds = Int.(y_prob .>= th)
		TP = sum((preds .== 1) .& (y_true .== 1))
		FP = sum((preds .== 1) .& (y_true .== 0))
		FN = sum((preds .== 0) .& (y_true .== 1))
		TN = sum((preds .== 0) .& (y_true .== 0))
		push!(tpr_vals, TP / max(TP + FN, 1))
		push!(fpr_vals, FP / max(FP + TN, 1))
	end

	# AUC via trapezoidal rule
	auc_val = 0.0
	for i in 2:length(fpr_vals)
		auc_val += (fpr_vals[i] - fpr_vals[i-1]) * (tpr_vals[i] + tpr_vals[i-1]) / 2
	end
	auc_val = abs(auc_val)

	plot(fpr_vals, tpr_vals, linewidth=2, color=:steelblue,
		label="ROC (AUC = $(round(auc_val; digits=3)))",
		title="Courbe ROC — Modèle xG", xlabel="Taux de faux positifs",
		ylabel="Taux de vrais positifs (Rappel)")
	plot!([0, 1], [0, 1], color=:black, linestyle=:dash, linewidth=1,
		label="Classifieur aléatoire")
end

# ╔═╡ a9b0c1d2-0020-0e8f-6b7c-2d3e4f5a6b7c
md"""
## Récapitulatif

Dans ce notebook, nous avons appris :

1. **Matrice de confusion** — tous les types de prédictions correctes/incorrectes
2. **Accuracy** — exactitude globale (peut tromper avec données déséquilibrées)
3. **Precision** — parmi les prédictions positives, combien étaient correctes
4. **Recall** — parmi les vrais positifs, combien ont été identifiés
5. **F1-Score** — moyenne harmonique de précision et rappel
6. **ROC/AUC** — performance globale à travers tous les seuils

## Points clés

- **Ne pas se fier uniquement à l'accuracy** pour des données déséquilibrées
- Le **compromis précision/rappel** dépend du cas d'usage
- L'**AUC** est indépendante du seuil, idéale pour comparer les modèles
- **Choisir les métriques** selon le coût des différents types d'erreurs

## Prochaine étape

Dans le prochain notebook, nous explorerons la **classification K-Nearest
Neighbors (KNN)** !
"""

# ╔═╡ a9b0c1d2-0021-0e8f-6b7c-2d3e4f5a6b7c
md"""
## Exercices

1. **Spécificité** — calculer TN / (TN + FP), le rappel de la classe négative.
2. **Seuil différent** — essayer seuil = 0,3 et recalculer les métriques.
3. **Données ultra-déséquilibrées** — créer un jeu avec 95 % de non-buts.
4. **Comparer deux modèles** — entraîner un second modèle et comparer l'AUC.
5. **Analyse de coût** — si FN coûte 100 € et FP 10 €, quel est le coût total ?
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
DataFrames = "1"
GLM = "1"
JSON3 = "1"
Plots = "1"
"""

# ╔═╡ Cell order:
# ╟─a9b0c1d2-0001-0e8f-6b7c-2d3e4f5a6b7c
# ╟─a9b0c1d2-0002-0e8f-6b7c-2d3e4f5a6b7c
# ╠═a9b0c1d2-0003-0e8f-6b7c-2d3e4f5a6b7c
# ╟─a9b0c1d2-0004-0e8f-6b7c-2d3e4f5a6b7c
# ╠═a9b0c1d2-0005-0e8f-6b7c-2d3e4f5a6b7c
# ╟─a9b0c1d2-0006-0e8f-6b7c-2d3e4f5a6b7c
# ╠═a9b0c1d2-0007-0e8f-6b7c-2d3e4f5a6b7c
# ╟─a9b0c1d2-0008-0e8f-6b7c-2d3e4f5a6b7c
# ╠═a9b0c1d2-0009-0e8f-6b7c-2d3e4f5a6b7c
# ╟─a9b0c1d2-0010-0e8f-6b7c-2d3e4f5a6b7c
# ╠═a9b0c1d2-0011-0e8f-6b7c-2d3e4f5a6b7c
# ╟─a9b0c1d2-0012-0e8f-6b7c-2d3e4f5a6b7c
# ╠═a9b0c1d2-0013-0e8f-6b7c-2d3e4f5a6b7c
# ╟─a9b0c1d2-0014-0e8f-6b7c-2d3e4f5a6b7c
# ╠═a9b0c1d2-0015-0e8f-6b7c-2d3e4f5a6b7c
# ╟─a9b0c1d2-0016-0e8f-6b7c-2d3e4f5a6b7c
# ╠═a9b0c1d2-0017-0e8f-6b7c-2d3e4f5a6b7c
# ╟─a9b0c1d2-0018-0e8f-6b7c-2d3e4f5a6b7c
# ╠═a9b0c1d2-0019-0e8f-6b7c-2d3e4f5a6b7c
# ╟─a9b0c1d2-0020-0e8f-6b7c-2d3e4f5a6b7c
# ╟─a9b0c1d2-0021-0e8f-6b7c-2d3e4f5a6b7c
# ╠═00000000-0000-0000-0000-000000000001
