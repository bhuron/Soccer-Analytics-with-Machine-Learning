### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ c1d2e3f4-0003-2a0b-8d9e-4f5a6b7c8d9e
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

# ╔═╡ c1d2e3f4-0001-2a0b-8d9e-4f5a6b7c8d9e
md"""
# Prédire le résultat d'un match — étude de cas complète

**Chapitre 4 · Prédire les résultats de matchs avec la classification**

## Ce que vous allez apprendre

- Construire un classifieur multiclasse de bout en bout
- Extraire des variables par match depuis les données d'événements
- Utiliser la stratégie « un contre tous » (one-vs-rest)
- Interpréter une matrice de confusion à 3 classes
- Comprendre où le modèle réussit… et où il échoue

## Le problème

**Objectif :** prédire le résultat d'un match (Victoire à domicile, Nul,
Victoire à l'extérieur) à partir des statistiques du match.

On utilise les données réelles de la **Premier League 2015/16**
(38 matchs, données StatsBomb ouvertes).
"""

# ╔═╡ c1d2e3f4-0002-2a0b-8d9e-4f5a6b7c8d9e
md"""
## Imports et configuration
"""

# ╔═╡ c1d2e3f4-0004-2a0b-8d9e-4f5a6b7c8d9e
md"""
## Étape 1 — Construire le jeu de données

On parcourt chaque match de Premier League 2015/16 et on compte les
tirs (`Shot`), passes (`Pass`) et duels (`Duel`) de chaque équipe.
"""

# ╔═╡ c1d2e3f4-0005-2a0b-8d9e-4f5a6b7c8d9e
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

	function count_events(df, team, evt_type)
		subset(df,
			"team.name" => ByRow(==(team)),
			"type.name" => ByRow(==(evt_type));
			skipmissing=true) |> nrow
	end

	# Load PL 2015/16 match list
	match_raw = JSON3.read(read(joinpath(DATA_DIR, "matches", "2", "44.json"), String))

	match_features = DataFrame(
		match_id     = Int[],
		home_team    = String[],
		away_team    = String[],
		home_shots   = Int[],
		away_shots   = Int[],
		home_passes  = Int[],
		away_passes  = Int[],
		home_tackles = Int[],
		away_tackles = Int[],
		outcome      = String[],
	)

	for m in match_raw
		mid   = m["match_id"]
		home  = m["home_team"]["home_team_name"]
		away  = m["away_team"]["away_team_name"]
		hs    = m["home_score"]
		aws   = m["away_score"]

		evt_file = joinpath(DATA_DIR, "events", "$mid.json")
		if !isfile(evt_file); continue; end

		raw = JSON3.read(read(evt_file, String))
		dicts = flatten_dict.(raw)
		keys_set = union((keys(d) for d in dicts)...)
		rows = [let row = Dict{String,Any}()
			for k in keys_set; row[k] = get(d, k, missing); end; row
		end for d in dicts]
		df = DataFrame(rows)

		outcome_str = hs > aws ? "Home Win" : hs < aws ? "Away Win" : "Draw"

		push!(match_features, [
			mid, home, away,
			count_events(df, home, "Shot"),
			count_events(df, away, "Shot"),
			count_events(df, home, "Pass"),
			count_events(df, away, "Pass"),
			count_events(df, home, "Duel"),
			count_events(df, away, "Duel"),
			outcome_str,
		])
	end

	global df_matches = match_features
	first(df_matches, 8)
end

# ╔═╡ c1d2e3f4-0006-2a0b-8d9e-4f5a6b7c8d9e
md"""
### Distribution des résultats
"""

# ╔═╡ c1d2e3f4-0007-2a0b-8d9e-4f5a6b7c8d9e
let
	counts = combine(groupby(df_matches, :outcome), nrow => :count)
	sort!(counts, :count, rev=true)

	bar(counts.outcome, counts.count,
		legend=false, color=[:steelblue, :gold, :coral], alpha=0.7,
		title="Distribution des résultats — PL 2015/16",
		ylabel="Nombre de matchs")
end

# ╔═╡ c1d2e3f4-0008-2a0b-8d9e-4f5a6b7c8d9e
md"""
## Étape 2 — Préparer les données
"""

# ╔═╡ c1d2e3f4-0009-2a0b-8d9e-4f5a6b7c8d9e
begin
	feature_cols = [:home_shots, :away_shots, :home_passes, :away_passes,
	                :home_tackles, :away_tackles]
	X = Matrix(df_matches[!, feature_cols])
	y = Vector(df_matches[!, :outcome])

	# Standardize
	X_mean = mean(X; dims=1)
	X_std  = std(X; dims=1)
	X_s = (X .- X_mean) ./ X_std

	# Train/test split 70/30
	n_all = nrow(df_matches)
	train_ix = shuffle(1:n_all)[1:Int(round(0.7 * n_all))]
	test_ix  = setdiff(1:n_all, train_ix)

	global X_train = X_s[train_ix, :]
	global y_train = y[train_ix]
	global X_test  = X_s[test_ix, :]
	global y_test  = y[test_ix]

	md"""
	- Entraînement : **$(length(train_ix)) matchs**
	- Test : **$(length(test_ix)) matchs**
	- Variables : $(length(feature_cols)) (tirs, passes, duels × domicile/extérieur)
	"""
end

# ╔═╡ c1d2e3f4-0010-2a0b-8d9e-4f5a6b7c8d9e
md"""
## Étape 3 — Approche « un contre tous » (One-vs-Rest)

La régression logistique est binaire par nature. Pour trois classes,
on entraîne **un classifieur binaire par classe** :

1. **Home Win** vs le reste
2. **Draw** vs le reste
3. **Away Win** vs le reste

Pour prédire, on choisit la classe avec la **plus haute probabilité**.

> 📘 Dans `sklearn`, c'est `LogisticRegression(multi_class="multinomial")`
> qui utilise une approche légèrement différente (softmax).  Mais le
> one-vs-rest donne des résultats très similaires et est plus facile
> à comprendre.
"""

# ╔═╡ c1d2e3f4-0011-2a0b-8d9e-4f5a6b7c8d9e
function train_ovr(X, y, classes)
	models = Dict{String,Any}()
	n_feat = size(X, 2)
	for cls in classes
		y_bin = Int.(y .== cls)
		df_bin = DataFrame(X, :auto)
		df_bin.y = y_bin
		col_names = [Symbol("x$i") for i in 1:n_feat]
		rename!(df_bin, [col_names..., :y])
		rhs = join(string.(col_names), " + ")
		f = @eval(@formula(y ~ $(Meta.parse(rhs))))
		models[cls] = glm(f, df_bin, Binomial(), LogitLink())
	end
	return models
end

function predict_ovr(models, X)
	classes = collect(keys(models))
	n_feat = size(X, 2)
	probs = zeros(size(X, 1), length(classes))
	df_te = DataFrame(X, :auto)
	df_te_names = [Symbol("x$i") for i in 1:n_feat]
	rename!(df_te, df_te_names)
	for (j, cls) in enumerate(classes)
		probs[:, j] = predict(models[cls], df_te)
	end
	return [classes[argmax(probs[i, :])] for i in 1:size(X, 1)]
end

# ╔═╡ c1d2e3f4-0012-2a0b-8d9e-4f5a6b7c8d9e
md"""
### Entraînement et matrice de confusion
"""

# ╔═╡ c1d2e3f4-0013-2a0b-8d9e-4f5a6b7c8d9e
let
	classes = ["Home Win", "Draw", "Away Win"]
	models = train_ovr(X_train, y_train, classes)
	preds  = predict_ovr(models, X_test)
	global ovr_preds = preds

	cm = zeros(Int, 3, 3)
	for (p, t) in zip(preds, y_test)
		i = findfirst(==(t), classes)
		j = findfirst(==(p), classes)
		cm[i, j] += 1
	end

	cm_plot = heatmap(cm, aspect_ratio=:equal, color=:Blues,
		title="Matrice de confusion",
		xlabel="Prédit", ylabel="Réel",
		xticks=([1, 2, 3], classes), yticks=([1, 2, 3], classes),
		clims=(0, maximum(cm)))
	for i in 1:3, j in 1:3
		annotate!(cm_plot, j, i,
			Plots.text("$(cm[i,j])", 11, cm[i,j] > maximum(cm)/2 ? :white : :black, :center))
	end
	cm_plot
end

# ╔═╡ c1d2e3f4-0014-2a0b-8d9e-4f5a6b7c8d9e
md"""
### Rapport de classification
"""

# ╔═╡ c1d2e3f4-0015-2a0b-8d9e-4f5a6b7c8d9e
let
	classes = ["Home Win", "Draw", "Away Win"]
	acc = sum(ovr_preds .== y_test) / length(y_test)

	report_lines = String[]
	push!(report_lines, "**Accuracy globale : $(round(acc * 100; digits=1)) %**")
	push!(report_lines, "")

	for cls in classes
		y_bin = Int.(y_test .== cls)
		p_bin = Int.(ovr_preds .== cls)
		TP = sum((p_bin .== 1) .& (y_bin .== 1))
		FP = sum((p_bin .== 1) .& (y_bin .== 0))
		FN = sum((p_bin .== 0) .& (y_bin .== 1))
		TN = sum((p_bin .== 0) .& (y_bin .== 0))
		prec = TP / max(TP + FP, 1)
		rec  = TP / max(TP + FN, 1)
		f1   = 2 * prec * rec / max(prec + rec, 0.001)
		support = TP + FN
		push!(report_lines, "| $cls | $(round(prec; digits=2)) | $(round(rec; digits=2)) | $(round(f1; digits=2)) | $support |")
	end

	md"""
	| Classe | Précision | Rappel | F1 | Support |
	|---|---|---|---|---|
	$(join(report_lines, "\n"))
	"""
end

# ╔═╡ c1d2e3f4-0016-2a0b-8d9e-4f5a6b7c8d9e
md"""
## Étape 4 — Interpréter les résultats

### Comment lire la matrice de confusion

- **Lignes** = résultat réel (ce qui s'est passé)
- **Colonnes** = résultat prédit (ce que le modèle a dit)
- **Diagonale** = prédictions correctes ✓
- **Hors diagonale** = erreurs ✗

### Ce qu'on observe

1. **Les victoires à domicile sont les mieux prédites** — le signal
   (plus de tirs, plus de passes) est le plus fort quand l'équipe
   qui reçoit domine.
2. **Les nuls sont les plus difficiles** — quand les statistiques
   sont équilibrées, le modèle hésite entre les trois issues.
3. **Les victoires à l'extérieur** sont en position intermédiaire —
   elles ressemblent parfois à des nuls dominés par l'équipe visiteuse.

### Limites et améliorations possibles

- **38 matchs, c'est très peu** — un modèle de production utiliserait
  plusieurs saisons (des centaines de matchs).
- **Variables intra-match** — on utilise les stats du match lui-même.
  En pratique, on voudrait les *moyennes mobiles* des 5 matchs précédents.
- **Modèles plus puissants** — forêts aléatoires et XGBoost (chapitre 5)
  capturent mieux les interactions entre variables.
"""

# ╔═╡ c1d2e3f4-0017-2a0b-8d9e-4f5a6b7c8d9e
md"""
## Récapitulatif

Dans cette étude de cas, nous avons :

1. **Construit un jeu de données match par match** — tirs, passes et
   duels pour chaque équipe, extraits des événements StatsBomb de la
   Premier League 2015/16
2. **Implémenté la stratégie one-vs-rest** — trois classifieurs binaires
   combinés pour la prédiction multiclasse
3. **Visualisé la matrice de confusion** — 3×3 pour comprendre où le
   modèle se trompe
4. **Produit un rapport de classification complet** — précision, rappel
   et F1 par classe

## Points clés

- **One-vs-rest** est la méthode la plus simple pour la classification
  multiclasse avec des modèles binaires
- **38 matchs, c'est peu** — les modèles de production utilisent des
  milliers de matchs
- **Les nuls sont intrinsèquement difficiles** — l'équilibre statistique
  rend la distinction ardue

## Prochaine étape

Le chapitre 5 explore les **méthodes arborescentes** — arbres de décision,
forêts aléatoires et XGBoost — qui capturent mieux les interactions
complexes entre variables.
"""

# ╔═╡ c1d2e3f4-0018-2a0b-8d9e-4f5a6b7c8d9e
md"""
## Exercices

1. **Ajouter plus de variables** — inclure xG, possession et précision
   des passes dans le modèle.
2. **Saison complète** — utiliser les 52 matchs de la Coupe du Monde
   féminine 2019 (dossier `matches/72/30.json`).  Les résultats
   changent-ils ?
3. **One-vs-One** — implémenter la stratégie one-vs-one (3 classifieurs
   par paire de classes) et comparer avec one-vs-rest.
4. **Données historiques** — au lieu des stats du match, utiliser les
   moyennes mobiles sur 5 matchs.
5. **Validation croisée** — évaluer avec une CV 5-fold et comparer
   l'écart-type des métriques.
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
# ╟─c1d2e3f4-0001-2a0b-8d9e-4f5a6b7c8d9e
# ╟─c1d2e3f4-0002-2a0b-8d9e-4f5a6b7c8d9e
# ╠═c1d2e3f4-0003-2a0b-8d9e-4f5a6b7c8d9e
# ╟─c1d2e3f4-0004-2a0b-8d9e-4f5a6b7c8d9e
# ╠═c1d2e3f4-0005-2a0b-8d9e-4f5a6b7c8d9e
# ╟─c1d2e3f4-0006-2a0b-8d9e-4f5a6b7c8d9e
# ╠═c1d2e3f4-0007-2a0b-8d9e-4f5a6b7c8d9e
# ╟─c1d2e3f4-0008-2a0b-8d9e-4f5a6b7c8d9e
# ╠═c1d2e3f4-0009-2a0b-8d9e-4f5a6b7c8d9e
# ╟─c1d2e3f4-0010-2a0b-8d9e-4f5a6b7c8d9e
# ╠═c1d2e3f4-0011-2a0b-8d9e-4f5a6b7c8d9e
# ╟─c1d2e3f4-0012-2a0b-8d9e-4f5a6b7c8d9e
# ╠═c1d2e3f4-0013-2a0b-8d9e-4f5a6b7c8d9e
# ╟─c1d2e3f4-0014-2a0b-8d9e-4f5a6b7c8d9e
# ╠═c1d2e3f4-0015-2a0b-8d9e-4f5a6b7c8d9e
# ╟─c1d2e3f4-0016-2a0b-8d9e-4f5a6b7c8d9e
# ╟─c1d2e3f4-0017-2a0b-8d9e-4f5a6b7c8d9e
# ╟─c1d2e3f4-0018-2a0b-8d9e-4f5a6b7c8d9e
# ╠═00000000-0000-0000-0000-000000000001
