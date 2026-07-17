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
- Utiliser la classification « un contre tous » (one-vs-rest)
- Interpréter une matrice de confusion à 3 classes
- Comprendre les forces et limites du modèle

## Le problème

**Objectif :** prédire si une équipe à domicile va gagner, faire match
nul ou perdre, en utilisant les statistiques du match lui-même.

Pourquoi c'est utile : comprendre *quelles* statistiques de match sont
les plus prédictives du résultat. Dans un scénario réel, on utiliserait
les moyennes historiques de ces métriques pour faire des prédictions
*avant* le match.
"""

# ╔═╡ c1d2e3f4-0002-2a0b-8d9e-4f5a6b7c8d9e
md"""
## Imports et configuration
"""

# ╔═╡ c1d2e3f4-0004-2a0b-8d9e-4f5a6b7c8d9e
md"""
## Étape 1 — Construire le jeu de données

On parcourt tous les matchs de la Coupe du Monde féminine 2019 (52 matchs)
et on extrait pour chaque match les tirs, passes et duels de chaque équipe.
"""

# ╔═╡ c1d2e3f4-0005-2a0b-8d9e-4f5a6b7c8d9e
begin
	DATA_DIR = joinpath(@__DIR__, "..", "..", "open-data", "data")

	# Load match list
	match_raw = JSON3.read(read(joinpath(DATA_DIR, "matches", "72", "30.json"), String))
	match_list = [(d["match_id"], d["home_team"]["home_team_name"],
	               d["away_team"]["away_team_name"],
	               d["home_score"], d["away_score"]) for d in match_raw]

	function count_events(events_df, team_name, event_type)
		subset(events_df,
			"team.name" => ByRow(==(team_name)),
			"type.name" => ByRow(==(event_type));
			skipmissing=true) |> nrow
	end

	match_features = DataFrame(
		match_id     = Int[],
		home_team    = String[],
		away_team    = String[],
		home_shots   = Int[],
		away_shots   = Int[],
		home_passes  = Int[],
		away_passes  = Int[],
		home_duels   = Int[],
		away_duels   = Int[],
		outcome      = String[],
	)

	for (mid, home, away, hs, aws) in match_list
		evt_file = joinpath(DATA_DIR, "events", "$mid.json")
		if !isfile(evt_file); continue; end

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

		raw = JSON3.read(read(evt_file, String))
		dicts = flatten_dict.(raw)
		keys_set = union((keys(d) for d in dicts)...)
		rows = [let row = Dict{String,Any}()
			for k in keys_set; row[k] = get(d, k, missing); end; row
		end for d in dicts]
		df = DataFrame(rows)

		outcome_str = hs > aws ? "Domicile" : hs < aws ? "Extérieur" : "Nul"

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
	md"""
	**$(nrow(df_matches)) matchs** chargés avec les statistiques domicile/extérieur.

	Distribution des résultats :
	"""
end

# ╔═╡ c1d2e3f4-0006-2a0b-8d9e-4f5a6b7c8d9e
combine(groupby(df_matches, :outcome), nrow => :count)

# ╔═╡ c1d2e3f4-0007-2a0b-8d9e-4f5a6b7c8d9e
md"""
## Étape 2 — Approche « un contre tous »

La régression logistique standard est binaire (deux classes).  Pour trois
classes (Domicile, Nul, Extérieur), on utilise la stratégie **one-vs-rest** :

1. On entraîne un classifieur binaire pour chaque classe : « Domicile vs
   le reste », « Nul vs le reste », « Extérieur vs le reste »
2. Pour prédire, chaque modèle donne une probabilité
3. La classe avec la plus haute probabilité l'emporte
"""

# ╔═╡ c1d2e3f4-0008-2a0b-8d9e-4f5a6b7c8d9e
begin
	feature_cols = [:home_shots, :away_shots, :home_passes, :away_passes,
	                :home_duels, :away_duels]
	X_match = Matrix(df_matches[!, feature_cols])

	# Standardize
	X_mean = mean(X_match; dims=1)
	X_std  = std(X_match; dims=1)
	X_scaled = (X_match .- X_mean) ./ X_std

	# Train/test split
	n_m = nrow(df_matches)
	train_ix_m = shuffle(1:n_m)[1:Int(round(0.7 * n_m))]
	test_ix_m  = setdiff(1:n_m, train_ix_m)
	y_all_m = Vector(df_matches[!, :outcome])

	global X_tr_m = X_scaled[train_ix_m, :]
	global y_tr_m = y_all_m[train_ix_m]
	global X_te_m = X_scaled[test_ix_m, :]
	global y_te_m = y_all_m[test_ix_m]

	md"""
	- Entraînement : **$(length(train_ix_m)) matchs**
	- Test : **$(length(test_ix_m)) matchs**
	"""
end

# ╔═╡ c1d2e3f4-0009-2a0b-8d9e-4f5a6b7c8d9e
function train_ovr(X, y, classes)
	models = Dict{String,Any}()
	for cls in classes
		# Binary target: 1 if this class, 0 otherwise
		y_bin = Int.(y .== cls)
		df_bin = DataFrame(X, :auto)
		df_bin.y = y_bin
		df_bin_names = Symbol.("x$i" for i in 1:size(X,2))
		rename!(df_bin, [df_bin_names..., :y])
		formula_str = "y ~ " * join(string.(df_bin_names), " + ")
		formula = @eval(@formula($(Meta.parse(formula_str))))
		models[cls] = glm(formula, df_bin, Binomial(), LogitLink())
	end
	return models
end

function predict_ovr(models, X)
	classes = collect(keys(models))
	probs = zeros(size(X, 1), length(classes))
	df_te = DataFrame(X, :auto)
	df_te_names = Symbol.("x$i" for i in 1:size(X,2))
	rename!(df_te, df_te_names)
	for (j, cls) in enumerate(classes)
		probs[:, j] = predict(models[cls], df_te)
	end
	# Pick class with highest probability
	return [classes[argmax(probs[i, :])] for i in 1:size(X, 1)]
end

# ╔═╡ c1d2e3f4-0010-2a0b-8d9e-4f5a6b7c8d9e
md"""
### Entraîner les trois classifieurs binaires
"""

# ╔═╡ c1d2e3f4-0011-2a0b-8d9e-4f5a6b7c8d9e
let
	classes = ["Domicile", "Nul", "Extérieur"]
	global ovr_models = train_ovr(X_tr_m, y_tr_m, classes)

	preds = predict_ovr(ovr_models, X_te_m)
	global ovr_preds = preds

	# Confusion matrix
	cm = zeros(Int, 3, 3)
	for (pred, true_val) in zip(preds, y_te_m)
		i = findfirst(==(true_val), classes)
		j = findfirst(==(pred), classes)
		cm[i, j] += 1
	end

	heatmap(cm, aspect_ratio=:equal, color=:Blues,
		title="Matrice de confusion — 3 classes",
		xlabel="Prédit", ylabel="Réel",
		xticks=([1, 2, 3], classes), yticks=([1, 2, 3], classes))
	for i in 1:3, j in 1:3
		annotate!(j, i, Plots.text("$(cm[i,j])", 10, :black, :center))
	end
end

# ╔═╡ c1d2e3f4-0012-2a0b-8d9e-4f5a6b7c8d9e
let
	acc = sum(ovr_preds .== y_te_m) / length(y_te_m)

	lines = ["**Performance sur le test ($(length(y_te_m)) matchs) :**", "", "**Accuracy : $(round(acc * 100; digits=1)) %**", ""]
	classes = ["Domicile", "Nul", "Extérieur"]
	for cls in classes
		y_bin = Int.(y_te_m .== cls)
		p_bin = Int.(ovr_preds .== cls)
		TP = sum((p_bin .== 1) .& (y_bin .== 1))
		FP = sum((p_bin .== 1) .& (y_bin .== 0))
		FN = sum((p_bin .== 0) .& (y_bin .== 1))
		prec = TP / max(TP + FP, 1)
		rec  = TP / max(TP + FN, 1)
		push!(lines, "- **$cls** — précision : $(round(prec; digits=2)), rappel : $(round(rec; digits=2))")
	end

	md"""$(join(lines, "\n"))"""
end

# ╔═╡ c1d2e3f4-0013-2a0b-8d9e-4f5a6b7c8d9e
md"""
## Étape 3 — Interpréter les résultats

**Ce que la matrice de confusion nous apprend :**

- La diagonale (haut-gauche → bas-droite) = prédictions correctes
- Les cellules hors diagonale = erreurs du modèle
- Le modèle confond souvent les nuls avec des victoires ou défaites —
  les matchs nuls sont intrinsèquement plus difficiles à prédire

### Comment lire la matrice

Les **lignes** représentent le résultat réel, les **colonnes** la
prédiction.  Par exemple, si la cellule (Domicile, Extérieur) vaut 3,
cela signifie que 3 matchs gagnés à domicile ont été prédits comme
des victoires à l'extérieur.

### Pourquoi les nuls sont-ils difficiles ?

Un match nul résulte souvent d'un équilibre statistique entre les deux
équipes — les variables comme les tirs ou les passes sont similaires
des deux côtés, ce qui rend la distinction difficile pour le modèle.

### Limites et prochaines étapes

- **Plus de données** — 52 matchs, c'est peu. Les ligues professionnelles
  offrent des centaines de matchs par saison.
- **Plus de variables** — xG, possession, précision des passes, tacles
  réussis apporteraient plus de signal.
- **Modèles plus puissants** — forêts aléatoires et XGBoost (chapitre 5)
  capturent mieux les interactions complexes.
- **Données historiques** — utiliser les moyennes des 5 derniers matchs
  plutôt que les stats du match lui-même (qui ne sont pas connues avant).
"""

# ╔═╡ c1d2e3f4-0014-2a0b-8d9e-4f5a6b7c8d9e
md"""
## Récapitulatif

Dans cette étude de cas, nous avons :

1. **Construit un jeu de données match par match** — tirs, passes et duels
   pour chaque équipe, extraits des événements StatsBomb
2. **Implémenté la stratégie one-vs-rest** — trois classifieurs binaires
   combinés pour la prédiction multiclasse
3. **Visualisé la matrice de confusion** — 3×3 pour comprendre où le
   modèle se trompe
4. **Interprété les résultats** — les victoires à domicile sont les plus
   faciles à prédire, les nuls les plus difficiles

## Points clés

- **One-vs-rest** est la méthode la plus simple pour étendre la
  classification binaire au multiclasse
- **La matrice de confusion** révèle les confusions systématiques du
  modèle (nuls confondus avec victoires/défaites)
- **52 matchs**, c'est peu — les modèles de production utilisent des
  milliers de matchs
- **Les variables intra-match** (tirs, passes) sont prédictives du
  résultat, mais dans la vraie vie on utilise des données *antérieures*
  au match

## Prochaine étape

Le chapitre 5 explore les **méthodes arborescentes** — arbres de décision,
forêts aléatoires et XGBoost — qui capturent mieux les interactions
complexes entre variables.
"""

# ╔═╡ c1d2e3f4-0015-2a0b-8d9e-4f5a6b7c8d9e
md"""
## Exercices

1. **Ajouter plus de variables** — inclure xG, possession, précision
   des passes dans le modèle.
2. **One-vs-one** — implémenter la stratégie one-vs-one (3 classifieurs
   par paire de classes) et comparer avec one-vs-rest.
3. **Données historiques** — au lieu des stats du match, utiliser les
   moyennes mobiles sur 5 matchs.
4. **Validation croisée** — évaluer le modèle avec une CV 5-fold plutôt
   qu'un simple split.
5. **Frontières de décision** — visualiser les régions de décision dans
   l'espace des deux premières composantes principales.
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
# ╠═c1d2e3f4-0006-2a0b-8d9e-4f5a6b7c8d9e
# ╟─c1d2e3f4-0007-2a0b-8d9e-4f5a6b7c8d9e
# ╠═c1d2e3f4-0008-2a0b-8d9e-4f5a6b7c8d9e
# ╠═c1d2e3f4-0009-2a0b-8d9e-4f5a6b7c8d9e
# ╟─c1d2e3f4-0010-2a0b-8d9e-4f5a6b7c8d9e
# ╠═c1d2e3f4-0011-2a0b-8d9e-4f5a6b7c8d9e
# ╠═c1d2e3f4-0012-2a0b-8d9e-4f5a6b7c8d9e
# ╟─c1d2e3f4-0013-2a0b-8d9e-4f5a6b7c8d9e
# ╟─c1d2e3f4-0014-2a0b-8d9e-4f5a6b7c8d9e
# ╟─c1d2e3f4-0015-2a0b-8d9e-4f5a6b7c8d9e
# ╠═00000000-0000-0000-0000-000000000001
