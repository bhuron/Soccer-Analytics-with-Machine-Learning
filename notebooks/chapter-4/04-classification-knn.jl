### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ b0c1d2e3-0003-1f9a-7c8d-3e4f5a6b7c8d
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

# ╔═╡ b0c1d2e3-0001-1f9a-7c8d-3e4f5a6b7c8d
md"""
# Classification K-Nearest Neighbors (KNN)

**Chapitre 4 · Prédire les résultats de matchs avec la classification**

## Ce que vous allez apprendre

- Comprendre l'algorithme KNN en classification
- Implémenter KNN manuellement pour prédire les buts
- Choisir le K optimal par validation croisée
- Comparer KNN avec la régression logistique
"""

# ╔═╡ b0c1d2e3-0002-1f9a-7c8d-3e4f5a6b7c8d
md"""
## Imports et configuration
"""

# ╔═╡ b0c1d2e3-0004-1f9a-7c8d-3e4f5a6b7c8d
md"""
## Qu'est-ce que KNN ?

KNN (K-Nearest Neighbors) est un algorithme de classification simple et
intuitif.  Au lieu d'apprendre une formule comme la régression logistique,
il **mémorise** toutes les données d'entraînement.

**Pour classer un nouveau tir :**
1. 📏 Calculer sa distance à tous les tirs connus
2. 👥 Prendre les K tirs les plus proches
3. 🗳️ Vote majoritaire : la classe la plus fréquente l'emporte

**Exemple avec K=3 :** si les 3 tirs les plus similaires sont
[Pas but, But, Pas but] → la classe majoritaire est **Pas but**.

**Avantages :** simple, aucun entraînement, s'adapte à des frontières complexes.
**Inconvénients :** lent en prédiction, sensible à l'échelle des variables.
"""

# ╔═╡ b0c1d2e3-0005-1f9a-7c8d-3e4f5a6b7c8d
md"""
## Charger et préparer les données

On reprend les tirs de la finale de la Ligue des Champions 2019.
"""

# ╔═╡ b0c1d2e3-0006-1f9a-7c8d-3e4f5a6b7c8d
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
	events_knn = DataFrame(rows)

	# Extract shots
	shots_knn = subset(events_knn, "type.name" => ByRow(==("Shot")); skipmissing=true)
	shots_knn.goal = isequal.(shots_knn[!, "shot.outcome.name"], "Goal") .|> Int
	shots_knn.x = [(ismissing(loc) || loc === nothing) ? missing : loc[1] for loc in shots_knn.location]
	shots_knn.y = [(ismissing(loc) || loc === nothing) ? missing : loc[2] for loc in shots_knn.location]
	clean_knn = dropmissing(shots_knn, [:x, :y])
	clean_knn.distance = sqrt.((120 .- clean_knn.x).^2 .+ (40 .- clean_knn.y).^2)
	clean_knn.angle    = abs.(atand.(7.32 .* (120 .- clean_knn.x) ./
		((120 .- clean_knn.x).^2 .+ (40 .- clean_knn.y).^2 .- (7.32/2)^2)))

	global df_knn = clean_knn
	md"""**$(nrow(df_knn)) tirs** avec distance et angle calculés."""
end

# ╔═╡ b0c1d2e3-0007-1f9a-7c8d-3e4f5a6b7c8d
md"""
## Implémenter KNN manuellement

Un classifieur KNN en ~10 lignes de Julia :
"""

# ╔═╡ b0c1d2e3-0008-1f9a-7c8d-3e4f5a6b7c8d
function knn_classify(X_train, y_train, X_query, k)
	# Compute all pairwise distances
	dists = [sqrt(sum((X_query[i, :] .- X_train[j, :]).^2))
	         for i in 1:size(X_query, 1), j in 1:size(X_train, 1)]
	preds = Int[]
	for i in 1:size(X_query, 1)
		neighbors = sortperm(dists[i, :])[1:k]    # K nearest
		votes = y_train[neighbors]                 # their labels
		push!(preds, argmax([count(==(c), votes) for c in [0, 1]]) - 1)
	end
	return preds
end

# ╔═╡ b0c1d2e3-0009-1f9a-7c8d-3e4f5a6b7c8d
md"""
## Entraînement et première évaluation

On standardise les variables (KNN est sensible à l'échelle !) puis on
teste avec K=5.
"""

# ╔═╡ b0c1d2e3-0010-1f9a-7c8d-3e4f5a6b7c8d
let
	# Prepare features and standardize
	X_mat = Matrix(df_knn[!, [:distance, :angle]])
	y_vec = Vector(df_knn[!, :goal])

	X_mean = mean(X_mat; dims=1)
	X_std  = std(X_mat; dims=1)
	X_scaled = (X_mat .- X_mean) ./ X_std

	# Train/test split
	n_knn = nrow(df_knn)
	train_ix = shuffle(1:n_knn)[1:Int(round(0.8 * n_knn))]
	test_ix  = setdiff(1:n_knn, train_ix)

	X_tr = X_scaled[train_ix, :]
	y_tr = y_vec[train_ix]
	X_te = X_scaled[test_ix, :]
	y_te = y_vec[test_ix]

	global X_tr, y_tr, X_te, y_te

	# KNN with K=5
	k = 5
	preds = knn_classify(X_tr, y_tr, X_te, k)

	TP = sum((preds .== 1) .& (y_te .== 1))
	FP = sum((preds .== 1) .& (y_te .== 0))
	FN = sum((preds .== 0) .& (y_te .== 1))
	acc = (TP + sum((preds .== 0) .& (y_te .== 0))) / length(y_te)
	prec = TP / max(TP + FP, 1)
	rec  = TP / max(TP + FN, 1)

	md"""
	**KNN (K=$k) — Performance sur le test :**
	- Accuracy : **$(round(acc * 100; digits=1)) %**
	- Precision : **$(round(prec; digits=3))**
	- Recall : **$(round(rec; digits=3))**
	"""
end

# ╔═╡ b0c1d2e3-0011-1f9a-7c8d-3e4f5a6b7c8d
md"""
## Choisir le K optimal

K contrôle le compromis biais-variance :
- **K=1** — trop flexible, sensible au bruit
- **K grand** — trop lisse, peut manquer des motifs
- **K optimal** — quelque part entre les deux

Testons K de 1 à 15.
"""

# ╔═╡ b0c1d2e3-0012-1f9a-7c8d-3e4f5a6b7c8d
let
	ks = 1:15
	accs = Float64[]
	for k in ks
		preds = knn_classify(X_tr, y_tr, X_te, k)
		acc = sum(preds .== y_te) / length(y_te)
		push!(accs, acc)
	end

	best_k = ks[argmax(accs)]

	plot(collect(ks), accs, marker=:circle, legend=false, color=:steelblue,
		title="Performance KNN vs K", xlabel="K (nombre de voisins)",
		ylabel="Accuracy sur le test")
	vline!([best_k], color=:red, linestyle=:dash, linewidth=1.5,
		label="Meilleur K = $best_k")
end

# ╔═╡ b0c1d2e3-0013-1f9a-7c8d-3e4f5a6b7c8d
md"""
## Comparaison KNN vs Régression logistique

Entraînons les deux modèles sur les mêmes données et comparons.
"""

# ╔═╡ b0c1d2e3-0014-1f9a-7c8d-3e4f5a6b7c8d
let
	# Logistic regression
	df_tr = DataFrame(X_tr, [:distance, :angle])
	df_tr.goal = y_tr
	df_te = DataFrame(X_te, [:distance, :angle])

	logit_knn = glm(@formula(goal ~ distance + angle), df_tr, Binomial(), LogitLink())
	logit_probs = predict(logit_knn, df_te)
	logit_preds = Int.(logit_probs .>= 0.5)

	# KNN with best K
	best_k = ks[argmax(accs)]
	knn_preds = knn_classify(X_tr, y_tr, X_te, best_k)

	# Compute metrics for both
	metrics_fn(y_pred, y_true) = begin
		TP = sum((y_pred .== 1) .& (y_true .== 1))
		FP = sum((y_pred .== 1) .& (y_true .== 0))
		FN = sum((y_pred .== 0) .& (y_true .== 1))
		acc = sum(y_pred .== y_true) / length(y_true)
		prec = TP / max(TP + FP, 1)
		rec  = TP / max(TP + FN, 1)
		f1   = 2 * prec * rec / max(prec + rec, 0.001)
		(acc, prec, rec, f1)
	end

	lin_m = metrics_fn(logit_preds, y_te)
	knn_m = metrics_fn(knn_preds, y_te)

	DataFrame(
		modèle    = ["Régression logistique", "KNN (K=$best_k)"],
		accuracy  = round.([lin_m[1], knn_m[1]]; digits=3),
		precision = round.([lin_m[2], knn_m[2]]; digits=3),
		recall    = round.([lin_m[3], knn_m[3]]; digits=3),
		f1_score  = round.([lin_m[4], knn_m[4]]; digits=3),
	)
end

# ╔═╡ b0c1d2e3-0015-1f9a-7c8d-3e4f5a6b7c8d
md"""
## Visualiser la frontière de décision

Dans l'espace 2D (distance × angle), on peut voir comment KNN divise
le plan en régions « But » et « Pas but ».
"""

# ╔═╡ b0c1d2e3-0016-1f9a-7c8d-3e4f5a6b7c8d
let
	k = 5
	# Create a grid of points
	d_range = range(minimum(X_tr[:, 1]), maximum(X_tr[:, 1]); length=60)
	a_range = range(minimum(X_tr[:, 2]), maximum(X_tr[:, 2]); length=60)
	grid = [[d, a] for d in d_range, a in a_range]
	grid_flat = vcat([grid[i, j] for i in 1:60, j in 1:60]...)
	grid_flat = reshape(grid_flat, 3600, 2)

	grid_preds = knn_classify(X_tr, y_tr, grid_flat, k)

	# Scatter with decision regions
	scatter(X_tr[y_tr .== 0, 1], X_tr[y_tr .== 0, 2],
		color=:coral, alpha=0.5, markersize=5, label="Pas but (train)",
		title="Frontière de décision KNN (K=$k)", xlabel="Distance std", ylabel="Angle std")
	scatter!(X_tr[y_tr .== 1, 1], X_tr[y_tr .== 1, 2],
		color=:lime, alpha=0.8, markersize=8, label="But (train)")

	# Overlay grid predictions as background
	but_region = grid_flat[grid_preds .== 1, :]
	if nrow(DataFrame(but_region)) > 0
		scatter!(but_region[:, 1], but_region[:, 2],
			color=:green, alpha=0.03, markersize=1, label="Région But")
	end
end

# ╔═╡ b0c1d2e3-0017-1f9a-7c8d-3e4f5a6b7c8d
md"""
## Récapitulatif

Dans ce notebook, nous avons :

1. Compris le principe de KNN : vote majoritaire des K plus proches voisins
2. Implémenté un classifieur KNN manuellement en ~10 lignes
3. Testé différentes valeurs de K pour trouver l'optimal
4. Comparé KNN avec la régression logistique
5. Visualisé la frontière de décision en 2D

## Points clés

- **KNN ne s'entraîne pas** — il mémorise les données
- **K contrôle la flexibilité** — K petit = flexible, K grand = lisse
- **La standardisation est cruciale** — KNN est basé sur les distances
- **KNN capture des frontières complexes** que la régression logistique
  ne peut pas modéliser
- **Compromis :** KNN est plus lent en prédiction que la régression
  logistique

## Prochaine étape

Dans le prochain notebook, nous construirons un **prédicteur de résultat
de match** complet !
"""

# ╔═╡ b0c1d2e3-0018-1f9a-7c8d-3e4f5a6b7c8d
md"""
## Exercices

1. **Métrique de distance** — essayer la distance de Manhattan au lieu
   d'Euclidienne.  L'accuracy change-t-elle ?
2. **KNN pondéré** — pondérer les votes par l'inverse de la distance.
3. **Plus de variables** — ajouter `shot.body_part.name` ou
   `shot.technique.name` au modèle KNN.
4. **Validation croisée** — implémenter une CV 5-fold pour choisir K
   de façon plus robuste.
5. **Temps de calcul** — mesurer le temps de prédiction pour K=1, K=10,
   K=100.  Que remarquez-vous ?
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
# ╟─b0c1d2e3-0001-1f9a-7c8d-3e4f5a6b7c8d
# ╟─b0c1d2e3-0002-1f9a-7c8d-3e4f5a6b7c8d
# ╠═b0c1d2e3-0003-1f9a-7c8d-3e4f5a6b7c8d
# ╟─b0c1d2e3-0004-1f9a-7c8d-3e4f5a6b7c8d
# ╟─b0c1d2e3-0005-1f9a-7c8d-3e4f5a6b7c8d
# ╠═b0c1d2e3-0006-1f9a-7c8d-3e4f5a6b7c8d
# ╟─b0c1d2e3-0007-1f9a-7c8d-3e4f5a6b7c8d
# ╠═b0c1d2e3-0008-1f9a-7c8d-3e4f5a6b7c8d
# ╟─b0c1d2e3-0009-1f9a-7c8d-3e4f5a6b7c8d
# ╠═b0c1d2e3-0010-1f9a-7c8d-3e4f5a6b7c8d
# ╟─b0c1d2e3-0011-1f9a-7c8d-3e4f5a6b7c8d
# ╠═b0c1d2e3-0012-1f9a-7c8d-3e4f5a6b7c8d
# ╟─b0c1d2e3-0013-1f9a-7c8d-3e4f5a6b7c8d
# ╠═b0c1d2e3-0014-1f9a-7c8d-3e4f5a6b7c8d
# ╟─b0c1d2e3-0015-1f9a-7c8d-3e4f5a6b7c8d
# ╠═b0c1d2e3-0016-1f9a-7c8d-3e4f5a6b7c8d
# ╟─b0c1d2e3-0017-1f9a-7c8d-3e4f5a6b7c8d
# ╟─b0c1d2e3-0018-1f9a-7c8d-3e4f5a6b7c8d
# ╠═00000000-0000-0000-0000-000000000001
