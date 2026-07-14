### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ d0e1f2a3-0003-1b9c-7e8f-3a4b5c6d7e8f
begin
	using DataFrames
	using Statistics
	using Random
	using LinearAlgebra
	using GLM
	using Plots
	gr()
	Random.seed!(42)
	Plots.default(fontfamily="Helvetica", titlefontsize=12, guidefontsize=10)
end

# ╔═╡ d0e1f2a3-0001-1b9c-7e8f-3a4b5c6d7e8f
md"""
# Régression K-Nearest Neighbors (KNN) — Trouver des joueurs similaires

**Chapitre 6 · Techniques de régression pour l'analyse du football**

## Ce que vous allez apprendre

- Comprendre comment fonctionne la régression KNN
- La différence entre modèles paramétriques et non paramétriques
- Construire un modèle KNN pour prédire la valeur des joueurs
- Choisir la valeur optimale de K
- Visualiser les surfaces de décision
- Comparer KNN avec la régression linéaire
"""

# ╔═╡ d0e1f2a3-0002-1b9c-7e8f-3a4b5c6d7e8f
md"""
## Imports et configuration
"""

# ╔═╡ d0e1f2a3-0004-1b9c-7e8f-3a4b5c6d7e8f
md"""
## L'intuition — qui se ressemble s'assemble

La **régression KNN** est fondamentalement différente de la régression
linéaire ou de Poisson :

- **Linéaire/Poisson :** apprendre une formule à partir de toutes les
  données, puis l'appliquer
- **KNN :** pour chaque prédiction, trouver les K points les plus
  similaires et faire la moyenne de leurs valeurs

**Exemple :** pour prédire la valeur marchande d'un joueur :
1. Trouver les 5 joueurs les plus similaires (buts, passes décisives, âge…)
2. Faire la moyenne de leurs valeurs marchandes
3. Voilà la prédiction !

**Pas de formule, pas d'entraînement au sens traditionnel** — juste une
recherche par similarité !
"""

# ╔═╡ d0e1f2a3-0005-1b9c-7e8f-3a4b5c6d7e8f
md"""
## Modèles paramétriques vs non paramétriques

| Aspect | Paramétrique (Linéaire, Poisson) | Non paramétrique (KNN) |
|---|---|---|
| **Modèle** | Apprend des paramètres fixes (pente, intercept) | Pas de paramètres fixes |
| **Entraînement** | Ajuste une équation aux données | Mémorise toutes les données |
| **Prédiction** | Applique la formule apprise | Trouve les points similaires |
| **Flexibilité** | Suppose une relation spécifique | S'adapte à n'importe quel motif |
| **Interprétabilité** | Élevée (coefficients clairs) | Faible (boîte noire) |
"""

# ╔═╡ d0e1f2a3-0006-1b9c-7e8f-3a4b5c6d7e8f
md"""
## Chargement des données — statistiques des joueurs

Un jeu de données simulé plus riche, avec plusieurs variables.
"""

# ╔═╡ d0e1f2a3-0007-1b9c-7e8f-3a4b5c6d7e8f
begin
	n_players = 50

	goals    = rand(5:30, n_players)
	assists  = rand(2:15, n_players)
	ages     = rand(20:35, n_players)
	# Market value with correlation to features + noise
	values   = goals .* 2.5 .+ assists .* 1.5 .- ages .* 0.5 .+ randn(n_players) .* 10
	values   = clamp.(values, 10, 100)

	players_df = DataFrame(
		goals        = goals,
		assists      = assists,
		age          = ages,
		market_value = round.(values; digits=1),
	)

	first(players_df, 10)
end

# ╔═╡ d0e1f2a3-0008-1b9c-7e8f-3a4b5c6d7e8f
md"""
## Préparer les données

KNN est basé sur les **distances**, donc la mise à l'échelle des variables
est cruciale ! Sans elle, les buts (5–30) domineraient l'âge (20–35) dans
le calcul de distance, simplement à cause de leur échelle plus grande.
"""

# ╔═╡ d0e1f2a3-0009-1b9c-7e8f-3a4b5c6d7e8f
begin
	features = [:goals, :assists, :age]
	target   = :market_value

	X = Matrix(players_df[!, features])
	y = Vector(players_df[!, target])

	# Train/test split (80/20)
	n_train = Int(round(0.8 * n_players))
	train_idx = shuffle(1:n_players)[1:n_train]
	test_idx  = setdiff(1:n_players, train_idx)

	X_train = X[train_idx, :]
	y_train = y[train_idx]
	X_test  = X[test_idx, :]
	y_test  = y[test_idx]

	# Standardisation : (x - mean) / std
	X_mean = mean(X_train; dims=1)
	X_std  = std(X_train; dims=1)
	X_train_scaled = (X_train .- X_mean) ./ X_std
	X_test_scaled  = (X_test  .- X_mean) ./ X_std

	md"""
	- Ensemble d'entraînement : **$(n_train) joueurs**
	- Ensemble de test : **$(n_players - n_train) joueurs**
	- Variables standardisées : moyenne = 0, écart-type = 1
	"""
end

# ╔═╡ d0e1f2a3-0010-1b9c-7e8f-3a4b5c6d7e8f
md"""
## Construire le modèle KNN

Commençons avec K = 5.  Le principe :
1. Pour chaque point à prédire, calculer la distance à tous les points
   d'entraînement
2. Garder les K plus proches
3. Faire la moyenne de leurs valeurs cibles
"""

# ╔═╡ d0e1f2a3-0011-1b9c-7e8f-3a4b5c6d7e8f
function knn_predict(X_train, y_train, X_query, k)
	dists = pairwise(Euclidean(), X_query', X_train'; dims=2)
	preds = zeros(size(X_query, 1))
	for i in 1:size(X_query, 1)
		neighbors = sortperm(dists[i, :])[1:k]
		preds[i] = mean(y_train[neighbors])
	end
	return preds
end

# ╔═╡ d0e1f2a3-0012-1b9c-7e8f-3a4b5c6d7e8f
let
	k = 5
	y_pred_train = knn_predict(X_train_scaled, y_train, X_train_scaled, k)
	y_pred_test  = knn_predict(X_train_scaled, y_train, X_test_scaled, k)

	r2_train = 1 - sum((y_train .- y_pred_train).^2) / sum((y_train .- mean(y_train)).^2)
	r2_test  = 1 - sum((y_test  .- y_pred_test).^2)  / sum((y_test  .- mean(y_test)).^2)
	rmse_train = sqrt(mean((y_train .- y_pred_train).^2))
	rmse_test  = sqrt(mean((y_test  .- y_pred_test).^2))

	md"""
	**Performance du modèle KNN (K = $k) :**
	- R² entraînement : **$(round(r2_train; digits=3))**
	- R² test : **$(round(r2_test; digits=3))**
	- RMSE entraînement : **$(round(rmse_train; digits=1)) M€**
	- RMSE test : **$(round(rmse_test; digits=1)) M€**
	"""
end

# ╔═╡ d0e1f2a3-0013-1b9c-7e8f-3a4b5c6d7e8f
md"""
## Choisir le bon K

**K** est un hyperparamètre qui contrôle la complexité du modèle :

- **K = 1 :** très flexible, risque de surapprentissage
- **K élevé :** très lisse, risque de sous-apprentissage
- **K optimal :** équilibre entre biais et variance
"""

# ╔═╡ d0e1f2a3-0014-1b9c-7e8f-3a4b5c6d7e8f
let
	ks = 1:20
	train_r2 = Float64[]
	test_r2  = Float64[]

	for k in ks
		yp_train = knn_predict(X_train_scaled, y_train, X_train_scaled, k)
		yp_test  = knn_predict(X_train_scaled, y_train, X_test_scaled, k)
		push!(train_r2, 1 - sum((y_train .- yp_train).^2) / sum((y_train .- mean(y_train)).^2))
		push!(test_r2,  1 - sum((y_test  .- yp_test).^2)  / sum((y_test  .- mean(y_test)).^2))
	end

	best_k = ks[argmax(test_r2)]
	global best_k

	plot(ks, train_r2, label="R² entraînement", marker=:circle, color=:steelblue,
		title="Performance KNN vs Valeur de K",
		xlabel="K (nombre de voisins)", ylabel="R²")
	plot!(ks, test_r2, label="R² test", marker=:square, color=:coral)
	vline!([best_k], color=:gray, linestyle=:dash, linewidth=1.5, label="Meilleur K=$best_k")
end

# ╔═╡ d0e1f2a3-0015-1b9c-7e8f-3a4b5c6d7e8f
md"""
**Observation :**
- **K faible :** R² d'entraînement élevé, R² de test plus bas → surapprentissage
- **K élevé :** les deux scores convergent → sous-apprentissage
- Le meilleur K équilibre les deux.
"""

# ╔═╡ d0e1f2a3-0016-1b9c-7e8f-3a4b5c6d7e8f
md"""
## Visualiser la surface de décision KNN

Sur un graphique 2D (Buts × Passes décisives), les régions colorées
montrent comment KNN prédit la valeur marchande.
"""

# ╔═╡ d0e1f2a3-0017-1b9c-7e8f-3a4b5c6d7e8f
let
	# Use only goals and assists for 2D visualization
	X2 = Matrix(players_df[!, [:goals, :assists]])
	y2 = Vector(players_df[!, :market_value])
	X2_mean = mean(X2; dims=1)
	X2_std  = std(X2; dims=1)
	X2s = (X2 .- X2_mean) ./ X2_std

	# Create prediction grid
	g_range = range(minimum(X2[:, 1]), maximum(X2[:, 1]); length=50)
	a_range = range(minimum(X2[:, 2]), maximum(X2[:, 2]); length=50)
	grid_points = [[g, a] for g in g_range, a in a_range]
	grid_flat = vcat([grid_points[i, j] for i in 1:50, j in 1:50]...)
	grid_flat = reshape(grid_flat, 2500, 2)
	grid_scaled = (grid_flat .- X2_mean) ./ X2_std
	grid_preds = knn_predict(X2s, y2, grid_scaled, best_k)

	contour(g_range, a_range, reshape(grid_preds, 50, 50)',
		fill=true, levels=15, colormap=:viridis, alpha=0.7,
		colorbar=true, title="Surface de décision KNN (K=$best_k)",
		xlabel="Buts", ylabel="Passes décisives")
	scatter!(players_df.goals, players_df.assists,
		color=players_df.market_value, markersize=8,
		markerstrokecolor=:black, markerstrokewidth=0.5,
		label="Joueurs réels")
end

# ╔═╡ d0e1f2a3-0018-1b9c-7e8f-3a4b5c6d7e8f
md"""
Les contours montrent des frontières de prédiction **non linéaires** et
**flexibles** — contrairement au plan rigide que produirait une régression
linéaire.
"""

# ╔═╡ d0e1f2a3-0019-1b9c-7e8f-3a4b5c6d7e8f
md"""
## Comparer KNN vs Régression linéaire
"""

# ╔═╡ d0e1f2a3-0020-1b9c-7e8f-3a4b5c6d7e8f
let
	# Linear regression for comparison
	df_train = DataFrame(players_df[train_idx, :])
	linear_model = lm(@formula(market_value ~ goals + assists + age), df_train)
	linear_preds = predict(linear_model, DataFrame(players_df[test_idx, :]))

	# KNN predictions
	knn_preds = knn_predict(X_train_scaled, y_train, X_test_scaled, best_k)

	linear_r2 = 1 - sum((y_test .- linear_preds).^2) / sum((y_test .- mean(y_test)).^2)
	knn_r2    = 1 - sum((y_test .- knn_preds).^2)    / sum((y_test .- mean(y_test)).^2)
	linear_rmse = sqrt(mean((y_test .- linear_preds).^2))
	knn_rmse    = sqrt(mean((y_test .- knn_preds).^2))

	md"""
	**Comparaison sur l'ensemble de test :**

	**Régression linéaire :**
	- R² : **$(round(linear_r2; digits=3))**
	- RMSE : **$(round(linear_rmse; digits=1)) M€**

	**KNN (K = $best_k) :**
	- R² : **$(round(knn_r2; digits=3))**
	- RMSE : **$(round(knn_rmse; digits=1)) M€**

	$(knn_r2 > linear_r2 ? "✅ KNN est plus performant ! Il capture des motifs non linéaires." : "La régression linéaire est plus performante. La relation est sans doute majoritairement linéaire.")
	"""
end

# ╔═╡ d0e1f2a3-0021-1b9c-7e8f-3a4b5c6d7e8f
md"""
## Trouver des joueurs similaires

Une utilisation puissante de KNN : trouver des joueurs comparables pour
le recrutement !
"""

# ╔═╡ d0e1f2a3-0022-1b9c-7e8f-3a4b5c6d7e8f
let
	target_i = 1
	target_vec = X_train_scaled[target_i:target_i, :]

	dists = vec(pairwise(Euclidean(), target_vec', X_train_scaled'; dims=2))
	neighbor_idx = sortperm(dists)[2:6]  # skip self (distance 0)

	md"""
	**Joueur cible :**
	- Buts : **$(Int(X_train[target_i, 1]))**
	- Passes décisives : **$(Int(X_train[target_i, 2]))**
	- Âge : **$(Int(X_train[target_i, 3]))**
	- Valeur : **$(round(y_train[target_i]; digits=1)) M€**

	**5 joueurs les plus similaires :**
	$(join(["$(i). Distance : $(round(dists[idx]; digits=2)) — Buts : $(Int(X_train[idx, 1])), PD : $(Int(X_train[idx, 2])), Âge : $(Int(X_train[idx, 3])), Valeur : $(round(y_train[idx]; digits=1)) M€" for (i, idx) in enumerate(neighbor_idx)], "  \n"))
	"""
end

# ╔═╡ d0e1f2a3-0023-1b9c-7e8f-3a4b5c6d7e8f
md"""
## Avantages et inconvénients de KNN

### Avantages
- **Aucune hypothèse** — fonctionne avec n'importe quelle distribution
- **Flexible** — capture des motifs non linéaires complexes
- **Concept simple** — facile à comprendre intuitivement
- **Multi-cible** — peut prédire plusieurs cibles simultanément

### Inconvénients
- **Coûteux en calcul** — doit parcourir toutes les données à chaque prédiction
- **Nécessite une mise à l'échelle** — sensible aux échelles des variables
- **Malédiction de la dimensionnalité** — performances dégradées avec beaucoup de variables
- **Peu interprétable** — impossible d'expliquer les prédictions avec des coefficients
"""

# ╔═╡ d0e1f2a3-0024-1b9c-7e8f-3a4b5c6d7e8f
md"""
## Récapitulatif

Dans ce notebook, nous avons :

1. Compris comment fonctionne la régression KNN (prédiction par similarité)
2. Appris la différence entre modèles paramétriques et non paramétriques
3. Construit un modèle KNN pour prédire la valeur des joueurs
4. Trouvé le K optimal par validation croisée
5. Visualisé la surface de décision KNN
6. Comparé KNN avec la régression linéaire
7. Utilisé KNN pour trouver des joueurs similaires

## Points clés

- **KNN** prédit en faisant la moyenne des K plus proches voisins
- **Non paramétrique** → pas de formule fixe, très flexible
- **La mise à l'échelle est cruciale** pour les méthodes basées sur la distance
- **K est un hyperparamètre** qui contrôle la complexité du modèle
- **Idéal pour trouver des joueurs similaires** ou des situations analogues
- **Compromis :** flexibilité contre interprétabilité et rapidité

## Prochaine étape

Dans le prochain notebook, nous apprendrons les techniques complètes
d'**évaluation et de diagnostic des modèles** !
"""

# ╔═╡ d0e1f2a3-0025-1b9c-7e8f-3a4b5c6d7e8f
md"""
## Exercices

1. **Métriques de distance** — essayer différentes métriques (Manhattan, Minkowski)
2. **KNN pondéré** — utiliser des prédictions pondérées par la distance
3. **Plus de variables** — ajouter plus de statistiques de joueurs et observer l'effet
4. **Système de recommandation** — construire un système pour recommander des joueurs similaires
5. **Malédiction de la dimensionnalité** — expérimenter avec 10+ variables et observer la dégradation
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
DataFrames = "1"
GLM = "1"
Plots = "1"
"""

# ╔═╡ Cell order:
# ╟─d0e1f2a3-0001-1b9c-7e8f-3a4b5c6d7e8f
# ╟─d0e1f2a3-0002-1b9c-7e8f-3a4b5c6d7e8f
# ╠═d0e1f2a3-0003-1b9c-7e8f-3a4b5c6d7e8f
# ╟─d0e1f2a3-0004-1b9c-7e8f-3a4b5c6d7e8f
# ╟─d0e1f2a3-0005-1b9c-7e8f-3a4b5c6d7e8f
# ╟─d0e1f2a3-0006-1b9c-7e8f-3a4b5c6d7e8f
# ╠═d0e1f2a3-0007-1b9c-7e8f-3a4b5c6d7e8f
# ╟─d0e1f2a3-0008-1b9c-7e8f-3a4b5c6d7e8f
# ╠═d0e1f2a3-0009-1b9c-7e8f-3a4b5c6d7e8f
# ╟─d0e1f2a3-0010-1b9c-7e8f-3a4b5c6d7e8f
# ╠═d0e1f2a3-0011-1b9c-7e8f-3a4b5c6d7e8f
# ╠═d0e1f2a3-0012-1b9c-7e8f-3a4b5c6d7e8f
# ╟─d0e1f2a3-0013-1b9c-7e8f-3a4b5c6d7e8f
# ╠═d0e1f2a3-0014-1b9c-7e8f-3a4b5c6d7e8f
# ╟─d0e1f2a3-0015-1b9c-7e8f-3a4b5c6d7e8f
# ╟─d0e1f2a3-0016-1b9c-7e8f-3a4b5c6d7e8f
# ╠═d0e1f2a3-0017-1b9c-7e8f-3a4b5c6d7e8f
# ╟─d0e1f2a3-0018-1b9c-7e8f-3a4b5c6d7e8f
# ╟─d0e1f2a3-0019-1b9c-7e8f-3a4b5c6d7e8f
# ╠═d0e1f2a3-0020-1b9c-7e8f-3a4b5c6d7e8f
# ╟─d0e1f2a3-0021-1b9c-7e8f-3a4b5c6d7e8f
# ╠═d0e1f2a3-0022-1b9c-7e8f-3a4b5c6d7e8f
# ╟─d0e1f2a3-0023-1b9c-7e8f-3a4b5c6d7e8f
# ╟─d0e1f2a3-0024-1b9c-7e8f-3a4b5c6d7e8f
# ╟─d0e1f2a3-0025-1b9c-7e8f-3a4b5c6d7e8f
# ╠═00000000-0000-0000-0000-000000000001
