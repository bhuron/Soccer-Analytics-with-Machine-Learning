### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ c9d0e1f2-0003-0a8b-6d7e-2f3a4b5c6d7e
begin
	using DataFrames
	using Statistics
	using GLM
	using Plots
	gr()
	Plots.default(fontfamily="Helvetica", titlefontsize=12, guidefontsize=10)
end

# ╔═╡ c9d0e1f2-0001-0a8b-6d7e-2f3a4b5c6d7e
md"""
# Régression de Poisson — Prédire les buts dans les matchs

**Chapitre 6 · Techniques de régression pour l'analyse du football**

## Ce que vous allez apprendre

- Pourquoi la régression de Poisson est idéale pour les données de comptage
- La différence entre régression linéaire et régression de Poisson
- Construire un modèle de Poisson pour prédire les buts marqués
- Interpréter les coefficients sur une échelle logarithmique
- Visualiser les relations non linéaires
"""

# ╔═╡ c9d0e1f2-0002-0a8b-6d7e-2f3a4b5c6d7e
md"""
## Imports et configuration
"""

# ╔═╡ c9d0e1f2-0004-0a8b-6d7e-2f3a4b5c6d7e
md"""
## Pourquoi la régression de Poisson pour les buts ?

La régression linéaire est idéale pour les valeurs continues, mais
**les buts sont des comptages**. Voici pourquoi Poisson est meilleur :

| Problème | Régression linéaire | Régression de Poisson |
|---|---|---|
| **Valeurs fractionnaires** | Peut prédire 2,7 buts | Prédit des valeurs entières |
| **Valeurs négatives** | Peut prédire −1 but | Uniquement non négatif |
| **Relation** | Suppose une droite | Gère les motifs non linéaires |

La **régression de Poisson** est un type de **modèle linéaire généralisé (GLM)**
spécifiquement conçu pour les données de comptage.
"""

# ╔═╡ c9d0e1f2-0005-0a8b-6d7e-2f3a4b5c6d7e
md"""
## La distribution de Poisson

**Hypothèses clés :**
1. Les événements (buts) se produisent à un taux moyen constant
2. Les événements sont indépendants
3. Deux événements ne peuvent pas se produire exactement au même instant

Cela correspond bien aux matchs de football ! Les buts sont des événements
discrets qui se produisent indépendamment tout au long d'un match.
"""

# ╔═╡ c9d0e1f2-0006-0a8b-6d7e-2f3a4b5c6d7e
md"""
## Chargement des données — buts marqués

Nous allons prédire le nombre de buts marqués par une équipe en fonction
des tirs cadrés.
"""

# ╔═╡ c9d0e1f2-0007-0a8b-6d7e-2f3a4b5c6d7e
begin
	goals_df = DataFrame(
		team           = ["Équipe $i" for i in 1:20],
		shots_on_target = [5, 8, 12, 4, 6, 9, 10, 3, 7, 11, 5, 8, 12, 4, 6, 9, 10, 3, 7, 11],
		goals           = [1, 2, 3, 0, 1, 2, 2, 0, 1, 3, 1, 2, 4, 1, 1, 2, 3, 0, 2, 3],
	)

	goals_df
end

# ╔═╡ c9d0e1f2-0008-0a8b-6d7e-2f3a4b5c6d7e
md"""
## Visualiser les données
"""

# ╔═╡ c9d0e1f2-0009-0a8b-6d7e-2f3a4b5c6d7e
scatter(goals_df.shots_on_target, goals_df.goals,
	legend=false, markersize=8, color=:steelblue,
	title="Buts marqués vs Tirs cadrés",
	xlabel="Tirs cadrés", ylabel="Nombre de buts")

# ╔═╡ c9d0e1f2-0010-0a8b-6d7e-2f3a4b5c6d7e
md"""
**Remarquez :** les buts sont discrets (0, 1, 2, 3…) et non continus.
C'est précisément pourquoi nous avons besoin de la régression de Poisson !
"""

# ╔═╡ c9d0e1f2-0011-0a8b-6d7e-2f3a4b5c6d7e
md"""
## Construire le modèle de Poisson

Avec `GLM.jl` : `glm(formule, données, Poisson(), LogLink())`.
"""

# ╔═╡ c9d0e1f2-0012-0a8b-6d7e-2f3a4b5c6d7e
let
	model = glm(@formula(goals ~ shots_on_target), goals_df, Poisson(), LogLink())

	coeffs = coef(model)
	intercept = coeffs[1]
	slope = coeffs[2]
	multiplicative = exp(slope)

	md"""
	Modèle entraîné !

	**Coefficients :**
	- Ordonnée à l'origine (log) : **$(round(intercept; digits=4))**
	- Pente (log) : **$(round(slope; digits=4))**

	**Interprétation :**
	- Effet multiplicatif : exp(pente) = **$(round(multiplicative; digits=4))**
	- Chaque tir cadré supplémentaire multiplie les buts attendus par
	  **$(round(multiplicative; digits=2))**, soit une augmentation de
	  **$(round((multiplicative - 1) * 100; digits=1)) %**.
	"""
end

# ╔═╡ c9d0e1f2-0013-0a8b-6d7e-2f3a4b5c6d7e
md"""
## Interpréter les coefficients

**Important :** la régression de Poisson utilise une **fonction de lien
logarithmique**, donc les coefficients sont sur une échelle logarithmique.

Pour interpréter :
- **Exponentier le coefficient** : `exp(coef)` donne l'effet multiplicatif
- Si `coef = 0,15`, alors `exp(0,15) ≈ 1,16`
- **Interprétation :** chaque tir cadré supplémentaire multiplie les buts
  attendus par 1,16 (une augmentation de 16 %)
"""

# ╔═╡ c9d0e1f2-0014-0a8b-6d7e-2f3a4b5c6d7e
md"""
## Visualiser le modèle de Poisson

Contrairement à la régression linéaire, Poisson produit une
**courbe de prédiction incurvée**.
"""

# ╔═╡ c9d0e1f2-0015-0a8b-6d7e-2f3a4b5c6d7e
let
	model = glm(@formula(goals ~ shots_on_target), goals_df, Poisson(), LogLink())
	goals_df.predicted = predict(model, goals_df)

	sorted_df = sort(goals_df, :shots_on_target)

	scatter(sorted_df.shots_on_target, sorted_df.goals,
		label="Buts réels", markersize=8, color=:steelblue,
		title="Régression de Poisson — Buts vs Tirs cadrés",
		xlabel="Tirs cadrés", ylabel="Nombre de buts")

	plot!(sorted_df.shots_on_target, sorted_df.predicted,
		color=:red, linewidth=3, linestyle=:dash,
		label="Prédictions (Poisson)")
end

# ╔═╡ c9d0e1f2-0016-0a8b-6d7e-2f3a4b5c6d7e
md"""
**Remarquez :** la courbe de prédiction s'incurve vers le haut ! Elle
capture la relation non linéaire mieux qu'une ligne droite. La courbe
ne descend jamais en dessous de zéro — contrairement à ce que ferait
une régression linéaire.
"""

# ╔═╡ c9d0e1f2-0017-0a8b-6d7e-2f3a4b5c6d7e
md"""
## Faire des prédictions

Buts attendus pour des équipes avec différents nombres de tirs cadrés.
"""

# ╔═╡ c9d0e1f2-0018-0a8b-6d7e-2f3a4b5c6d7e
let
	model = glm(@formula(goals ~ shots_on_target), goals_df, Poisson(), LogLink())
	new_data = DataFrame(shots_on_target=[3, 6, 9, 12, 15])
	preds = predict(model, new_data)

	DataFrame(tirs_cadrés=new_data.shots_on_target, buts_attendus=round.(preds; digits=2))
end

# ╔═╡ c9d0e1f2-0019-0a8b-6d7e-2f3a4b5c6d7e
md"""
## Comparaison — Régression linéaire vs Poisson

Voyons la différence entre les deux approches côte à côte.
"""

# ╔═╡ c9d0e1f2-0020-0a8b-6d7e-2f3a4b5c6d7e
let
	# Poisson model
	poisson_model = glm(@formula(goals ~ shots_on_target), goals_df, Poisson(), LogLink())
	goals_df.poisson_pred = predict(poisson_model, goals_df)

	# Linear model
	linear_model = lm(@formula(goals ~ shots_on_target), goals_df)
	goals_df.linear_pred = predict(linear_model, goals_df)

	sorted_df = sort(goals_df, :shots_on_target)

	scatter(sorted_df.shots_on_target, sorted_df.goals,
		label="Buts réels", markersize=8, color=:steelblue,
		title="Poisson vs Régression linéaire",
		xlabel="Tirs cadrés", ylabel="Buts")

	plot!(sorted_df.shots_on_target, sorted_df.poisson_pred,
		color=:red, linewidth=3, linestyle=:dash, label="Poisson")

	plot!(sorted_df.shots_on_target, sorted_df.linear_pred,
		color=:blue, linewidth=3, linestyle=:dot, label="Linéaire")
end

# ╔═╡ c9d0e1f2-0021-0a8b-6d7e-2f3a4b5c6d7e
md"""
**Différence clé :**
- **Linéaire :** ligne droite (peut prédire des buts négatifs ou fractionnaires)
- **Poisson :** ligne courbe (toujours positive, mieux adaptée aux comptages)
"""

# ╔═╡ c9d0e1f2-0022-0a8b-6d7e-2f3a4b5c6d7e
md"""
## Évaluation du modèle

Pour les données de comptage, on utilise l'**erreur absolue moyenne (MAE)**.
"""

# ╔═╡ c9d0e1f2-0023-0a8b-6d7e-2f3a4b5c6d7e
let
	mae_poisson = mean(abs.(goals_df.goals .- goals_df.poisson_pred))
	mae_linear  = mean(abs.(goals_df.goals .- goals_df.linear_pred))

	md"""
	**Performance des modèles :**
	- MAE Poisson : **$(round(mae_poisson; digits=3))**
	- MAE Linéaire : **$(round(mae_linear; digits=3))**

	$(mae_poisson < mae_linear ? "✅ La régression de Poisson est plus performante pour ces données de comptage !" : "La régression linéaire donne des résultats similaires, mais Poisson reste plus appropriée pour les comptages.")
	"""
end

# ╔═╡ c9d0e1f2-0024-0a8b-6d7e-2f3a4b5c6d7e
md"""
## Récapitulatif

Dans ce notebook, nous avons :

1. Compris pourquoi la régression de Poisson est idéale pour les données de comptage
2. Construit un modèle de Poisson pour prédire les buts à partir des tirs cadrés
3. Utilisé GLM.jl avec `Poisson()` et `LogLink()`
4. Interprété les coefficients sur une échelle logarithmique
5. Visualisé la relation non linéaire
6. Comparé Poisson à la régression linéaire

## Points clés

- La **régression de Poisson** est conçue pour les données de comptage (buts, cartons, corners)
- Utilise une **fonction de lien log** → les coefficients doivent être exponentiels
- Produit des **prédictions courbes** qui respectent les propriétés des comptages
- Prédictions toujours **non négatives**
- Meilleure que la régression linéaire pour les résultats discrets

## Prochaine étape

Dans le prochain notebook, nous explorerons la **régression K-Nearest Neighbors
(KNN)** pour trouver des joueurs similaires !
"""

# ╔═╡ c9d0e1f2-0025-0a8b-6d7e-2f3a4b5c6d7e
md"""
## Exercices

1. **Prédicteurs multiples** — ajouter des variables comme la possession,
   le xG ou la force de l'adversaire.
2. **Prédiction du résultat** — prédire les buts des deux équipes et
   déterminer le vainqueur.
3. **Distribution de Poisson** — tracer la distribution de Poisson pour
   différentes valeurs de lambda.
4. **Données réelles** — appliquer à des données réelles de matchs de
   StatsBomb ou similaire.
5. **Surdispersion** — rechercher et tester si les données présentent
   une surdispersion.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
DataFrames = "1"
GLM = "1"
Plots = "1"
"""

# ╔═╡ Cell order:
# ╟─c9d0e1f2-0001-0a8b-6d7e-2f3a4b5c6d7e
# ╟─c9d0e1f2-0002-0a8b-6d7e-2f3a4b5c6d7e
# ╠═c9d0e1f2-0003-0a8b-6d7e-2f3a4b5c6d7e
# ╟─c9d0e1f2-0004-0a8b-6d7e-2f3a4b5c6d7e
# ╟─c9d0e1f2-0005-0a8b-6d7e-2f3a4b5c6d7e
# ╟─c9d0e1f2-0006-0a8b-6d7e-2f3a4b5c6d7e
# ╠═c9d0e1f2-0007-0a8b-6d7e-2f3a4b5c6d7e
# ╟─c9d0e1f2-0008-0a8b-6d7e-2f3a4b5c6d7e
# ╠═c9d0e1f2-0009-0a8b-6d7e-2f3a4b5c6d7e
# ╟─c9d0e1f2-0010-0a8b-6d7e-2f3a4b5c6d7e
# ╟─c9d0e1f2-0011-0a8b-6d7e-2f3a4b5c6d7e
# ╠═c9d0e1f2-0012-0a8b-6d7e-2f3a4b5c6d7e
# ╟─c9d0e1f2-0013-0a8b-6d7e-2f3a4b5c6d7e
# ╟─c9d0e1f2-0014-0a8b-6d7e-2f3a4b5c6d7e
# ╠═c9d0e1f2-0015-0a8b-6d7e-2f3a4b5c6d7e
# ╟─c9d0e1f2-0016-0a8b-6d7e-2f3a4b5c6d7e
# ╟─c9d0e1f2-0017-0a8b-6d7e-2f3a4b5c6d7e
# ╠═c9d0e1f2-0018-0a8b-6d7e-2f3a4b5c6d7e
# ╟─c9d0e1f2-0019-0a8b-6d7e-2f3a4b5c6d7e
# ╠═c9d0e1f2-0020-0a8b-6d7e-2f3a4b5c6d7e
# ╟─c9d0e1f2-0021-0a8b-6d7e-2f3a4b5c6d7e
# ╟─c9d0e1f2-0022-0a8b-6d7e-2f3a4b5c6d7e
# ╠═c9d0e1f2-0023-0a8b-6d7e-2f3a4b5c6d7e
# ╟─c9d0e1f2-0024-0a8b-6d7e-2f3a4b5c6d7e
# ╟─c9d0e1f2-0025-0a8b-6d7e-2f3a4b5c6d7e
# ╠═00000000-0000-0000-0000-000000000001
