### A Pluto.jl notebook ###
# v1.0.1

using Markdown
using InteractiveUtils

# ╔═╡ b8c9d0e1-0003-9f7a-5c6d-1e2f3a4b5c6d
begin
	using DataFrames
	using Statistics
	using GLM
	using Plots
	gr()
	Plots.default(fontfamily="Helvetica", titlefontsize=12, guidefontsize=10)
end

# ╔═╡ b8c9d0e1-0001-9f7a-5c6d-1e2f3a4b5c6d
md"""
# Régression linéaire — Prédire la valeur marchande des joueurs

**Chapitre 6 · Techniques de régression pour l'analyse du football**

## Ce que vous allez apprendre

- Comprendre les fondamentaux de la régression linéaire
- Construire un modèle pour prédire la valeur marchande des joueurs
- Visualiser la droite de meilleur ajustement
- Évaluer la performance avec R²
- Analyser les résidus pour diagnostiquer le modèle
"""

# ╔═╡ b8c9d0e1-0002-9f7a-5c6d-1e2f3a4b5c6d
md"""
## Imports et configuration
"""

# ╔═╡ b8c9d0e1-0004-9f7a-5c6d-1e2f3a4b5c6d
md"""
## L'intuition — trouver la droite de meilleur ajustement

La **régression linéaire** trouve une relation en ligne droite entre une
variable d'entrée (le prédicteur) et une variable de sortie (la cible).

**Pensez-y comme un recruteur qui estime la valeur d'un joueur :**
- Plus de buts marqués → valeur marchande plus élevée
- Plus de passes décisives → valeur marchande plus élevée
- Âge plus jeune → valeur potentiellement plus élevée

La régression linéaire fait cela mathématiquement en trouvant la
**meilleure droite** à travers les données.

**L'équation :** `Valeur = pente × Buts + ordonnée à l'origine`
"""

# ╔═╡ b8c9d0e1-0005-9f7a-5c6d-1e2f3a4b5c6d
md"""
## Chargement des données — valeurs marchandes des joueurs

Nous utilisons des données simulées représentant des attaquants d'un
championnat européen de première division.
"""

# ╔═╡ b8c9d0e1-0006-9f7a-5c6d-1e2f3a4b5c6d
begin
	players = DataFrame(
		name       = ["Joueur A", "Joueur B", "Joueur C", "Joueur D",
		              "Joueur E", "Joueur F", "Joueur G"],
		goals      = [22, 15, 8, 25, 12, 18, 5],
		value_M    = [90, 60, 25, 100, 45, 70, 15],
	)

	md"""
	| Joueur | Buts | Valeur (M€) |
	|---|---|---|
	$(join(["| $(r.name) | $(r.goals) | $(r.value_M) |" for r in eachrow(players)], "\n"))
	"""
end

# ╔═╡ b8c9d0e1-0007-9f7a-5c6d-1e2f3a4b5c6d
md"""
## Visualiser la relation

Avant de construire un modèle, il faut toujours visualiser ! Un nuage
de points montre la relation entre les buts et la valeur marchande.
"""

# ╔═╡ b8c9d0e1-0008-9f7a-5c6d-1e2f3a4b5c6d
scatter(players.goals, players.value_M,
	legend=false, markersize=10, color=:steelblue,
	title="Valeur marchande vs Buts marqués",
	xlabel="Buts marqués la saison dernière",
	ylabel="Valeur marchande (M€)")

# ╔═╡ b8c9d0e1-0009-9f7a-5c6d-1e2f3a4b5c6d
md"""
**Observation :** il y a une relation positive claire ! Les joueurs qui
marquent plus de buts ont tendance à avoir des valeurs marchandes plus
élevées.
"""

# ╔═╡ b8c9d0e1-0010-9f7a-5c6d-1e2f3a4b5c6d
md"""
## Construire le modèle de régression linéaire

Avec `GLM.jl`, la bibliothèque standard de régression de Julia.
"""

# ╔═╡ b8c9d0e1-0011-9f7a-5c6d-1e2f3a4b5c6d
let
	model = lm(@formula(value_M ~ goals), players)
	coeffs = coef(model)
	intercept = coeffs[1]
	slope = coeffs[2]

	md"""
	Modèle entraîné avec succès !

	**Paramètres du modèle :**
	| Paramètre | Valeur |
	|---|---|
	| Ordonnée à l'origine | **$(round(intercept; digits=2))** |
	| Pente (coefficient Buts) | **$(round(slope; digits=2))** |

	**Équation du modèle :** Valeur = $(round(slope; digits=2)) × Buts + $(round(intercept; digits=2))
	"""
end

# ╔═╡ b8c9d0e1-0012-9f7a-5c6d-1e2f3a4b5c6d
md"""
## Interpréter les coefficients

**Pente (≈ 4,15) :** pour chaque but supplémentaire marqué, la valeur
marchande d'un joueur augmente d'environ 4,15 M€.

**Ordonnée à l'origine (≈ −5,5) :** la valeur de base quand buts = 0.
Elle est négative, ce qui n'a pas de sens dans le monde réel. C'est
fréquent dans les modèles simples et souligne que l'ordonnée à l'origine
est souvent une nécessité mathématique pour positionner correctement
la droite.

**Perspective football :** la pente quantifie la relation entre la
capacité à marquer et la valeur marchande. Cela pourrait aider les clubs
à prendre des décisions de transfert basées sur les données !
"""

# ╔═╡ b8c9d0e1-0013-9f7a-5c6d-1e2f3a4b5c6d
md"""
## Visualiser le modèle

Traçons la droite de régression sur notre nuage de points.
"""

# ╔═╡ b8c9d0e1-0014-9f7a-5c6d-1e2f3a4b5c6d
let
	model = lm(@formula(value_M ~ goals), players)
	preds = predict(model, players)

	scatter(players.goals, players.value_M,
		legend=:topleft, markersize=10, color=:steelblue,
		label="Données réelles",
		title="Régression linéaire — Valeur marchande vs Buts",
		xlabel="Buts marqués la saison dernière",
		ylabel="Valeur marchande (M€)")

	plot!(players.goals, preds,
		color=:red, linewidth=3, label="Droite de régression")
end

# ╔═╡ b8c9d0e1-0015-9f7a-5c6d-1e2f3a4b5c6d
md"""
La ligne rouge représente les prédictions de notre modèle. Elle minimise
la distance globale à tous les points de données — c'est le principe des
**moindres carrés ordinaires**.
"""

# ╔═╡ b8c9d0e1-0016-9f7a-5c6d-1e2f3a4b5c6d
md"""
## Évaluer la performance — R²

**R² (coefficient de détermination)** indique quel pourcentage de la
variation de la valeur marchande peut être expliqué par les buts marqués.

- R² = 1 : ajustement parfait
- R² = 0 : le modèle n'explique rien
- R² = 0,95 : le modèle explique 95 % de la variation
"""

# ╔═╡ b8c9d0e1-0017-9f7a-5c6d-1e2f3a4b5c6d
let
	model = lm(@formula(value_M ~ goals), players)
	r2 = r²(model)

	md"""
	**R² = $(round(r2; digits=3))**

	**Interprétation :** $(round(r2 * 100; digits=1)) % de la variation
	des valeurs marchandes peut être expliquée par le nombre de buts
	marqués.
	"""
end

# ╔═╡ b8c9d0e1-0018-9f7a-5c6d-1e2f3a4b5c6d
md"""
## Analyse des résidus

Les **résidus** sont les différences entre les valeurs réelles et les
valeurs prédites. Un bon modèle doit avoir des résidus répartis
aléatoirement autour de zéro, sans motif visible.
"""

# ╔═╡ b8c9d0e1-0019-9f7a-5c6d-1e2f3a4b5c6d
let
	model = lm(@formula(value_M ~ goals), players)
	preds = predict(model, players)
	resids = players.value_M .- preds

	scatter(preds, resids,
		legend=false, markersize=10, color=:darkblue,
		title="Graphique des résidus",
		xlabel="Valeur marchande prédite (M€)",
		ylabel="Résidus (M€)")

	hline!([0], color=:red, linestyle=:dash, linewidth=2, label="Zéro")
end

# ╔═╡ b8c9d0e1-0020-9f7a-5c6d-1e2f3a4b5c6d
md"""
Si les résidus montrent un motif (par exemple en forme de U), la
régression linéaire n'est peut-être pas le meilleur choix. Une
dispersion aléatoire autour de zéro = bon modèle !
"""

# ╔═╡ b8c9d0e1-0021-9f7a-5c6d-1e2f3a4b5c6d
md"""
## Faire des prédictions

Maintenant, nous pouvons prédire les valeurs marchandes de nouveaux
joueurs !
"""

# ╔═╡ b8c9d0e1-0022-9f7a-5c6d-1e2f3a4b5c6d
let
	model = lm(@formula(value_M ~ goals), players)
	new = DataFrame(goals=[10, 20, 30])
	preds = predict(model, new)

	md"""
	**Prédictions de valeur marchande :**
	| Buts | Valeur prédite (M€) |
	|---|---|
	$(join(["| $(r.goals) | **$(round(p; digits=1))** |" for (r, p) in zip(eachrow(new), preds)], "\n"))
	"""
end

# ╔═╡ b8c9d0e1-0023-9f7a-5c6d-1e2f3a4b5c6d
md"""
## Récapitulatif

Dans ce notebook, nous avons :

1. Compris les fondamentaux de la régression linéaire
2. Construit un modèle pour prédire la valeur marchande à partir des buts
3. Visualisé la droite de meilleur ajustement
4. Évalué la performance avec R²
5. Analysé les résidus
6. Fait des prédictions pour de nouveaux joueurs

## Points clés

- La régression linéaire trouve la **meilleure droite** à travers les données
- La **pente** quantifie la relation entre la variable d'entrée et la cible
- **R²** mesure dans quelle mesure le modèle explique la variation
- L'**analyse des résidus** aide à identifier les limites du modèle
- Les modèles simples sont interprétables mais peuvent manquer des motifs complexes

## Prochaine étape

Dans le prochain notebook, nous explorerons la **régression de Poisson**
pour prédire le nombre de buts dans les matchs !
"""

# ╔═╡ b8c9d0e1-0024-9f7a-5c6d-1e2f3a4b5c6d
md"""
## Exercices

1. **Ajouter plus de variables** — étendre le modèle pour inclure les
   passes décisives et l'âge (`DataFrame(goals=..., assists=..., age=...)`).
2. **Régression linéaire multiple** — construire un modèle avec 2+
   variables.
3. **Métriques supplémentaires** — calculer la RMSE et la MAE.
4. **Données réelles** — essayer avec des données réelles de joueurs
   provenant d'un jeu de données public.
5. **Intervalles de prédiction** — implémenter des intervalles de
   confiance pour les prédictions.
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
# ╟─b8c9d0e1-0001-9f7a-5c6d-1e2f3a4b5c6d
# ╟─b8c9d0e1-0002-9f7a-5c6d-1e2f3a4b5c6d
# ╠═b8c9d0e1-0003-9f7a-5c6d-1e2f3a4b5c6d
# ╟─b8c9d0e1-0004-9f7a-5c6d-1e2f3a4b5c6d
# ╟─b8c9d0e1-0005-9f7a-5c6d-1e2f3a4b5c6d
# ╠═b8c9d0e1-0006-9f7a-5c6d-1e2f3a4b5c6d
# ╟─b8c9d0e1-0007-9f7a-5c6d-1e2f3a4b5c6d
# ╠═b8c9d0e1-0008-9f7a-5c6d-1e2f3a4b5c6d
# ╟─b8c9d0e1-0009-9f7a-5c6d-1e2f3a4b5c6d
# ╟─b8c9d0e1-0010-9f7a-5c6d-1e2f3a4b5c6d
# ╠═b8c9d0e1-0011-9f7a-5c6d-1e2f3a4b5c6d
# ╟─b8c9d0e1-0012-9f7a-5c6d-1e2f3a4b5c6d
# ╟─b8c9d0e1-0013-9f7a-5c6d-1e2f3a4b5c6d
# ╠═b8c9d0e1-0014-9f7a-5c6d-1e2f3a4b5c6d
# ╟─b8c9d0e1-0015-9f7a-5c6d-1e2f3a4b5c6d
# ╟─b8c9d0e1-0016-9f7a-5c6d-1e2f3a4b5c6d
# ╠═b8c9d0e1-0017-9f7a-5c6d-1e2f3a4b5c6d
# ╟─b8c9d0e1-0018-9f7a-5c6d-1e2f3a4b5c6d
# ╠═b8c9d0e1-0019-9f7a-5c6d-1e2f3a4b5c6d
# ╟─b8c9d0e1-0020-9f7a-5c6d-1e2f3a4b5c6d
# ╟─b8c9d0e1-0021-9f7a-5c6d-1e2f3a4b5c6d
# ╠═b8c9d0e1-0022-9f7a-5c6d-1e2f3a4b5c6d
# ╟─b8c9d0e1-0023-9f7a-5c6d-1e2f3a4b5c6d
# ╟─b8c9d0e1-0024-9f7a-5c6d-1e2f3a4b5c6d
# ╠═00000000-0000-0000-0000-000000000001
