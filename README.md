

## Structure du projet

Projet RO/
├── data/
│   ├── generate_instances.py   # script de génération des instances
│   ├── instance_A.json         # n = 20 villes  
│   ├── instance_B.json         # n = 50 villes 
│   └── instance_C.json         # n = 80 villes 
├── algorithms.py               # 8 métaheuristiques implémentées
├── tsp_utils.py                # utilitaires : distance, coût, voisinages
├── experiment.py               # protocole expérimental complet
├── bonus_analysis.py           # étude paramètres SA + comparaison TSPLIB
├── results/                    # créé automatiquement par experiment.py
│   ├── results.json
│   ├── summary.csv
│   └── figures/
│       ├── Instance_A_n20_bar.png
│       ├── Instance_A_n20_box.png
│       ├── Instance_A_n20_conv.png
│       ├── Instance_B_n50_bar.png
│       ├── Instance_B_n50_box.png
│       ├── Instance_B_n50_conv.png
│       ├── Instance_C_n80_bar.png
│       ├── Instance_C_n80_box.png
│       ├── Instance_C_n80_conv.png
│       ├── bonus_sa_heatmap.png
│       └── bonus_tsplib_comparison.png
└── README.md


## Installation

## Installation

Python 3.8+ requis. Installer les dépendances :

pip install matplotlib numpy


## Exécution — 4 commandes dans l'ordre

### Étape 1 — Générer les instances TSP

python data/generate_instances.py
```


Lance 30 runs x 8 algorithmes x 3 instances. Durée : 2 à 5 minutes.

### Étape 3 — Lancer les analyses bonus
```bash
python bonus_analysis.py
```

### Étape 4 — Consulter les résultats
```bash
start results\summary.csv
start results\figures


## Algorithmes implémentés

| # | Algorithme | Statut |
|---|-----------|--------|
| 1 | Hill-Climbing Best Improvement | Obligatoire |
| 2 | Hill-Climbing First Improvement | Obligatoire |
| 3 | Multi-Start Hill-Climbing | Obligatoire |
| 4 | Recuit Simulé (SA) | Obligatoire |
| 5 | Recherche Tabou | BONUS |
| 6 | GRASP | BONUS |
| 7 | VNS (Variable Neighbourhood Search) | BONUS |
| 8 | ILS (Iterated Local Search) | BONUS |



## Paramètres utilisés

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| MAX_EVALS | 150 000 | Budget d'évaluations par run |
| N_RUNS | 30 | Runs indépendants par algorithme |
| neighbourhood | swap | Voisinage par défaut (ou 2opt) |
| SA T0 | 1 000 | Température initiale (optimal par étude bonus) |
| SA alpha | 0.999 | Taux de refroidissement (optimal par étude bonus) |
| SA T_min | 1e-3 | Température minimale d'arrêt |
| Tabou tenure | n // 5 | Durée d'interdiction (adaptatif) |
| GRASP alpha | 0.3 | Paramètre de la liste restreinte RCL |
| ILS perturbation | 4 | Nombre de swaps pour la perturbation |



## Résultats principaux

| Instance | Meilleur algorithme | Meilleur coût |
|---------|-------------------|--------------|
| A (n=20) | SA / Tabou / MSHC | 351 |
| B (n=50) | Recuit Simulé | 768 |
| C (n=80) | Recuit Simulé | 1 159 |



## Analyses

### Étude des paramètres SA
- 16 combinaisons de T0 et alpha testées sur Instance B
- Meilleure configuration : T0=1000, alpha=0.999 (coût moyen = 808)

### Comparaison TSPLIB berlin52
- Optimal connu = 7542
- SA obtient 9313 (+23.5% de l'optimal)



