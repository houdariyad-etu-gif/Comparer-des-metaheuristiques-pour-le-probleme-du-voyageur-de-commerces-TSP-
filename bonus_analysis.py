import random
import statistics
import os
import math

from tsp_utils import load_instance, random_tour, build_distance_matrix
from algorithms import simulated_annealing

# Configuration

N_RUNS    = 20       # runs par configuration
MAX_EVALS = 150_000
BASE_SEED = 2024

os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

# BONUS 1 — Étude des paramètres SA

def study_sa_parameters():
    """
    Teste différentes combinaisons de T0 et alpha
    sur l'Instance B (50 villes).
    Mesure : coût moyen sur N_RUNS runs.
    """
    print("\n" + "="*60)
    print("  BONUS 1 — Étude des paramètres du Recuit Simulé")
    print("  Instance B (n=50)")
    print("="*60)

    n, dist = load_instance("data/instance_B.json")

    # Grilles de paramètres à tester
    T0_values    = [100, 500, 1000, 2000]
    alpha_values = [0.99, 0.995, 0.998, 0.999]

    results = {}

    print(f"\n{'T0':<8} {'alpha':<8} {'Best':>8} {'Mean':>8} {'Std':>7}")
    print("-" * 45)

    for T0 in T0_values:
        for alpha in alpha_values:
            costs = []
            for run in range(N_RUNS):
                rng  = random.Random(BASE_SEED + run * 100)
                init = random_tour(n, rng)
                res  = simulated_annealing(
                    init, dist,
                    T0=T0, alpha=alpha, T_min=1e-3,
                    max_evals=MAX_EVALS, rng=rng
                )
                costs.append(res["best_cost"])

            best = min(costs)
            mean = statistics.mean(costs)
            std  = statistics.stdev(costs)
            results[(T0, alpha)] = {"best": best, "mean": mean, "std": std}

            print(f"{T0:<8} {alpha:<8} {best:>8.1f} {mean:>8.1f} {std:>7.1f}")

    # Meilleure configuration
    best_config = min(results, key=lambda k: results[k]["mean"])
    print(f"\n✅ Meilleure configuration : T0={best_config[0]}, alpha={best_config[1]}")
    print(f"   Mean={results[best_config]['mean']:.1f}, Std={results[best_config]['std']:.1f}")

    return results


# BONUS 2 — Instance TSPLIB berlin52

# Coordonnées exactes de berlin52 (source : TSPLIB)
BERLIN52_CITIES = [
    (565,575),(25,185),(345,750),(945,685),(845,655),
    (880,660),(25,230),(525,1000),(580,1175),(650,1130),
    (1605,620),(1220,580),(1465,200),(1530,5),(845,680),
    (725,370),(145,665),(415,635),(510,875),(560,365),
    (300,465),(520,585),(480,415),(835,625),(975,580),
    (1215,245),(1320,315),(1250,400),(660,180),(410,250),
    (420,555),(575,665),(1150,1160),(700,580),(685,595),
    (685,610),(770,610),(795,645),(720,635),(760,650),
    (475,960),(95,260),(875,920),(700,500),(555,815),
    (830,485),(1170,65),(830,610),(605,625),(595,360),
    (1340,725),(1740,245)
]
BERLIN52_OPTIMAL = 7542  # solution optimale connue (TSPLIB)


def compare_tsplib():
    """
    Résout berlin52 avec tous nos algorithmes et compare
    à la solution optimale connue (7542).
    """
    print("\n" + "="*60)
    print("  BONUS 2 — Comparaison TSPLIB : berlin52")
    print(f"  n=52 villes  |  Optimal connu = {BERLIN52_OPTIMAL}")
    print("="*60)

    from algorithms import (hill_climbing_best, hill_climbing_first,
                            multi_start_hc, simulated_annealing,
                            tabu_search, grasp)

    n    = len(BERLIN52_CITIES)
    dist = build_distance_matrix(BERLIN52_CITIES)

    algos = [
        ("HC Best Improvement",  lambda rng: hill_climbing_best(random_tour(n, rng), dist, max_evals=MAX_EVALS)),
        ("HC First Improvement", lambda rng: hill_climbing_first(random_tour(n, rng), dist, max_evals=MAX_EVALS)),
        ("Multi-Start HC",       lambda rng: multi_start_hc(n, dist, n_starts=30, max_evals_per_run=MAX_EVALS//30, rng=rng)),
        ("Simulated Annealing",  lambda rng: simulated_annealing(random_tour(n, rng), dist, T0=500, alpha=0.998, T_min=1e-3, max_evals=MAX_EVALS, rng=rng)),
        ("Tabu Search",          lambda rng: tabu_search(random_tour(n, rng), dist, tabu_tenure=max(5, n//5), max_evals=MAX_EVALS)),
        ("GRASP",                lambda rng: grasp(n, dist, alpha_greedy=0.3, n_iterations=30, max_evals_hc=MAX_EVALS//30, rng=rng)),
    ]

    print(f"\n{'Algorithme':<25} {'Best':>7} {'Mean':>7} {'Écart opt.':>11} {'%/opt':>7}")
    print("-" * 62)

    tsplib_results = {}
    for algo_name, algo_func in algos:
        costs = []
        for run in range(N_RUNS):
            rng = random.Random(BASE_SEED + run * 100)
            res = algo_func(rng)
            costs.append(res["best_cost"])

        best      = min(costs)
        mean      = statistics.mean(costs)
        gap       = best - BERLIN52_OPTIMAL
        gap_pct   = (gap / BERLIN52_OPTIMAL) * 100

        tsplib_results[algo_name] = {
            "best": best, "mean": mean, "gap": gap, "gap_pct": gap_pct
        }
        print(f"{algo_name:<25} {best:>7.0f} {mean:>7.1f} {gap:>+11.0f} {gap_pct:>6.1f}%")

    print(f"\n{'Optimal TSPLIB':<25} {BERLIN52_OPTIMAL:>7}")

    # Meilleur algo
    best_algo = min(tsplib_results, key=lambda k: tsplib_results[k]["best"])
    print(f"\n✅ Meilleur algorithme : {best_algo}")
    print(f"   Best={tsplib_results[best_algo]['best']:.0f}  "
          f"Écart={tsplib_results[best_algo]['gap_pct']:.1f}% de l'optimal")

    return tsplib_results


# Graphiques bonus

def make_bonus_plots(sa_results, tsplib_results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib non disponible — graphiques ignorés.")
        return

    colors = ["#2196F3","#F44336","#4CAF50","#FF9800","#9C27B0","#00BCD4","#795548","#607D8B",
              "#E91E63","#009688","#FF5722","#3F51B5","#8BC34A","#FFC107","#673AB7","#03A9F4"]

    # Heatmap paramètres SA 
    T0_values    = [100, 500, 1000, 2000]
    alpha_values = [0.99, 0.995, 0.998, 0.999]

    matrix = np.array([[sa_results[(T0, alpha)]["mean"]
                        for alpha in alpha_values]
                       for T0 in T0_values])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="YlOrRd_r", aspect="auto")
    ax.set_xticks(range(len(alpha_values)))
    ax.set_xticklabels(alpha_values)
    ax.set_yticks(range(len(T0_values)))
    ax.set_yticklabels(T0_values)
    ax.set_xlabel("alpha (taux de refroidissement)")
    ax.set_ylabel("T0 (température initiale)")
    ax.set_title("Coût moyen SA selon T0 et alpha — Instance B (n=50)")
    plt.colorbar(im, ax=ax, label="Coût moyen")
    for i in range(len(T0_values)):
        for j in range(len(alpha_values)):
            ax.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center",
                    fontsize=9, color="black")
    plt.tight_layout()
    plt.savefig("results/figures/bonus_sa_heatmap.png", dpi=150)
    plt.close()

    #Comparaison TSPLIB 
    fig, ax = plt.subplots(figsize=(10, 5))
    names   = list(tsplib_results.keys())
    bests   = [tsplib_results[a]["best"] for a in names]
    gaps    = [tsplib_results[a]["gap_pct"] for a in names]
    x       = np.arange(len(names))

    bars = ax.bar(x, bests, color=colors[:len(names)], alpha=0.85, edgecolor="black")
    ax.axhline(y=BERLIN52_OPTIMAL, color="red", linewidth=2,
               linestyle="--", label=f"Optimal TSPLIB = {BERLIN52_OPTIMAL}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Longueur du tour")
    ax.set_title("Comparaison avec l'optimal TSPLIB — berlin52 (n=52)")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f"+{gap:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig("results/figures/bonus_tsplib_comparison.png", dpi=150)
    plt.close()

    print("\n📊 Graphiques bonus sauvegardés :")
    print("   results/figures/bonus_sa_heatmap.png")
    print("   results/figures/bonus_tsplib_comparison.png")


if __name__ == "__main__":
    # Vérifier que les instances existent
    import subprocess, sys
    if not os.path.exists("data/instance_B.json"):
        subprocess.run([sys.executable, "data/generate_instances.py"], check=True)

    sa_results     = study_sa_parameters()
    tsplib_results = compare_tsplib()
    make_bonus_plots(sa_results, tsplib_results)

    print("\n✅ Analyse bonus terminée !")
    print("   Résultats dans results/figures/")
