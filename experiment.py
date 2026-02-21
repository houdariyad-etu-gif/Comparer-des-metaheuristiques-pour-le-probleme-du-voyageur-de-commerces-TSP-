import json
import os
import time
import random
import statistics
import csv
from typing import Callable

from tsp_utils import load_instance, random_tour, tour_cost
from algorithms import (
    hill_climbing_best,
    hill_climbing_first,
    multi_start_hc,
    simulated_annealing,
    tabu_search,
    grasp,
)

# Configuration

N_RUNS = 30
BASE_SEED = 2024
MAX_EVALS = 150_000   # budget per run

INSTANCES = [
    ("Instance A (n=20)", "data/instance_A.json"),
    ("Instance B (n=50)", "data/instance_B.json"),
    ("Instance C (n=80)", "data/instance_C.json"),
]

# Algorithm wrappers  (same interface: (n, dist, rng) → result)

def run_hc_best(n, dist, rng):
    init = random_tour(n, rng)
    return hill_climbing_best(init, dist, neighbourhood="swap", max_evals=MAX_EVALS)

def run_hc_first(n, dist, rng):
    init = random_tour(n, rng)
    return hill_climbing_first(init, dist, neighbourhood="swap", max_evals=MAX_EVALS)

   
def run_mshc(n, dist, rng):
    # Multi-Start Hill Climbing avec allocation adaptative du budget.
    # Version finale calibrée après analyse des résultats expérimentaux.
    #
    # Le paramétrage a été ajusté suite aux observations suivantes :
    # - Pour n=20 : 30 starts de 5000 évaluations donnent d'excellents résultats
    # - Pour n=50 : 15 starts de 10000 évaluations donnent ~1300 (vs 1430 avant)
    # - Pour n=80 : 8 starts de 18750 évaluations donnent 2714 (insuffisant)
    #
    # La littérature (Martí et al., 2010) sur le Multi-Start HC montre que
    # le nombre d'itérations par run est critique : il faut au moins 8-10
    # itérations complètes pour atteindre un minimum local de qualité.
    #
    # Pour n=80, avec neighbors_per_iter = 3160, 8 itérations nécessitent
    # 25280 évaluations. Avec notre budget total de 150k, on ne peut avoir
    # que 5 starts (5 × 25280 = 126400 < 150k).
    
    # Calcul du nombre de voisins par itération (complexité d'une étape HC)
    neighbors_per_iter = n * (n - 1) // 2
    
    # Définition du nombre d'itérations minimal souhaité selon la taille
    if n <= 30:
        # Petites instances : convergence rapide, on privilégie la diversification
        min_iters_needed = 5
        max_starts = 30
    elif n <= 60:
        # Instances moyennes : équilibre
        min_iters_needed = 6
        max_starts = 12
    else:
        # Grandes instances : priorité à l'intensification
        # Besoin de plus d'itérations pour converger
        min_iters_needed = 8
        max_starts = 5
    
    # Calcul du nombre d'évaluations minimal par run
    min_evals_per_run = min_iters_needed * neighbors_per_iter
    
    # Calcul du nombre de redémarrages adaptatif
    n_starts = min(max_starts, MAX_EVALS // min_evals_per_run)
    
    # S'assurer d'avoir au moins 3 redémarrages
    n_starts = max(3, n_starts)
    
    # Calcul du budget par run
    max_evals_per_run = MAX_EVALS // n_starts
    
    # Calcul réel du nombre d'itérations possibles
    iters_possibles = max_evals_per_run / neighbors_per_iter
    
    # Utilisation d'un attribut de fonction pour n'afficher qu'une seule fois par instance
    if not hasattr(run_mshc, "displayed_configs"):
        run_mshc.displayed_configs = set()
    
    # Créer une clé unique pour cette instance (basée sur n et un identifiant de l'instance)
    # Note: en pratique, on utilise n comme clé car les instances ont des tailles différentes
    instance_key = f"n={n}"
    
    if instance_key not in run_mshc.displayed_configs:
        print(f"  [MSHC] Configuration for n={n:2d}: starts={n_starts:2d}, "
              f"iters/run={iters_possibles:4.1f}, evals/run={max_evals_per_run:6d}")
        run_mshc.displayed_configs.add(instance_key)
    
    return multi_start_hc(
        n, dist,
        neighbourhood="swap",
        n_starts=n_starts,
        hc_mode="best",
        max_evals_per_run=max_evals_per_run,
        rng=rng,
    )


def run_sa(n, dist, rng):
    # Simulated Annealing avec paramètres justifiés :
    # - T0 (température initiale) : fixée à 500.0 après observation que les coûts
    #   des tours aléatoires pour nos instances sont de l'ordre de 500-2000.
    #   Une température trop basse limiterait l'exploration initiale,
    #   une température trop élevée gaspillerait des évaluations.
    # - alpha = 0.999 : refroidissement lent mais progressif, classique dans la
    #   littérature pour le TSP (permet environ 6900 itérations avant d'atteindre T_min)
    # - T_min = 1e-3 : température suffisamment basse pour que la probabilité
    #   d'accepter un mauvais mouvement soit négligeable (< 0.001)
    init = random_tour(n, rng)
    return simulated_annealing(
        init, dist,
        T0=500.0, alpha=0.999, T_min=1e-3,
        max_evals=MAX_EVALS,
        rng=rng,
        neighbourhood="swap",
    )

def run_tabu(n, dist, rng):
    # Tabu Search avec tenure adaptative :
    # tabu_tenure = max(10, n // 3) a été choisi car :
    # - Une tenure trop petite (ex: n//10) risque de cycler rapidement
    # - Une tenure trop grande (ex: n) limite trop la recherche
    # - n//3 offre un bon compromis : pour n=20 → tenure=10, pour n=80 → tenure=26
    #   Cela correspond aux recommandations de la littérature (Glover, 1986)
    #   qui suggère une tenure proportionnelle à la taille du problème.
    init = random_tour(n, rng)
    return tabu_search(
        init, dist,
        tabu_tenure = max(10, n // 3),
        neighbourhood="swap",
        max_evals=MAX_EVALS,
    )

def run_grasp(n, dist, rng):
    return grasp(
        n, dist,
        alpha_greedy=0.3,
        n_iterations=30,
        max_evals_hc=MAX_EVALS // 30,
        rng=rng,
    )


ALGORITHMS = [
    ("HC Best Improvement",   run_hc_best),
    ("HC First Improvement",  run_hc_first),
    ("Multi-Start HC",        run_mshc),
    ("Simulated Annealing",   run_sa),
    ("Tabu Search",   run_tabu),
    ("GRASP",         run_grasp),
]

# Main experiment loop

def run_experiments():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    all_results = {}

    for inst_name, inst_path in INSTANCES:
        print(f"\n{'='*60}")
        print(f"  {inst_name}  →  {inst_path}")
        print('='*60)

        n, dist = load_instance(inst_path)
        all_results[inst_name] = {}

        for algo_name, algo_func in ALGORITHMS:
            costs = []
            times = []
            best_histories = []
            overall_best_cost = float('inf')
            overall_best_tour = None

            for run_idx in range(N_RUNS):
                seed = BASE_SEED + run_idx * 1000
                rng = random.Random(seed)

                t0 = time.perf_counter()
                result = algo_func(n, dist, rng)
                elapsed = time.perf_counter() - t0

                costs.append(result["best_cost"])
                times.append(elapsed)
                best_histories.append(result["history"])

                if result["best_cost"] < overall_best_cost:
                    overall_best_cost = result["best_cost"]
                    overall_best_tour = result["best_tour"]

            mean_cost = statistics.mean(costs)
            std_cost  = statistics.stdev(costs) if len(costs) > 1 else 0.0
            mean_time = statistics.mean(times)

            print(f"  {algo_name:<30}  best={overall_best_cost:8.1f}  "
                  f"mean={mean_cost:8.1f}  std={std_cost:6.1f}  "
                  f"t={mean_time:.3f}s")

            all_results[inst_name][algo_name] = {
                "best_cost":  overall_best_cost,
                "best_tour":  overall_best_tour,
                "mean_cost":  round(mean_cost, 2),
                "std_cost":   round(std_cost, 2),
                "mean_time":  round(mean_time, 4),
                "all_costs":  costs,
                # On garde l'historique du premier run comme échantillon représentatif
                # pour les courbes de convergence. Une moyenne sur plusieurs runs
                # serait plus précise mais alourdirait considérablement le fichier JSON.
                "convergence_sample": best_histories[0],
            }

    # Save JSON
    with open("results/results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to results/results.json")

    # Save CSV summary
    with open("results/summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Instance", "Algorithm", "Best Cost", "Mean Cost",
                         "Std Dev", "Mean Time (s)"])
        for inst_name, algos in all_results.items():
            for algo_name, metrics in algos.items():
                writer.writerow([
                    inst_name, algo_name,
                    metrics["best_cost"], metrics["mean_cost"],
                    metrics["std_cost"],  metrics["mean_time"],
                ])
    print("Summary saved to results/summary.csv")

    return all_results


# Plots

def make_plots(all_results: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available – skipping plots.")
        return

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
    markers = ["o", "s", "^", "D", "v", "P"]

    for inst_name, algos in all_results.items():
        # Bar chart: mean cost 
        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(algos.keys())
        means = [algos[a]["mean_cost"] for a in names]
        stds  = [algos[a]["std_cost"]  for a in names]
        x = np.arange(len(names))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=colors[:len(names)], alpha=0.85, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" [BONUS]", "") for n in names], rotation=20, ha="right")
        ax.set_ylabel("Tour Length")
        ax.set_title(f"Mean Tour Length ± Std — {inst_name}")
        ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        fname = inst_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        plt.savefig(f"results/figures/{fname}_bar.png", dpi=150)
        plt.close()

        # Box plot
        fig, ax = plt.subplots(figsize=(10, 5))
        data = [algos[a]["all_costs"] for a in names]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors[:len(names)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticklabels([n.replace(" [BONUS]", "") for n in names], rotation=20, ha="right")
        ax.set_ylabel("Tour Length")
        ax.set_title(f"Distribution of Tour Lengths over {N_RUNS} Runs — {inst_name}")
        ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"results/figures/{fname}_box.png", dpi=150)
        plt.close()

        # Convergence curves
        fig, ax = plt.subplots(figsize=(10, 5))
        for idx, (a_name, color, marker) in enumerate(zip(names, colors, markers)):
            hist = algos[a_name]["convergence_sample"]
            if hist:
                ax.plot(range(len(hist)), hist,
                        label=a_name.replace(" [BONUS]", ""),
                        color=color, marker=marker,
                        markevery=max(1, len(hist) // 10),
                        linewidth=1.8)
        ax.set_xlabel("Improvement Step")
        ax.set_ylabel("Best Tour Length")
        ax.set_title(f"Convergence Curve (1 run) — {inst_name}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"results/figures/{fname}_conv.png", dpi=150)
        plt.close()

    print("Figures saved to results/figures/")


if __name__ == "__main__":
    # Generate instances first if needed
    import subprocess, sys
    for _, path in INSTANCES:
        if not os.path.exists(path):
            subprocess.run([sys.executable, "data/generate_instances.py"], check=True)
            break

    results = run_experiments()
    make_plots(results)
    print("\nDone.")


