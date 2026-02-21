import math
import random
import copy
from typing import List, Optional, Tuple

from tsp_utils import (
    Tour, DistMatrix,
    tour_cost, random_tour,
    swap_neighbours_gen, two_opt_neighbours_gen,
    delta_swap, delta_two_opt,
)


# Internal helpers


def _neighbourhood_gen(tour: Tour, neighbourhood: str):
    if neighbourhood == "swap":
        return swap_neighbours_gen(tour)
    elif neighbourhood == "2opt":
        return two_opt_neighbours_gen(tour)
    else:
        raise ValueError(f"Unknown neighbourhood: {neighbourhood}")


def _delta_pairs(n: int):
    """Generate all (i, j) pairs with i < j."""
    for i in range(n - 1):
        for j in range(i + 1, n):
            yield i, j



# 1. Hill-Climbing – Best Improvement


def hill_climbing_best(
    tour: Tour,
    dist: DistMatrix,
    neighbourhood: str = "swap",
    max_evals: int = 100_000,
) -> dict:
    """
    Best-improvement HC: at each step, evaluate all neighbours,
    move to the best improving neighbour (if any).
    """
    n = len(tour)
    current = tour[:]
    current_cost = tour_cost(current, dist)
    n_evals = n  # initial tour evaluation
    history = [current_cost]

    improved = True
    while improved and n_evals < max_evals:
        improved = False
        best_delta = 0.0
        best_i, best_j = -1, -1

        for i, j in _delta_pairs(n):
            if neighbourhood == "swap":
                delta = delta_swap(current, i, j, dist)
            else:  # 2opt
                delta = delta_two_opt(current, i, j, dist)
            n_evals += 1
            if delta < best_delta:
                best_delta = delta
                best_i, best_j = i, j
            if n_evals >= max_evals:
                break

        if best_delta < 0:
            if neighbourhood == "swap":
                current[best_i], current[best_j] = current[best_j], current[best_i]
            else:
                current[best_i:best_j + 1] = reversed(current[best_i:best_j + 1])
            current_cost = tour_cost(current, dist)  # Recalcul complet pour éviter erreurs
            history.append(current_cost)
            improved = True

    return {
        "best_tour": current,
        "best_cost": current_cost,
        "history": history,
        "n_evals": n_evals,
    }


# 2. Hill-Climbing – First Improvement


def hill_climbing_first(
    tour: Tour,
    dist: DistMatrix,
    neighbourhood: str = "swap",
    max_evals: int = 100_000,
) -> dict:
    """
    First-improvement HC: move to the first improving neighbour found.
    """
    n = len(tour)
    current = tour[:]
    current_cost = tour_cost(current, dist)
    n_evals = n
    history = [current_cost]

    improved = True
    while improved and n_evals < max_evals:
        improved = False
        for i, j in _delta_pairs(n):
            if neighbourhood == "swap":
                delta = delta_swap(current, i, j, dist)
            else:
                delta = delta_two_opt(current, i, j, dist)
            n_evals += 1
            if delta < 0:
                if neighbourhood == "swap":
                    current[i], current[j] = current[j], current[i]
                else:
                    current[i:j + 1] = reversed(current[i:j + 1])
                current_cost = tour_cost(current, dist)  # Recalcul complet
                history.append(current_cost)
                improved = True
                break
            if n_evals >= max_evals:
                break

    return {
        "best_tour": current,
        "best_cost": current_cost,
        "history": history,
        "n_evals": n_evals,
    }

# 3. Multi-Start Hill-Climbing

def multi_start_hc(
    n: int,
    dist: DistMatrix,
    neighbourhood: str = "swap",
    n_starts: int = 30,
    hc_mode: str = "best",
    max_evals_per_run: int = 10_000,
    rng: Optional[random.Random] = None,
) -> dict:
    """
    Launch HC from n_starts random starting tours; keep global best.
    hc_mode: "best" or "first"
    """
    if rng is None:
        rng = random.Random()

    hc_func = hill_climbing_best if hc_mode == "best" else hill_climbing_first

    global_best_cost = math.inf
    global_best_tour = None
    global_history = []
    total_evals = 0

    for _ in range(n_starts):
        init = random_tour(n, rng)
        result = hc_func(init, dist, neighbourhood, max_evals_per_run)
        total_evals += result["n_evals"]
        if result["best_cost"] < global_best_cost:
            global_best_cost = result["best_cost"]
            global_best_tour = result["best_tour"]
        global_history.append(global_best_cost)

    return {
        "best_tour": global_best_tour,
        "best_cost": global_best_cost,
        "history": global_history,   # one point per start
        "n_evals": total_evals,
    }


# 4. Simulated Annealing (CORRIGÉ)

def simulated_annealing(
    tour: Tour,
    dist: DistMatrix,
    T0: float = None,
    alpha: float = 0.995,
    T_min: float = 1e-3,
    max_evals: int = 200_000,
    rng: Optional[random.Random] = None,
    neighbourhood: str = "swap",
) -> dict:
    """
    Simulated Annealing with geometric cooling.
    CORRECTION: On recalcule toujours le coût complet après chaque mouvement
    pour éviter les erreurs d'accumulation qui causaient des coûts négatifs.
    """
    if rng is None:
        rng = random.Random()

    n = len(tour)
    current = tour[:]
    current_cost = tour_cost(current, dist)
    best = current[:]
    best_cost = current_cost

    # Initialisation de la température
    if T0 is None:
        T = 10 * current_cost
    else:
        T = T0

    n_evals = n
    history = [best_cost]

    while T > T_min and n_evals < max_evals:
        # Générer un voisin aléatoire
        i, j = sorted(rng.sample(range(n), 2))
        
        # Sauvegarder l'état courant pour pouvoir revenir en arrière
        old_tour = current[:]
        
        # Appliquer le mouvement temporairement
        if neighbourhood == "swap":
            current[i], current[j] = current[j], current[i]
        else:  # 2opt
            current[i:j + 1] = reversed(current[i:j + 1])
        
        # Calculer le nouveau coût (recalcul complet pour éviter les erreurs)
        new_cost = tour_cost(current, dist)
        n_evals += 1
        
        delta = new_cost - current_cost
        
        # Décider d'accepter ou non
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            # Accepter : mettre à jour le coût
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best = current[:]
                history.append(best_cost)
        else:
            # Rejeter : restaurer l'ancien tour
            current = old_tour
        
        # Refroidissement
        T *= alpha

    return {
        "best_tour": best,
        "best_cost": best_cost,
        "history": history,
        "n_evals": n_evals,
    }



# 5. BONUS : Tabu Search

def tabu_search(
    tour: Tour,
    dist: DistMatrix,
    tabu_tenure: Optional[int] = None,
    neighbourhood: str = "swap",
    max_evals: int = 100_000,
) -> dict:
    """
    Tabu Search with a fixed-length tabu list on moves (i, j).
    If `tabu_tenure` is None a safe default is used (max(5, n//5)).
    """
    n = len(tour)
    # sensible default computed at call time (cannot use `n` in signature)
    if tabu_tenure is None:
        tabu_tenure = max(5, n // 5)
    tabu_tenure = int(tabu_tenure)
    if tabu_tenure <= 0:
        raise ValueError("tabu_tenure must be a positive integer")

    current = tour[:]
    current_cost = tour_cost(current, dist)
    best = current[:]
    best_cost = current_cost

    tabu_list: dict = {}  # (i, j) -> iteration at which it becomes non-tabu
    n_evals = n
    history = [best_cost]
    iteration = 0

    while n_evals < max_evals:
        iteration += 1
        best_delta = math.inf
        best_move = None

        for i, j in _delta_pairs(n):
            if neighbourhood == "swap":
                delta = delta_swap(current, i, j, dist)
            else:
                delta = delta_two_opt(current, i, j, dist)
            n_evals += 1

            key = (min(i, j), max(i, j))          # normalize move key
            is_tabu = tabu_list.get(key, 0) >= iteration
            aspiration = (current_cost + delta) < best_cost

            if (not is_tabu or aspiration) and delta < best_delta:
                best_delta = delta
                best_move = (i, j)

            if n_evals >= max_evals:
                break

        if best_move is None:
            break

        i, j = best_move
        if neighbourhood == "swap":
            current[i], current[j] = current[j], current[i]
        else:
            current[i:j + 1] = reversed(current[i:j + 1])

        current_cost = tour_cost(current, dist)  # Recalcul complet
        tabu_list[(min(i, j), max(i, j))] = iteration + tabu_tenure

        if current_cost < best_cost:
            best_cost = current_cost
            best = current[:]
            history.append(best_cost)

    return {
        "best_tour": best,
        "best_cost": best_cost,
        "history": history,
        "n_evals": n_evals,
    }

# 6. BONUS : GRASP

def _greedy_random_construction(n: int, dist: DistMatrix, alpha: float, rng: random.Random) -> Tour:
    """
    Greedy randomized construction for TSP.
    alpha=0: pure greedy; alpha=1: pure random.
    Uses nearest-neighbour heuristic with a restricted candidate list.
    """
    unvisited = set(range(n))
    start = rng.randint(0, n - 1)
    tour = [start]
    unvisited.remove(start)

    while unvisited:
        last = tour[-1]
        distances = [(dist[last][c], c) for c in unvisited]
        distances.sort()
        d_min, d_max = distances[0][0], distances[-1][0]
        threshold = d_min + alpha * (d_max - d_min)
        rcl = [c for d, c in distances if d <= threshold]
        chosen = rng.choice(rcl)
        tour.append(chosen)
        unvisited.remove(chosen)

    return tour


def grasp(
    n: int,
    dist: DistMatrix,
    alpha_greedy: float = 0.2,
    n_iterations: int = 30,
    max_evals_hc: int = 10_000,
    rng: Optional[random.Random] = None,
) -> dict:
    """
    GRASP: Greedy Randomized Adaptive Search Procedure.
    Greedy randomized construction + first-improvement HC.
    """
    if rng is None:
        rng = random.Random()

    best_cost = math.inf
    best_tour = None
    total_evals = 0
    history = []

    for _ in range(n_iterations):
        init = _greedy_random_construction(n, dist, alpha_greedy, rng)
        result = hill_climbing_first(init, dist, neighbourhood="2opt", max_evals=max_evals_hc)
        total_evals += result["n_evals"]

        if result["best_cost"] < best_cost:
            best_cost = result["best_cost"]
            best_tour = result["best_tour"]
        history.append(best_cost)

    return {
        "best_tour": best_tour,
        "best_cost": best_cost,
        "history": history,
        "n_evals": total_evals,
    }

