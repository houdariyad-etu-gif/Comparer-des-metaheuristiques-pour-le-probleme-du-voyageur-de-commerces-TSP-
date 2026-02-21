
import math
import random
import json
from typing import List, Tuple

Tour = List[int]
DistMatrix = List[List[float]]


# Instance loading

def load_instance(path: str) -> Tuple[int, DistMatrix]:
    """Load a JSON instance and return (n, dist_matrix)."""
    with open(path) as f:
        data = json.load(f)
    cities = data["cities"]
    n = len(cities)
    dist = build_distance_matrix(cities)
    return n, dist


# Distance matrix

def build_distance_matrix(cities: List[Tuple[float, float]]) -> DistMatrix:
    """Euclidean distance matrix (rounded to nearest integer as TSPLIB convention)."""
    n = len(cities)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt((cities[i][0] - cities[j][0])**2 + (cities[i][1] - cities[j][1])**2)
            d = round(d)
            dist[i][j] = d
            dist[j][i] = d
    return dist


# Objective function

def tour_cost(tour: Tour, dist: DistMatrix) -> float:
    """Total length of a closed tour."""
    n = len(tour)
    return sum(dist[tour[k]][tour[(k + 1) % n]] for k in range(n))


# Random tour

def random_tour(n: int, rng: random.Random) -> Tour:
    tour = list(range(n))
    rng.shuffle(tour)
    return tour


# Neighbourhoods – generators (lazy, for first-improvement)

def swap_neighbours_gen(tour: Tour):
    """Yield all neighbours obtained by swapping two positions."""
    n = len(tour)
    for i in range(n - 1):
        for j in range(i + 1, n):
            nb = tour[:]
            nb[i], nb[j] = nb[j], nb[i]
            yield nb


def two_opt_neighbours_gen(tour: Tour):
    """Yield all neighbours obtained by a 2-opt reversal."""
    n = len(tour)
    for i in range(n - 1):
        for j in range(i + 2, n):
            nb = tour[:]
            nb[i:j + 1] = reversed(nb[i:j + 1])
            yield nb


# Delta evaluation (fast incremental cost change)

def delta_swap(tour: Tour, i: int, j: int, dist: DistMatrix) -> float:
    """Cost change when swapping positions i and j (i < j)."""
    n = len(tour)
    # edges affected: (prev_i, i), (i, next_i), (prev_j, j), (j, next_j)
    # Be careful when i and j are adjacent
    pi, ni = (i - 1) % n, (i + 1) % n
    pj, nj = (j - 1) % n, (j + 1) % n

    a, b = tour[i], tour[j]
    before = (dist[tour[pi]][a] + dist[a][tour[ni]] +
              dist[tour[pj]][b] + dist[b][tour[nj]])
    # after swap
    after  = (dist[tour[pi]][b] + dist[b][tour[ni]] +
              dist[tour[pj]][a] + dist[a][tour[nj]])
    # adjacent case: i+1==j
    if ni == j:
        before = dist[tour[pi]][a] + dist[a][b] + dist[b][tour[nj]]
        after  = dist[tour[pi]][b] + dist[b][a] + dist[a][tour[nj]]
    return after - before


def delta_two_opt(tour: Tour, i: int, j: int, dist: DistMatrix) -> float:
    """Cost change for 2-opt reversal of segment [i, j]."""
    n = len(tour)
    a, b = tour[i - 1], tour[i]
    c, d = tour[j], tour[(j + 1) % n]
    return dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
