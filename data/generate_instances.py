import json
import random
import math
import os

SEED = 42

def generate_instance(n: int, seed: int, name: str) -> dict:
    rng = random.Random(seed)
    cities = [(round(rng.uniform(0, 100), 2), round(rng.uniform(0, 100), 2)) for _ in range(n)]
    return {"name": name, "n": n, "cities": cities}

def euclidean(c1, c2) -> float:
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

if __name__ == "__main__":
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    configs = [
        (20, SEED,      "instance_A"),
        (50, SEED + 1,  "instance_B"),
        (80, SEED + 2,  "instance_C"),
    ]
    for n, seed, name in configs:
        inst = generate_instance(n, seed, name)
        path = os.path.join(os.path.dirname(__file__), f"{name}.json")
        with open(path, "w") as f:
            json.dump(inst, f, indent=2)
        print(f"Generated {name} with {n} cities → {path}")
