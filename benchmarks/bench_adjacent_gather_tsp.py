"""Benchmark: AdjacentGather vs baseline on the TSP objective.

The traveling salesperson problem expresses its tour cost as

    DISTANCE_MATRIX[ordered_cities[:-1], ordered_cities[1:]].sum()
        + DISTANCE_MATRIX[ordered_cities[-1], ordered_cities[0]]

This is exactly the pattern AdjacentGather collapses into a single node.
This script builds both formulations, verifies they produce identical
objective values on random tours, and reports:

    * model node count   (model.num_nodes())
    * model symbol count (model.num_symbols())
    * model state size   (model.state_size())     -- bytes per state
    * model build time
    * propagation throughput for full-tour resets and adjacent swaps

Phase 1 of AdjacentGather requires a fixed-size 1-D integer sequence,
so TSP (which uses model.list(num_cities)) is the natural drop-in.
CVRPTW uses dynamic disjoint-list sublists and requires Phase 2.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from dwave.optimization import Model
from dwave.optimization.generators import traveling_salesperson
from dwave.optimization.symbols import AdjacentGather


# --------------------------------------------------------------------------- #
# Model builders                                                              #
# --------------------------------------------------------------------------- #
def build_baseline(distance_matrix: np.ndarray) -> tuple[Model, object]:
    """Build the stock traveling_salesperson model and return (model, tour)."""
    model = traveling_salesperson(distance_matrix=distance_matrix)
    tour = next(model.iter_decisions())
    return model, tour


def build_adjacent_gather(distance_matrix: np.ndarray) -> tuple[Model, object]:
    """Build the same TSP objective using AdjacentGather + a single tail term."""
    n = distance_matrix.shape[0]
    model = Model()
    tour = model.list(n)
    M = model.constant(distance_matrix)

    # AdjacentGather replaces M[tour[:-1], tour[1:]] with a single node.
    forward_edges = AdjacentGather(M, tour)            # shape (n - 1,)
    return_edge = M[tour[-1], tour[0]]                 # shape ()

    model.minimize(forward_edges.sum() + return_edge)
    model.lock()
    return model, tour


# --------------------------------------------------------------------------- #
# Measurement utilities                                                       #
# --------------------------------------------------------------------------- #
def model_metrics(model: Model) -> dict:
    return {
        "num_nodes":   model.num_nodes(),
        "num_symbols": model.num_symbols(),
        "state_size":  model.state_size(),
    }


def time_build(builder, distance_matrix, repeats: int = 5) -> float:
    """Median build time in seconds."""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model, _ = builder(distance_matrix)
        samples.append(time.perf_counter() - t0)
        del model
    return float(np.median(samples))


def time_propagation(model: Model, tour, num_moves: int, seed: int = 0) -> float:
    """Mean wall time per (set_state + objective.state(0)) cycle, in microseconds.

    We mutate the tour with a random permutation each iteration and force the
    objective to recompute by reading its state. The first read does the full
    propagation; subsequent reads on the same state are cached, so each random
    perm represents one full re-propagation.
    """
    rng = np.random.default_rng(seed)
    n = tour.shape()[0]

    model.states.resize(1)
    # Warm up: prime the state so the first measured iteration is in the steady regime.
    tour.set_state(0, rng.permutation(n))
    _ = model.objective.state(0)

    perms = [rng.permutation(n) for _ in range(num_moves)]

    t0 = time.perf_counter()
    for perm in perms:
        tour.set_state(0, perm)
        _ = model.objective.state(0)
    elapsed = time.perf_counter() - t0
    return (elapsed / num_moves) * 1e6  # microseconds per propagation


def time_adjacent_swaps(model: Model, tour, num_moves: int, seed: int = 0) -> float:
    """Mean wall time per adjacent-swap update, in microseconds.

    Adjacent swaps model a sparse local-search move: only two tour positions
    change, so an incremental adjacency node should touch a constant number of
    edge costs even when the route itself is long.
    """
    rng = np.random.default_rng(seed)
    n = tour.shape()[0]

    model.states.resize(1)
    perm = rng.permutation(n)
    tour.set_state(0, perm)
    _ = model.objective.state(0)

    swap_positions = rng.integers(0, n - 1, size=num_moves)

    t0 = time.perf_counter()
    for i in swap_positions:
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
        tour.set_state(0, perm)
        _ = model.objective.state(0)
    elapsed = time.perf_counter() - t0
    return (elapsed / num_moves) * 1e6


# --------------------------------------------------------------------------- #
# Correctness check                                                           #
# --------------------------------------------------------------------------- #
def verify_equivalence(distance_matrix, num_checks: int = 25, seed: int = 1) -> None:
    """Sanity-check both models give identical objective values on random tours."""
    rng = np.random.default_rng(seed)
    n = distance_matrix.shape[0]

    base_model, base_tour = build_baseline(distance_matrix)
    new_model,  new_tour  = build_adjacent_gather(distance_matrix)

    base_model.states.resize(1)
    new_model.states.resize(1)

    for _ in range(num_checks):
        perm = rng.permutation(n)
        base_tour.set_state(0, perm)
        new_tour.set_state(0, perm)

        # Reference value computed in NumPy directly:
        ref = (distance_matrix[perm[:-1], perm[1:]].sum()
               + distance_matrix[perm[-1], perm[0]])

        b = float(base_model.objective.state(0))
        a = float(new_model.objective.state(0))

        assert np.isclose(b, ref, rtol=1e-10, atol=1e-10), (b, ref)
        assert np.isclose(a, ref, rtol=1e-10, atol=1e-10), (a, ref)
        assert np.isclose(a, b,   rtol=1e-10, atol=1e-10), (a, b)


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #
def fmt_int(x: int) -> str:
    return f"{x:>9,d}"


def fmt_bytes(x: int) -> str:
    if x < 1024:
        return f"{x:>9d} B"
    if x < 1024 ** 2:
        return f"{x / 1024:>9.1f} KB"
    if x < 1024 ** 3:
        return f"{x / 1024 ** 2:>9.1f} MB"
    return f"{x / 1024 ** 3:>9.1f} GB"


def fmt_time_ms(x_s: float) -> str:
    return f"{x_s * 1e3:>9.2f} ms"


def fmt_us(x_us: float) -> str:
    return f"{x_us:>9.1f} us"


def report_size(n: int, num_moves: int) -> dict:
    rng = np.random.default_rng(42)
    D = rng.random((n, n))
    np.fill_diagonal(D, 0.0)

    print(f"\n=== n = {n} cities ({num_moves} random tours per propagation test) ===")
    verify_equivalence(D)

    base_model, base_tour = build_baseline(D)
    new_model,  new_tour  = build_adjacent_gather(D)

    base = model_metrics(base_model)
    new  = model_metrics(new_model)

    base_build_s = time_build(build_baseline,         D)
    new_build_s  = time_build(build_adjacent_gather,  D)

    base_prop_us = time_propagation(base_model, base_tour, num_moves=num_moves)
    new_prop_us  = time_propagation(new_model,  new_tour,  num_moves=num_moves)
    base_swap_us = time_adjacent_swaps(base_model, base_tour, num_moves=num_moves)
    new_swap_us  = time_adjacent_swaps(new_model,  new_tour,  num_moves=num_moves)

    def pct(old, new):
        if old == 0:
            return "    n/a"
        return f"{100.0 * (new - old) / old:+7.1f}%"

    rows = [
        ("metric",      "baseline",                 "AdjacentGather",          "delta"),
        ("num_nodes",   fmt_int(base["num_nodes"]),   fmt_int(new["num_nodes"]),   pct(base["num_nodes"], new["num_nodes"])),
        ("num_symbols", fmt_int(base["num_symbols"]), fmt_int(new["num_symbols"]), pct(base["num_symbols"], new["num_symbols"])),
        ("state_size",  fmt_bytes(base["state_size"]), fmt_bytes(new["state_size"]), pct(base["state_size"], new["state_size"])),
        ("build_time",  fmt_time_ms(base_build_s),     fmt_time_ms(new_build_s),     pct(base_build_s, new_build_s)),
        ("full_reset",  fmt_us(base_prop_us),          fmt_us(new_prop_us),          pct(base_prop_us, new_prop_us)),
        ("adj_swap",    fmt_us(base_swap_us),          fmt_us(new_swap_us),          pct(base_swap_us, new_swap_us)),
    ]
    width = max(len(r[0]) for r in rows)
    for r in rows:
        print(f"  {r[0]:<{width}}  {r[1]}   {r[2]}   {r[3]}")

    return dict(
        n=n,
        baseline=dict(**base, build_s=base_build_s, prop_us=base_prop_us, swap_us=base_swap_us),
        adjacent=dict(**new,  build_s=new_build_s,  prop_us=new_prop_us, swap_us=new_swap_us),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[10, 25, 50, 100, 250, 500, 1000])
    parser.add_argument("--moves", type=int, default=200,
                        help="random tours per propagation measurement")
    args = parser.parse_args()

    print(f"AdjacentGather vs baseline TSP — sweep over n = {args.sizes}")
    results = [report_size(n, args.moves) for n in args.sizes]

    # Compact summary table.
    print("\n=== summary (AdjacentGather / baseline ratios) ===")
    header = f"  {'n':>5}  {'nodes':>10}  {'state':>10}  {'build':>10}  {'full_reset':>11}  {'adj_swap':>10}"
    print(header)
    for r in results:
        b, a = r["baseline"], r["adjacent"]
        print(f"  {r['n']:>5}  "
              f"{a['num_nodes'] / b['num_nodes']:>10.3f}  "
              f"{a['state_size'] / max(1, b['state_size']):>10.3f}  "
              f"{a['build_s'] / max(1e-12, b['build_s']):>10.3f}  "
              f"{a['prop_us'] / max(1e-12, b['prop_us']):>11.3f}  "
              f"{a['swap_us'] / max(1e-12, b['swap_us']):>10.3f}")


if __name__ == "__main__":
    main()
