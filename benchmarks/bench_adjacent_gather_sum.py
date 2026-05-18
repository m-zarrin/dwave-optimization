"""Benchmark fused AdjacentGatherSum on routing and scheduling-style sequences."""

from __future__ import annotations

import argparse
import time

import numpy as np

from dwave.optimization import Model
from dwave.optimization.symbols import AdjacentGather, AdjacentGatherSum


def build_vector_sum(matrix: np.ndarray, n: int) -> tuple[Model, object]:
    model = Model()
    seq = model.list(n)
    M = model.constant(matrix)
    model.minimize(AdjacentGather(M, seq).sum())
    model.lock()
    return model, seq


def build_fused_sum(matrix: np.ndarray, n: int) -> tuple[Model, object]:
    model = Model()
    seq = model.list(n)
    M = model.constant(matrix)
    model.minimize(AdjacentGatherSum(M, seq))
    model.lock()
    return model, seq


def time_adjacent_swaps(model: Model, seq, num_moves: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = seq.shape()[0]
    perm = rng.permutation(n)
    model.states.resize(1)
    seq.set_state(0, perm)
    _ = model.objective.state(0)
    positions = rng.integers(0, n - 1, size=num_moves)

    t0 = time.perf_counter()
    for i in positions:
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
        seq.set_state(0, perm)
        _ = model.objective.state(0)
    return (time.perf_counter() - t0) * 1e6 / num_moves


def time_build(builder, matrix: np.ndarray, n: int, repeats: int = 5) -> float:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model, _ = builder(matrix, n)
        samples.append(time.perf_counter() - t0)
        del model
    return float(np.median(samples))


def run_case(label: str, matrix: np.ndarray, n: int, moves: int) -> None:
    vector_model, vector_seq = build_vector_sum(matrix, n)
    fused_model, fused_seq = build_fused_sum(matrix, n)

    vector_build = time_build(build_vector_sum, matrix, n)
    fused_build = time_build(build_fused_sum, matrix, n)
    vector_swap = time_adjacent_swaps(vector_model, vector_seq, moves, seed=1)
    fused_swap = time_adjacent_swaps(fused_model, fused_seq, moves, seed=1)

    print(f"\n=== {label}: n = {n}, moves = {moves} ===")
    print(f"  {'metric':<12} {'vector_sum':>12} {'fused_sum':>12} {'ratio':>10}")
    print(f"  {'num_nodes':<12} {vector_model.num_nodes():>12,d} {fused_model.num_nodes():>12,d} "
          f"{fused_model.num_nodes() / vector_model.num_nodes():>10.3f}")
    print(f"  {'state_size':<12} {vector_model.state_size():>12,d} {fused_model.state_size():>12,d} "
          f"{fused_model.state_size() / vector_model.state_size():>10.3f}")
    print(f"  {'build_us':<12} {vector_build * 1e6:>12.1f} {fused_build * 1e6:>12.1f} "
          f"{fused_build / vector_build:>10.3f}")
    print(f"  {'adj_swap_us':<12} {vector_swap:>12.2f} {fused_swap:>12.2f} "
          f"{fused_swap / vector_swap:>10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[25, 100, 500])
    parser.add_argument("--moves", type=int, default=1000)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    for n in args.sizes:
        routing = rng.random((n, n))
        np.fill_diagonal(routing, 0.0)
        run_case("routing", routing, n, args.moves)

        setup = rng.integers(0, 100, size=(n, n)).astype(float)
        np.fill_diagonal(setup, 0.0)
        run_case("scheduling_setup", setup, n, args.moves)


if __name__ == "__main__":
    main()
