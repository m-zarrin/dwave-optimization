"""Benchmark: AdjacentGather (Phase 2, dynamic) vs baseline on CVRPTW.

The capacitated_vehicle_routing_with_time_windows generator computes its
inter-customer travel cost for each vehicle as

    t_cust[routes[r][1:], routes[r][:-1]].sum()

where ``routes[r]`` is one sublist of a DisjointLists decision. That sublist
is *dynamic*: whenever a customer is reassigned across vehicles, ``routes[r]``
grows or shrinks. This is exactly the pattern Phase 2 of AdjacentGather
collapses into a single node per vehicle.

This script:

    * builds the stock CVRPTW model (baseline),
    * builds a structurally-identical model where each per-vehicle inter-
      customer cost is replaced by one AdjacentGather node over the dynamic
      sublist,
    * verifies bit-identical objective values across random reassignments
      (pure-grow / pure-shrink / mutate-and-grow / mutate-and-shrink, all of
      the Phase-2 transitions tested in test_dynamic_sequence_numpy_equivalence),
    * reports node count, symbol count, state size, build time, and
      propagation throughput.

Phase 1 of AdjacentGather rejected dynamic sequences and so could not be
applied here at all; CVRPTW is therefore the headline beneficiary of Phase 2.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from dwave.optimization import Model
from dwave.optimization.generators import capacitated_vehicle_routing_with_time_windows
from dwave.optimization.mathematical import add, maximum, where
from dwave.optimization.symbols import AdjacentGather, AdjacentGatherSum


# --------------------------------------------------------------------------- #
# Random instance generation                                                  #
# --------------------------------------------------------------------------- #
def make_instance(num_customers: int, num_vehicles: int, *, seed: int = 0) -> dict:
    """Generate a random feasible CVRPTW instance.

    The horizon is wide and capacities generous so the random feasible
    assignments produced below are valid for the constraints we care about.
    Demand and time-distance scales follow the docstring example.
    """
    rng = np.random.default_rng(seed)

    # Layout customers + depot uniformly in a square.
    coords = rng.uniform(0.0, 100.0, size=(num_customers + 1, 2))
    diff = coords[:, None, :] - coords[None, :, :]
    time_distances = np.sqrt((diff ** 2).sum(axis=-1))

    demand = np.zeros(num_customers + 1)
    demand[1:] = rng.integers(1, 10, size=num_customers)

    capacity = float(demand.sum() / max(1, num_vehicles - 0)) * 3.0 + 50.0
    horizon = 1.0e6

    time_window_open = np.zeros(num_customers + 1)
    time_window_close = np.full(num_customers + 1, horizon)
    service_time = np.zeros(num_customers + 1)
    service_time[1:] = rng.uniform(1.0, 5.0, size=num_customers)

    return dict(
        demand=demand,
        number_of_vehicles=num_vehicles,
        vehicle_capacity=capacity,
        time_distances=time_distances,
        time_window_open=time_window_open,
        time_window_close=time_window_close,
        service_time=service_time,
    )


def random_assignment(num_customers: int, num_vehicles: int, rng) -> list[list[float]]:
    """Random partition of [0, num_customers) into num_vehicles ordered lists."""
    perm = rng.permutation(num_customers).tolist()
    cuts = sorted(rng.integers(0, num_customers + 1, size=num_vehicles - 1).tolist())
    routes: list[list[float]] = []
    prev = 0
    for c in cuts:
        routes.append([float(x) for x in perm[prev:c]])
        prev = c
    routes.append([float(x) for x in perm[prev:]])
    return routes


# --------------------------------------------------------------------------- #
# Model builders                                                              #
# --------------------------------------------------------------------------- #
def build_baseline(instance: dict) -> tuple[Model, object]:
    model = capacitated_vehicle_routing_with_time_windows(**instance)
    routes = next(model.iter_decisions())
    return model, routes


def _build_adjacent_variant(instance: dict, *, fused_sum: bool) -> tuple[Model, object]:
    """Same constraints/objective as the baseline, but with the per-vehicle
    inter-customer cost expressed as AdjacentGather over the dynamic sublist.
    """
    demand = np.asarray(instance["demand"], dtype=float)
    num_vehicles = instance["number_of_vehicles"]
    vehicle_capacity = instance["vehicle_capacity"]
    time_distances_array = np.asarray(instance["time_distances"], dtype=float)
    time_window_open = np.asarray(instance["time_window_open"], dtype=float)
    time_window_close = np.asarray(instance["time_window_close"], dtype=float)
    service_time = np.asarray(instance["service_time"], dtype=float)

    customer_demand = demand[1:]
    num_customers = len(customer_demand)

    model = Model()

    # Same extended-array layout as the stock generator.
    time_distance_ext = np.zeros((num_customers + 1, num_customers + 1))
    time_distance_ext[:num_customers, :num_customers] = time_distances_array[1:, 1:]

    time_from_depo_ext = np.zeros(num_customers + 1)
    time_from_depo_ext[:num_customers] = time_distances_array[0, 1:]

    time_to_depo_ext = np.zeros(num_customers + 1)
    time_to_depo_ext[:num_customers] = time_distances_array[1:, 0]

    if np.equal(time_from_depo_ext, time_to_depo_ext).all():
        t_from_depo = t_to_depo = model.constant(time_from_depo_ext)
    else:
        t_from_depo = model.constant(time_from_depo_ext)
        t_to_depo = model.constant(time_to_depo_ext)

    t_cust = model.constant(time_distance_ext)

    time_open_ext = np.zeros(num_customers + 1)
    time_open_ext[:num_customers] = time_window_open[1:]
    t_open = model.constant(time_open_ext)

    time_close_ext = np.zeros(num_customers + 1)
    time_close_ext[:num_customers] = time_window_close[1:]
    time_close_ext[-1] = 1e6
    t_close = model.constant(time_close_ext)

    service_time_ext = np.zeros(num_customers + 1)
    service_time_ext[:num_customers] = service_time[1:]
    t_service = model.constant(service_time_ext)

    demand_c = model.constant(customer_demand)
    capacity = model.constant(vehicle_capacity)
    t_depo_close = model.constant(float(time_window_close[0]))

    one = model.constant(1)

    routes = model.disjoint_lists_symbol(
        primary_set_size=num_customers,
        num_disjoint_lists=num_vehicles,
    )

    # Capacity constraint (identical to baseline).
    for v in range(num_vehicles):
        model.add_constraint(demand_c[routes[v]].sum() <= capacity)

    rh = np.arange(num_customers + 1).astype(int)
    range_helper = model.constant(rh)

    num_clients_in_route = {f"route{v}": routes[v].size() for v in range(num_vehicles)}

    # Same per-route length cap as baseline.
    max_loc_per_route = min(num_customers, max(10, 3 * int(num_customers / num_vehicles)))
    max_loc_per_route_constant = model.constant(max_loc_per_route)
    for v in range(num_vehicles):
        model.add_constraint(num_clients_in_route[f"route{v}"] <= max_loc_per_route_constant)

    # Time-window machinery -- *unchanged* from baseline. The only difference
    # is the inter-customer travel cost below.
    times_back = []
    for vehicle_idx in range(num_vehicles):
        this_t_leaving = []
        this_t_windows_c = []

        for client_idx in range(max_loc_per_route):
            cond = (num_clients_in_route[f"route{vehicle_idx}"] >= range_helper[client_idx] + one)

            if client_idx == 0:
                idx = where(cond, routes[vehicle_idx][:1].sum(), range_helper[-1])
                this_t_serving = maximum(t_from_depo[idx], t_open[idx])
                this_t_leaving.append(this_t_serving + t_service[idx])
            else:
                prev_idx = where(cond, routes[vehicle_idx][client_idx - 1:client_idx].sum(),
                                 range_helper[-1])
                idx = where(cond, routes[vehicle_idx][client_idx:client_idx + 1].sum(),
                            range_helper[-1])
                this_t_serving = maximum(this_t_leaving[-1] + t_cust[prev_idx, idx], t_open[idx])
                this_t_leaving.append(this_t_serving + t_service[idx])

            this_t_windows_c.append(this_t_leaving[-1] <= t_close[idx])

        cond_last = (num_clients_in_route[f"route{vehicle_idx}"] >= one)
        last_cust_idx = where(cond_last, routes[vehicle_idx][-1:].sum(), range_helper[-1])
        times_back.append(this_t_leaving[-1] + t_to_depo[last_cust_idx])

        for ct in this_t_windows_c:
            model.add_constraint(ct)

    for tb in times_back:
        model.add_constraint(tb <= t_depo_close)

    # ---- The Phase-2 win: replace the dynamic adjacent-gather per vehicle ----
    #
    # Baseline does, per vehicle r:
    #     t_cust[routes[r][1:], routes[r][:-1]].sum()
    # which is the (transposed) adjacent-gather pattern. We pick the matching
    # orientation by feeding (i, i+1) pairs from the row-major t_cust matrix.
    #
    # Note that the baseline formulation actually swaps row/column (it uses
    # [1:], [:-1] rather than [:-1], [1:]), so to stay numerically identical
    # we transpose the cost matrix once before constructing AdjacentGather.
    t_cust_T = model.constant(time_distance_ext.T)
    route_costs = []
    for r in range(num_vehicles):
        route_costs.append(t_from_depo[routes[r][:1]].sum())
        route_costs.append(t_to_depo[routes[r][-1:]].sum())
        if fused_sum:
            route_costs.append(AdjacentGatherSum(t_cust_T, routes[r]))
        else:
            route_costs.append(AdjacentGather(t_cust_T, routes[r]).sum())

    model.minimize(add(*route_costs))
    model.lock()
    return model, routes


def build_adjacent_gather(instance: dict) -> tuple[Model, object]:
    return _build_adjacent_variant(instance, fused_sum=False)


def build_adjacent_gather_sum(instance: dict) -> tuple[Model, object]:
    return _build_adjacent_variant(instance, fused_sum=True)


# --------------------------------------------------------------------------- #
# Correctness                                                                 #
# --------------------------------------------------------------------------- #
def verify_equivalence(instance: dict, *, num_checks: int = 40, seed: int = 1) -> None:
    """Sanity-check both models give identical objective values + per-constraint
    state across many random reassignments of customers to vehicles.
    """
    num_customers = len(instance["demand"]) - 1
    num_vehicles = instance["number_of_vehicles"]
    rng = np.random.default_rng(seed)

    base_model, base_routes = build_baseline(instance)
    new_model, new_routes = build_adjacent_gather(instance)
    fused_model, fused_routes = build_adjacent_gather_sum(instance)

    base_model.states.resize(1)
    new_model.states.resize(1)
    fused_model.states.resize(1)

    # DisjointLists requires every assignment be a *complete* partition of
    # [0, num_customers). We therefore exercise per-sublist grow/shrink/mutate
    # by moving customers between vehicles, which strictly grows one sublist
    # and shrinks another -- the canonical Phase-2 dynamic transition.
    assignments: list[list[list[float]]] = []

    # Phase A: pile everything onto vehicle 0 (other sublists are empty -- the
    # pure-empty branch of AdjacentGather).
    state = [list(range(num_customers))] + [[] for _ in range(num_vehicles - 1)]
    assignments.append([[float(x) for x in r] for r in state])

    # Phase B: peel customers off vehicle 0 one at a time, distributing round
    # robin across the remaining vehicles. This grows each non-zero sublist
    # incrementally while shrinking sublist 0.
    if num_vehicles > 1:
        for i in range(num_customers):
            cust = state[0].pop(0)
            state[1 + (i % (num_vehicles - 1))].append(cust)
            assignments.append([[float(x) for x in r] for r in state])

    # Phase C: random complete reassignments (mutate-and-grow / mutate-and-shrink).
    for _ in range(num_checks):
        assignments.append(random_assignment(num_customers, num_vehicles, rng))

    for assignment in assignments:
        base_routes.set_state(0, assignment)
        new_routes.set_state(0, assignment)
        fused_routes.set_state(0, assignment)

        b_obj = float(base_model.objective.state(0))
        a_obj = float(new_model.objective.state(0))
        f_obj = float(fused_model.objective.state(0))
        assert np.isclose(b_obj, a_obj, rtol=1e-9, atol=1e-9), (b_obj, a_obj, assignment)
        assert np.isclose(b_obj, f_obj, rtol=1e-9, atol=1e-9), (b_obj, f_obj, assignment)

        # Per-constraint feasibility must agree too (same constraint count + order).
        b_cons = [bool(c.state(0)) for c in base_model.iter_constraints()]
        a_cons = [bool(c.state(0)) for c in new_model.iter_constraints()]
        f_cons = [bool(c.state(0)) for c in fused_model.iter_constraints()]
        assert b_cons == a_cons == f_cons, (b_cons, a_cons, f_cons, assignment)


# --------------------------------------------------------------------------- #
# Measurement utilities                                                       #
# --------------------------------------------------------------------------- #
def model_metrics(model: Model) -> dict:
    return dict(
        num_nodes=model.num_nodes(),
        num_symbols=model.num_symbols(),
        state_size=model.state_size(),
    )


def time_build(builder, instance, repeats: int = 5) -> float:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model, _ = builder(instance)
        samples.append(time.perf_counter() - t0)
        del model
    return float(np.median(samples))


def time_propagation(model: Model, routes, num_customers: int, num_vehicles: int,
                     num_moves: int, *, seed: int = 0) -> float:
    """Mean wall time per (set_state + objective.state(0)) cycle, in microseconds."""
    rng = np.random.default_rng(seed)
    model.states.resize(1)
    routes.set_state(0, random_assignment(num_customers, num_vehicles, rng))
    _ = model.objective.state(0)

    moves = [random_assignment(num_customers, num_vehicles, rng) for _ in range(num_moves)]

    t0 = time.perf_counter()
    for assignment in moves:
        routes.set_state(0, assignment)
        _ = model.objective.state(0)
    elapsed = time.perf_counter() - t0
    return (elapsed / num_moves) * 1e6


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


def report_size(num_customers: int, num_vehicles: int, num_moves: int) -> dict:
    instance = make_instance(num_customers, num_vehicles, seed=42)
    print(f"\n=== n_cust = {num_customers}, n_veh = {num_vehicles} "
          f"({num_moves} random reassignments per propagation test) ===")
    verify_equivalence(instance)

    base_model, base_routes = build_baseline(instance)
    new_model, new_routes = build_adjacent_gather(instance)
    fused_model, fused_routes = build_adjacent_gather_sum(instance)

    base = model_metrics(base_model)
    new = model_metrics(new_model)
    fused = model_metrics(fused_model)

    base_build_s = time_build(build_baseline, instance)
    new_build_s = time_build(build_adjacent_gather, instance)
    fused_build_s = time_build(build_adjacent_gather_sum, instance)

    base_prop_us = time_propagation(base_model, base_routes,
                                    num_customers, num_vehicles, num_moves=num_moves)
    new_prop_us = time_propagation(new_model, new_routes,
                                   num_customers, num_vehicles, num_moves=num_moves)
    fused_prop_us = time_propagation(fused_model, fused_routes,
                                     num_customers, num_vehicles, num_moves=num_moves)

    def pct(old, new):
        if old == 0:
            return "    n/a"
        return f"{100.0 * (new - old) / old:+7.1f}%"

    rows = [
        ("metric", "baseline", "AdjacentGather", "AGSum", "AGSum delta"),
        ("num_nodes", fmt_int(base["num_nodes"]), fmt_int(new["num_nodes"]),
         fmt_int(fused["num_nodes"]), pct(base["num_nodes"], fused["num_nodes"])),
        ("num_symbols", fmt_int(base["num_symbols"]), fmt_int(new["num_symbols"]),
         fmt_int(fused["num_symbols"]), pct(base["num_symbols"], fused["num_symbols"])),
        ("state_size", fmt_bytes(base["state_size"]), fmt_bytes(new["state_size"]),
         fmt_bytes(fused["state_size"]), pct(base["state_size"], fused["state_size"])),
        ("build_time", fmt_time_ms(base_build_s), fmt_time_ms(new_build_s),
         fmt_time_ms(fused_build_s), pct(base_build_s, fused_build_s)),
        ("propagate", fmt_us(base_prop_us), fmt_us(new_prop_us),
         fmt_us(fused_prop_us), pct(base_prop_us, fused_prop_us)),
    ]
    width = max(len(r[0]) for r in rows)
    for r in rows:
        print(f"  {r[0]:<{width}}  {r[1]}   {r[2]}   {r[3]}   {r[4]}")

    return dict(
        n_cust=num_customers, n_veh=num_vehicles,
        baseline=dict(**base, build_s=base_build_s, prop_us=base_prop_us),
        adjacent=dict(**new, build_s=new_build_s, prop_us=new_prop_us),
        fused=dict(**fused, build_s=fused_build_s, prop_us=fused_prop_us),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[10, 25, 50, 100, 200],
        help="customer counts to sweep over",
    )
    parser.add_argument(
        "--vehicles", type=int, default=4,
        help="number of vehicles per instance",
    )
    parser.add_argument(
        "--moves", type=int, default=100,
        help="random reassignments per propagation measurement",
    )
    args = parser.parse_args()

    print(f"AdjacentGather (Phase 2, dynamic) vs baseline CVRPTW "
          f"-- sweep over n_cust = {args.sizes}, n_veh = {args.vehicles}")
    results = [report_size(n, args.vehicles, args.moves) for n in args.sizes]

    print("\n=== summary (AGSum / baseline ratios) ===")
    header = (f"  {'n_cust':>7}  {'nodes':>10}  {'state':>10}  {'build':>10}  "
              f"{'propagate':>11}")
    print(header)
    for r in results:
        b, a = r["baseline"], r["fused"]
        print(f"  {r['n_cust']:>7}  "
              f"{a['num_nodes'] / b['num_nodes']:>10.3f}  "
              f"{a['state_size'] / max(1, b['state_size']):>10.3f}  "
              f"{a['build_s'] / max(1e-12, b['build_s']):>10.3f}  "
              f"{a['prop_us'] / max(1e-12, b['prop_us']):>11.3f}")


if __name__ == "__main__":
    main()
