"""
Verification of the LP bounds reported in

    S.L. Gadegaard, "On the strength of the peak formulation for the capacitated
    vehicle routing problem" (Gadegaard_main_rev.tex)

for the 10-customer instance of Table 1, using Gurobi through gurobipy.

The script builds and solves, for that instance:

  (1) LP(peak)  -- the LP relaxation of the basic peak formulation, eqs. (6)-(15);
  (2) LP(GG)    -- the LP relaxation of the one-commodity flow formulation of
                   Gavish and Graves with the strengthened flow bounds, eq. (17);
  (3) LP(GLM)   -- the LP relaxation of the two-index formulation, eqs. (1)-(4),
                   with all 2^10 - 1 GLM inequalities (5) written out explicitly.

and checks that

  * z_GG = z_GLM                      (projection result of Gouveia 1995),
  * z_peak > z_GLM                    (Proposition 7),
  * the numbers quoted in the paper are reproduced,
  * removing the sequencing constraints (14)-(15) does not change z_peak,
  * relaxing the fixing p_n = 1 to 0 <= p_n <= 1 reduces z_peak to z_GLM,
  * no relabelling of the customers gives z_peak < z_GLM, and z_peak depends on
    the labelling only through which customer receives the label n.

Usage
-----
    python verify_peak_vs_glm.py                # core checks + all 10 labellings of n
    python verify_peak_vs_glm.py --random 200   # + 200 random relabellings
    python verify_peak_vs_glm.py --full         # + all 10! = 3628800 relabellings (slow)

Requires gurobipy with a working Gurobi licence.
"""

from __future__ import annotations

import argparse
import itertools
import math
from collections import Counter, defaultdict

import gurobipy as gp
from gurobipy import GRB

# --------------------------------------------------------------------------- #
# Instance of Table 1
# --------------------------------------------------------------------------- #

COORDS = {
    0: (50, 50),   # depot
    1: (50, 45),
    2: (45, 72),
    3: (65, 79),
    4: (84, 43),
    5: (76, 26),
    6: (13, 85),
    7: (23, 95),
    8: (76, 0),
    9: (90, 3),
    10: (9, 3),
}
DEMAND = {0: 0, 1: 1, 2: 9, 3: 1, 4: 7, 5: 4, 6: 7, 7: 1, 8: 9, 9: 4, 10: 8}
CAPACITY = 17.0
N = 10

# Values reported in the paper.
Z_PEAK_PAPER = 430.7654
Z_GLM_PAPER = 429.1332

TOL = 1e-6


def instance(perm=None):
    """Return (V, Vc, n, q, c, Q) for the instance, optionally relabelled.

    perm[k - 1] is the original customer that receives the new label k, so
    perm = (1, ..., n) is the labelling of Table 1.
    """
    n = N
    if perm is None:
        perm = tuple(range(1, n + 1))
    lab = {0: 0}
    lab.update({k: perm[k - 1] for k in range(1, n + 1)})
    V = list(range(n + 1))
    Vc = list(range(1, n + 1))
    q = {i: float(DEMAND[lab[i]]) for i in V}
    c = {}
    for i in V:
        for j in V:
            if i != j:
                xi, yi = COORDS[lab[i]]
                xj, yj = COORDS[lab[j]]
                c[i, j] = math.hypot(xi - xj, yi - yj)
    return V, Vc, n, q, c, CAPACITY


def _env(verbose=False):
    env = gp.Env(empty=True)
    if not verbose:
        env.setParam("OutputFlag", 0)
    env.start()
    return env


# --------------------------------------------------------------------------- #
# (1) LP relaxation of the basic peak formulation, eqs. (6)-(15)
# --------------------------------------------------------------------------- #

def solve_peak(perm=None, sequencing=True, fix_pn=True, env=None, verbose=False):
    """LP relaxation of the basic peak formulation.

    sequencing : include the sequencing constraints (14)-(15).
    fix_pn     : fix p_n = 1; if False, p_n is only bounded by 0 <= p_n <= 1.
    """
    V, Vc, n, q, c, Q = instance(perm)
    own_env = env is None
    env = env or _env(verbose)
    try:
        m = gp.Model("peak", env=env)

        Ac = [(i, j) for i in Vc for j in Vc if i != j]      # customer arcs
        A0 = [(0, j) for j in Vc]                            # depot arcs

        # Variables. Integrality is relaxed everywhere (this is the LP relaxation).
        x = m.addVars(Ac, lb=0.0, ub=1.0, obj=[c[a] for a in Ac], name="x")
        x0 = m.addVars(Vc, lb=0.0, ub=2.0, obj=[c[0, j] for j in Vc], name="x0")
        p = m.addVars(Vc, lb=0.0, ub=1.0, name="p")
        K = m.addVar(lb=0.0, name="K")
        f = m.addVars(Ac, lb=0.0, name="f")     # no flow on depot arcs: f_0j = 0
        t = m.addVars(Vc, lb=0.0, name="t")
        m.ModelSense = GRB.MINIMIZE

        if fix_pn:
            p[n].LB = 1.0                       # p_n = 1

        # (7) out-degree: x(delta^+({i})) + p_i = 1
        m.addConstrs(
            (gp.quicksum(x[i, j] for j in Vc if j != i) + p[i] == 1 for i in Vc),
            name="out",
        )
        # (8) in-degree: x(delta^-({i})) - p_i = 1   (delta^-({i}) contains arc (0,i))
        m.addConstrs(
            (x0[i] + gp.quicksum(x[j, i] for j in Vc if j != i) - p[i] == 1 for i in Vc),
            name="in",
        )
        # (9) x(delta^+({0})) = 2K
        m.addConstr(gp.quicksum(x0[j] for j in Vc) == 2 * K, name="depot")
        # (10) sum_i p_i = K
        m.addConstr(gp.quicksum(p[i] for i in Vc) == K, name="numpeaks")
        # (11) flow conservation: t_i + f(delta^+({i})) = f(delta^-({i})) + q_i
        m.addConstrs(
            (
                t[i]
                + gp.quicksum(f[i, j] for j in Vc if j != i)
                == gp.quicksum(f[j, i] for j in Vc if j != i) + q[i]
                for i in Vc
            ),
            name="flow",
        )
        # (12) flow bounds on customer arcs
        m.addConstrs((f[i, j] >= q[i] * x[i, j] for (i, j) in Ac), name="flb")
        m.addConstrs((f[i, j] <= (Q - q[j]) * x[i, j] for (i, j) in Ac), name="fub")
        # (13) load bounds
        m.addConstrs((t[i] >= q[i] * p[i] for i in Vc), name="tlb")
        m.addConstrs((t[i] <= Q * p[i] for i in Vc), name="tub")

        # (14)-(15) sequencing constraints
        if sequencing:
            u = m.addVars(range(1, n), lb=[i for i in range(1, n)], name="u")
            m.addConstrs(
                (u[i] <= i * p[i] + (n - 1) * (1 - p[i]) for i in range(1, n)),
                name="ubnd",
            )
            m.addConstrs(
                (
                    u[i] - u[j] + (n - j - 1) * x[i, j] <= n - j - 1
                    for i in range(1, n)
                    for j in range(1, n)
                    if i != j
                ),
                name="mtz",
            )

        m.optimize()
        assert m.Status == GRB.OPTIMAL, f"peak LP status {m.Status}"
        return m.ObjVal
    finally:
        if own_env:
            env.dispose()


# --------------------------------------------------------------------------- #
# (2) LP relaxation of the strengthened Gavish-Graves formulation, eq. (17)
# --------------------------------------------------------------------------- #

def solve_gg(perm=None, env=None, verbose=False):
    V, Vc, n, q, c, Q = instance(perm)
    own_env = env is None
    env = env or _env(verbose)
    try:
        m = gp.Model("gg", env=env)
        A = [(i, j) for i in V for j in V if i != j]

        y = m.addVars(A, lb=0.0, ub=1.0, obj=[c[a] for a in A], name="y")
        g = m.addVars(A, lb=0.0, name="g")
        K = m.addVar(lb=0.0, name="K")
        m.ModelSense = GRB.MINIMIZE

        # (2)-(3) degree constraints
        m.addConstrs(
            (gp.quicksum(y[i, j] for j in V if j != i) == 1 for i in Vc), name="out"
        )
        m.addConstrs(
            (gp.quicksum(y[j, i] for j in V if j != i) == 1 for i in Vc), name="in"
        )
        # (4) depot degree
        m.addConstr(gp.quicksum(y[0, j] for j in Vc) == K, name="depot_out")
        m.addConstr(gp.quicksum(y[j, 0] for j in Vc) == K, name="depot_in")
        # flow conservation: g(delta^+({i})) - g(delta^-({i})) = q_i
        m.addConstrs(
            (
                gp.quicksum(g[i, j] for j in V if j != i)
                - gp.quicksum(g[j, i] for j in V if j != i)
                == q[i]
                for i in Vc
            ),
            name="flow",
        )
        # (17) strengthened flow bounds, with q_0 = 0
        m.addConstrs((g[i, j] >= q[i] * y[i, j] for (i, j) in A), name="glb")
        m.addConstrs((g[i, j] <= (Q - q[j]) * y[i, j] for (i, j) in A), name="gub")

        m.optimize()
        assert m.Status == GRB.OPTIMAL, f"GG LP status {m.Status}"
        return m.ObjVal
    finally:
        if own_env:
            env.dispose()


# --------------------------------------------------------------------------- #
# (3) LP relaxation of the two-index formulation with all GLM inequalities
# --------------------------------------------------------------------------- #

def solve_glm(perm=None, env=None, verbose=False):
    """All 2^n - 1 GLM inequalities written out explicitly (n = 10 is small)."""
    V, Vc, n, q, c, Q = instance(perm)
    own_env = env is None
    env = env or _env(verbose)
    try:
        m = gp.Model("glm", env=env)
        A = [(i, j) for i in V for j in V if i != j]

        y = m.addVars(A, lb=0.0, ub=1.0, obj=[c[a] for a in A], name="y")
        K = m.addVar(lb=0.0, name="K")
        m.ModelSense = GRB.MINIMIZE

        m.addConstrs(
            (gp.quicksum(y[i, j] for j in V if j != i) == 1 for i in Vc), name="out"
        )
        m.addConstrs(
            (gp.quicksum(y[j, i] for j in V if j != i) == 1 for i in Vc), name="in"
        )
        m.addConstr(gp.quicksum(y[0, j] for j in Vc) == K, name="depot_out")
        m.addConstr(gp.quicksum(y[j, 0] for j in Vc) == K, name="depot_in")

        # (5) y(delta^+(S)) >= q(S)/Q + (1/Q) sum_{i in S} sum_{j in Vc \ S} q_j (y_ij + y_ji)
        for r in range(1, n + 1):
            for S in itertools.combinations(Vc, r):
                Sset = set(S)
                out = gp.quicksum(y[i, j] for i in Sset for j in V if j not in Sset)
                multistar = gp.quicksum(
                    q[j] * (y[i, j] + y[j, i])
                    for i in Sset
                    for j in Vc
                    if j not in Sset
                )
                m.addConstr(
                    Q * out >= sum(q[i] for i in Sset) + multistar,
                    name=f"glm_{'_'.join(map(str, S))}",
                )

        m.optimize()
        assert m.Status == GRB.OPTIMAL, f"GLM LP status {m.Status}"
        return m.ObjVal
    finally:
        if own_env:
            env.dispose()


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def check(label, condition):
    print(f"  [{'OK ' if condition else 'FAIL'}] {label}")
    return condition


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--random", type=int, default=0,
                    help="additionally test this many random relabellings")
    ap.add_argument("--full", action="store_true",
                    help="additionally test all 10! relabellings (slow)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="show Gurobi output")
    args = ap.parse_args()

    print(f"gurobipy version{gp.gurobi.version()}")
    env = _env(args.verbose)
    ok = True

    print("\nBounds for the instance of Table 1")
    z_peak = solve_peak(env=env)
    z_gg = solve_gg(env=env)
    z_glm = solve_glm(env=env)
    print(f"  z_peak = {z_peak:.6f}")
    print(f"  z_GG   = {z_gg:.6f}")
    print(f"  z_GLM  = {z_glm:.6f}   (all {2 ** N - 1} GLM inequalities)")

    print("\nChecks")
    ok &= check("z_GG = z_GLM (Gouveia 1995 projection result)",
                abs(z_gg - z_glm) < 1e-6 * max(1.0, abs(z_glm)))
    ok &= check("z_peak > z_GLM (Proposition 7)", z_peak > z_glm + TOL)
    ok &= check(f"z_peak matches the paper's {Z_PEAK_PAPER}",
                abs(z_peak - Z_PEAK_PAPER) < 5e-5)
    ok &= check(f"z_GLM matches the paper's {Z_GLM_PAPER}",
                abs(z_glm - Z_GLM_PAPER) < 5e-5)

    z_noseq = solve_peak(sequencing=False, env=env)
    ok &= check("dropping the sequencing constraints (14)-(15) leaves z_peak unchanged",
                abs(z_noseq - z_peak) < 1e-6 * max(1.0, abs(z_peak)))

    z_free = solve_peak(fix_pn=False, env=env)
    ok &= check("relaxing p_n = 1 to 0 <= p_n <= 1 gives z_peak = z_GLM",
                abs(z_free - z_glm) < 1e-6 * max(1.0, abs(z_glm)))
    z_free_noseq = solve_peak(sequencing=False, fix_pn=False, env=env)
    ok &= check("the same holds without the sequencing constraints",
                abs(z_free_noseq - z_glm) < 1e-6 * max(1.0, abs(z_glm)))
    print(f"      z_peak without (14)-(15)            = {z_noseq:.6f}")
    print(f"      z_peak with 0 <= p_n <= 1           = {z_free:.6f}")

    # Relabellings: one per choice of the customer that receives the label n.
    print("\nOne relabelling per customer carrying the label n")
    values = {}
    for k in range(1, N + 1):
        rest = [i for i in range(1, N + 1) if i != k]
        perm = tuple(rest + [k])
        values[k] = solve_peak(perm=perm, env=env)
        print(f"  customer {k:2d} labelled n : z_peak = {values[k]:.6f}")
    ok &= check("no labelling gives z_peak < z_GLM",
                min(values.values()) > z_glm - TOL)

    # Optional: random or exhaustive relabellings.
    perms = []
    if args.full:
        perms = itertools.permutations(range(1, N + 1))
        print(f"\nAll {math.factorial(N)} relabellings (this takes a while)")
    elif args.random > 0:
        import random
        rng = random.Random(args.seed)
        perms = [tuple(rng.sample(range(1, N + 1), N)) for _ in range(args.random)]
        print(f"\n{args.random} random relabellings")

    if perms:
        counts = Counter()
        by_last = defaultdict(Counter)
        worst = (float("inf"), None)
        for perm in perms:
            v = solve_peak(perm=perm, env=env)
            counts[round(v, 4)] += 1
            by_last[perm[-1]][round(v, 4)] += 1
            if v < worst[0]:
                worst = (v, perm)
        print(f"  {sum(counts.values())} relabellings solved")
        for value in sorted(counts):
            print(f"    z_peak = {value:.4f} : {counts[value]:>9d} relabellings")
        ok &= check("no relabelling gives z_peak < z_GLM", worst[0] > z_glm - TOL)
        ok &= check("z_peak depends on the labelling only through the customer "
                    "labelled n", all(len(c) == 1 for c in by_last.values()))
        print(f"  smallest value {worst[0]:.6f} at labelling {worst[1]}")

    env.dispose()
    print("\nAll checks passed." if ok else "\nSOME CHECKS FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
