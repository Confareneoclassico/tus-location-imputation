#!/usr/bin/env python3
"""Checks of factorial_indices (src/location_imputation.py) against analytic Sobol' indices.

Run with  python tests/test_indices.py  (no data needed, a few seconds). Exits with 1 on failure.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from location_imputation import factorial_indices  # noqa: E402

FAILED = []


def close(name, got, want, tol=1e-12):
    ok = np.allclose(got, want, atol=tol, equal_nan=True)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f": got {np.round(got, 6)}, want {want}"))
    if not ok:
        FAILED.append(name)


def grid(**levels):
    rows = list(itertools.product(*levels.values()))
    return pd.DataFrame(rows, columns=list(levels))


# 1. Additive model: Y = a + b, a in {0, 1}, b in {0, 2}: Var(a) = 1/4, Var(b) = 1, no interaction
X = grid(a=[0, 1], b=[0, 2])
S, ST = factorial_indices(X, X.a + X.b)
close("additive, first-order", S[:, 0], [0.2, 0.8])
close("additive, total equals first-order", ST[:, 0], [0.2, 0.8])

# 2. Pure interaction: Y = a * b with a, b in {-1, 1}: no main effects, each total index is 1
X = grid(a=[-1, 1], b=[-1, 1])
S, ST = factorial_indices(X, X.a * X.b)
close("interaction, first-order", S[:, 0], [0.0, 0.0])
close("interaction, total", ST[:, 0], [1.0, 1.0])

# 3. Three factors with uneven numbers of levels and an interaction, against a direct ANOVA
#    decomposition: Y = a + 2 b + a * c, a in {0, 1, 2}, b in {0, 1}, c in {-1, 0, 1, 2}
X = grid(a=[0, 1, 2], b=[0, 1], c=[-1, 0, 1, 2])
Y = X.a + 2 * X.b + X.a * X.c
S, ST = factorial_indices(X, Y)
f0 = Y.mean()
fa = X.a.map(Y.groupby(X.a).mean()) - f0
fb = X.b.map(Y.groupby(X.b).mean()) - f0
fc = X.c.map(Y.groupby(X.c).mean()) - f0
fac = Y.groupby([X.a, X.c]).transform("mean") - f0 - fa - fc
V = Y.var(ddof=0)
close("uneven levels, first-order", S[:, 0], [fa.var(ddof=0) / V, fb.var(ddof=0) / V, fc.var(ddof=0) / V])
close("uneven levels, total", ST[:, 0], [(fa.var(ddof=0) + fac.var(ddof=0)) / V, fb.var(ddof=0) / V,
                                         (fc.var(ddof=0) + fac.var(ddof=0)) / V])
close("uneven levels, decomposition sums to one",
      S[:, 0].sum() + fac.var(ddof=0) / V, 1.0)

# 4. Several outputs at once, one of them constant (indices undefined)
S, ST = factorial_indices(X, np.column_stack([Y, np.ones(len(Y))]))
close("constant output gives NaN", np.r_[S[:, 1], ST[:, 1]], [np.nan] * 6)

# 5. Random outputs on a 3 x 2 x 4 x 13 design: bounds that hold for any function
rng = np.random.default_rng(1)
X = grid(a=[1, 5, 20], b=["no", "yes"], c=list("wxyz"), d=[f"D{i}" for i in range(13)])
Yr = rng.normal(size=(len(X), 5)) + np.outer(X.a, [1, 0, 2, 0, 0.1])
S, ST = factorial_indices(X, Yr)
ok = bool((S >= -1e-12).all() and (S <= ST + 1e-12).all() and (ST <= 1 + 1e-12).all()
          and (S.sum(0) <= 1 + 1e-12).all() and (ST.sum(0) >= 1 - 1e-12).all())
print(f"{'PASS' if ok else 'FAIL'}  random outputs: 0 <= S <= ST <= 1, sum S <= 1 <= sum ST")
if not ok:
    FAILED.append("bounds")

# 6. An incomplete design is refused
try:
    factorial_indices(X.iloc[:-1], Yr[:-1])
    print("FAIL  incomplete design refused")
    FAILED.append("incomplete")
except ValueError:
    print("PASS  incomplete design refused")

# 7. Optimal-transport index (src/make_results_figures.py). The squared 2-Wasserstein distance
#    between two samples of any sizes equals the optimal assignment between copies of them
#    replicated to a common size; the index is 0 for a factor without effect, 1 for a factor that
#    determines the output, and lies between half the first-order and the total Sobol' index.
from make_results_figures import assignment, ot_univariate, w2sq_exact  # noqa: E402


def w2_assignment(a, b):
    n = int(np.lcm(len(a), len(b)))
    A, B = np.repeat(np.sort(a), n // len(a)), np.repeat(np.sort(b), n // len(b))
    C = (A[:, None] - B[None, :]) ** 2
    return float(C[np.arange(n), assignment(C)].mean())


got, want = [], []
for n, m in ((6, 4), (9, 6), (12, 8), (10, 15), (7, 3), (5, 5)):
    a, b = rng.lognormal(size=n), rng.standard_t(2, size=m)
    got.append(w2sq_exact(np.sort(a), np.sort(b)))
    want.append(w2_assignment(a, b))
close("OT: W2 of samples of unequal sizes equals the optimal assignment", got, want, tol=1e-10)
X = grid(a=[0, 1, 2], b=[0, 1], c=[0, 1, 2, 3])
iota = ot_univariate(X.assign(y=X.a + 0.3 * X.b), "y", ["a", "b", "c"])
close("OT: factor without effect has index 0", iota["c"], 0.0)
iota = ot_univariate(X.assign(y=X.a ** 2), "y", ["a", "b", "c"])
close("OT: factor that determines the output has index 1", iota["a"], 1.0)
ok = True
for trial in range(25):
    y = rng.lognormal(size=len(X)) * rng.choice([0.1, 1.0, 10.0]) + X.a * rng.normal()
    S, ST = factorial_indices(X, y.to_numpy())
    iota = ot_univariate(X.assign(y=y), "y", ["a", "b", "c"])
    v = np.array([iota[f] for f in ("a", "b", "c")])
    ok = ok and bool((v >= S[:, 0] / 2 - 1e-12).all() and (v <= ST[:, 0] + 1e-12).all())
print(f"{'PASS' if ok else 'FAIL'}  OT: random outputs: S/2 <= iota <= ST")
if not ok:
    FAILED.append("OT bounds")

print(f"{'all' if not FAILED else len(FAILED)} {'tests passed' if not FAILED else 'tests failed'}")
sys.exit(1 if FAILED else 0)
