# Graph k-Coloring: Test Inputs + Evaluation Pipeline

Given an undirected graph G(V, E) and an integer k, assign one of k colors to every vertex so that no two adjacent vertices share a color. (Decision problem: a k-coloring either exists or it doesn't.)

Everything (test graphs, evaluation, visualization) lives in `input_eval.py`. **You only need to write your algorithm function and register it; you shouldn't have to touch anything else.**

---

## 1. Quick start

```bash
pip install networkx matplotlib
python input_eval.py        # runs all algorithms -> prints a results table + saves pictures to ./viz/
```

## 2. Plugging in your algorithm

### Function contract

```python
def my_algo(adj, k):
    # adj[v] = list of v's neighbors; nodes are 0..n-1; graph is undirected (u in adj[v] <=> v in adj[u])
    # k      = number of allowed colors; colors are the ints 0..k-1
    return colors    # list[int], colors[v] in range(k)
    # return None if no k-coloring was found
    # (exact algorithms: None means "proved none exists")
```

### Rules (please follow)

- Must be a **top-level `def`**: no lambdas, no nested functions (the runner pickles the function by name).
- **Do not import networkx in your algorithm.** Use only `adj`. networkx is only for generating and drawing graphs.
- Return a `list[int]` of length n, or `None`. numpy integers and bools are rejected as INVALID.
- Don't mutate `adj` (you get a copy during evaluation, but it's good practice).
- Each call has a **10-second timeout** (`evaluate(..., timeout=10)`); on timeout the worker is killed and the run is recorded as TIMEOUT.
- The recursion limit inside the evaluation worker is set to 10000 (enough for backtracking on 1000 nodes).

### Registering

Add a line to the `ALGORITHMS` dict at the bottom of the file:

```python
ALGORITHMS = {
    "greedy": example_greedy,
    "brute_force": brute_force,
    "dsatur": your_dsatur,              # <- add here
    "backtracking": your_backtracking,
}
```

Tip: write your algorithm in its own file (e.g. `algo_dsatur.py`) and `from algo_dsatur import dsatur` in `input_eval.py`, so we avoid merge conflicts from everyone editing the same file.

---

## 3. Test graphs (`build_suite`, seed=42)

| Family | Instances | Best known | Purpose |
|---|---|---|---|
| `complete_n` | complete_20 | 20 (exact) | Sanity check: k=n must succeed, k=n-1 must fail |
| `cycle_*` | cycle_even_100 / cycle_odd_101 | 2 / 3 (exact) | Odd vs. even ring |
| `grid_RxC` | grid_30x30 | 2 (exact) | Large and sparse; a failure at k=2 is easy to spot |
| `crown_n` | crown_50 | 2 (exact) | Greedy trap: greedy visiting a0,b0,a1,b1,... needs about n colors |
| `mycielski_k` | mycielski_6 | 6 (exact) | Triangle-free but high chromatic number; algorithms that guess from local structure (cliques, degrees) tend to fail |
| `random_n{100,500,1000}_p{0.05,0.3,0.7}` | 9 graphs | unknown | The "typical" case; compare algorithms against each other: who still succeeds at the smallest k? |
| `planted_n{100,500,1000}_k{5,20}` | 6 graphs | k (upper bound) | Hidden valid k-coloring (p=0.3); failing at k means missing a coloring that exists |

### Which k values are tried (`test_ks`)

- **Exact known:** `(chi, expect SAT)` and `(chi-1, expect UNSAT)`.
- **Planted:** `(k, expect SAT)` and `(k-1, unknown)` (k is only an upper bound).
- **Random:** starting from the number of colors plain greedy needs, sweep downward over 4 values of k; all have expect = `?`.

---

## 4. Reading the results table

Columns: `graph / n / edges / k / expect / algorithm / result / time(s)`

**expect:** `SAT` = a k-coloring exists; `UNSAT` = none exists; `?` = unknown.

**result:**

| Status | Meaning |
|---|---|
| `OK` | Returned a valid k-coloring |
| `NONE` | Returned None (nothing found / exact algorithm proved none exists) |
| `INVALID` | Returned something that is not a valid k-coloring (conflict, color out of range, wrong format) |
| `TIMEOUT` | Exceeded 10 seconds |
| `ERROR` | Algorithm raised an exception (the message is printed above the table row) |

**Flags at the end of a row:**

- `<- BUG: not a valid coloring`: your algorithm returned an invalid result; fix this first.
- `<- missed`: expect=SAT but you returned None. Normal for heuristics; **for an exact algorithm this is a bug.**
- `<- IMPOSSIBLE (bug)`: expect=UNSAT but you returned a valid coloring, so either the algorithm or the checker is wrong.

---

## 5. Visualization

After `python input_eval.py`, pictures are in `./viz/`:

- `visualize(G, algo, name)`: left = input graph, right = the algorithm's result at k = optimum (or greedy's count for random graphs); conflicting edges are drawn in red. Only small graphs are drawn (`build_viz_suite`); big graphs appear in the table only.
- Three showcases (one picture per algorithm each):
  - `showcase_crown`: greedy fails at k=2 while the optimum is 2
  - `showcase_planted`: regrouped by hidden group, edges only run between clusters
  - `showcase_mycielski`: triangle-free yet needs 4 colors

> ⚠️ **Heads-up: the drawing stage has no timeout.** `_result_panel` calls your algorithm directly in the main process, without the 10-second limit. If your algorithm is slow even on small graphs (e.g. `brute_force` on `grid_6x8` is 2^48), the drawing stage will hang. When debugging, temporarily comment out the drawing loops in `__main__` and just run the table, or add a size check to your algorithm.

---

## 6. Handy snippets

Run one algorithm on one graph:

```python
from input_eval import *
suite = build_suite(seed=42)
G = dict(suite)["crown_50"]
adj = to_adj(G)
print(run_one(your_algo, adj, 2, timeout=10)[:2])   # (status, seconds)
```

Run only some algorithms:

```python
evaluate({"dsatur": your_dsatur}, suite, timeout=10)
```

Check whether a return value is valid: `check_coloring(adj, colors, k)`.

---

## 7. Common pitfalls

1. **Lambda / nested function** -> pickling fails -> shows up as ERROR.
2. **Returning numpy types** -> `check_coloring` says INVALID; convert to plain Python `int`.
3. **Exact algorithms timing out on UNSAT k** (e.g. k=19 on `complete_20`, k=5 on `mycielski_6`) is expected; it takes good pruning / symmetry breaking to get through.
4. **Random graphs have no known optimum**, so there is no `missed` flag; compare by who is still OK at smaller k.
5. Changing `seed` changes the random and planted graphs; keep `seed=42` when comparing results.

Questions or want a new kind of test graph? Ask Suoang.
