"""
Graph coloring: test inputs + evaluation pipeline.

HOW TO USE
    1. Teammates write algorithms as functions with this contract:
           def my_algo(adj):
               # adj: list of lists. adj[v] = list of neighbors of v.
               # nodes are 0..n-1, graph is undirected (u in adj[v] <=> v in adj[u]).
               ...
               return colors      # list of ints, colors[v] = color of v (0, 1, 2, ...)
       Algorithms must NOT import networkx. networkx is only used here to
       generate and draw graphs.
    2. Add them to the ALGORITHMS dict at the bottom.
    3. Run:  python coloring_eval.py          (table + pictures in ./viz/)

Node labels are always integers 0..n-1.
"""
import itertools
import os
import random
import time

import matplotlib
matplotlib.use("Agg")  # remove this line if you want plt.show() windows
import networkx as nx


# =====================================================================
# PART 1: TEST GRAPHS
# Each function returns a graph. "best_known" = the best number of colors
# we know is possible (exact chromatic number when marked "exact").
# An algorithm using fewer colors than best_known is impossible, so if you
# see that, the algorithm's coloring is wrong.
# =====================================================================

def random_graph(n, p, seed):
    """
    Erdos-Renyi random graph: each pair of nodes is connected with probability p.
    Characteristics: no special structure; p controls density.
      - low p (sparse): few edges, easy, few colors needed.
      - high p (dense): many edges, needs many colors, hard to color well.
    Why: the "typical" case. Best-known color count is unknown (finding the
    optimum is NP-hard), so we compare algorithms against each other.
    """
    G = nx.gnp_random_graph(n, p, seed=seed)
    G.graph["best_known"] = None
    return G


def planted_graph(n, k, p, seed):
    """
    Random graph that is GUARANTEED to be colorable with k colors.
    How: nodes are secretly split into k groups; edges are only added between
    different groups (with probability p), never inside a group.
    Characteristics: looks random, but hides a valid k-coloring.
    Why: gives a known target. An algorithm using exactly k colors is optimal;
    using more shows how far from optimal it is. (Fewer than k may be possible
    in rare cases, so k is an upper bound on the true optimum.)
    """
    rng = random.Random(seed)
    group = {v: v % k for v in range(n)}
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v in itertools.combinations(range(n), 2):
        if group[u] != group[v] and rng.random() < p:
            G.add_edge(u, v)
    G.graph["best_known"] = k
    return G


def complete_graph(n):
    """
    Complete graph: every node is connected to every other node.
    Characteristics: the densest possible graph. Every node needs its own color.
    Needs exactly n colors (exact). Why: sanity check; any correct algorithm
    must use exactly n colors here.
    """
    G = nx.complete_graph(n)
    G.graph["best_known"] = n
    return G


def cycle_graph(n):
    """
    Cycle: nodes in a ring, each connected to 2 neighbors.
    Characteristics: very sparse (every node has degree 2).
      - even n: needs exactly 2 colors.
      - odd n: needs exactly 3 colors (can't alternate 2 colors around an odd ring).
    Why: tiny, simple, with a known answer; tests handling of the odd/even case.
    """
    G = nx.cycle_graph(n)
    G.graph["best_known"] = 2 if n % 2 == 0 else 3
    return G


def grid_graph(rows, cols):
    """
    Grid (like a chessboard): each node connected to up/down/left/right.
    Characteristics: sparse, regular, no triangles. Needs exactly 2 colors
    (checkerboard pattern). Why: a large graph where the optimum is known and
    small, so extra colors from an algorithm are easy to spot.
    """
    G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(rows, cols))
    G.graph["best_known"] = 2
    return G


def crown_graph(n):
    """
    Crown graph: two rows of n nodes (a0..a(n-1) and b0..b(n-1));
    a_i is connected to b_j for every j != i.
    Characteristics: needs only 2 colors (all a's one color, all b's another),
    BUT a greedy algorithm that happens to color a0, b0, a1, b1, ... in that
    order wastes colors and uses n colors.
    Why: a classic trap that shows how much a greedy algorithm depends on the
    order in which it visits nodes. Node order here is deliberately a0,b0,a1,b1,...
    """
    G = nx.Graph()
    G.add_nodes_from(range(2 * n))   # fixes the node order: a0, b0, a1, b1, ...
    # label a_i = 2*i, b_i = 2*i+1
    for i, j in itertools.product(range(n), repeat=2):
        if i != j:
            G.add_edge(2 * i, 2 * j + 1)
    G.graph["best_known"] = 2
    return G


def mycielski_graph(k):
    """
    Mycielski graph: built by a recursive construction that raises the number
    of colors needed by 1 each step, while never creating a triangle.
    Characteristics: sparse-looking (no triangles), yet needs exactly k colors
    (exact). Why: shows that "few edges / no triangles" does NOT mean "few
    colors". Algorithms that guess from local structure (cliques, degrees)
    tend to use too many colors here.
    """
    G = nx.mycielski_graph(k)
    G = nx.convert_node_labels_to_integers(G)
    G.graph["best_known"] = k
    return G


def build_suite(seed=0):
    """Returns a list of (name, graph). Same seed -> same graphs every time."""
    suite = [
        ("complete_20", complete_graph(20)),
        ("cycle_even_100", cycle_graph(100)),
        ("cycle_odd_101", cycle_graph(101)),
        ("grid_30x30", grid_graph(30, 30)),
        ("crown_50", crown_graph(50)),
        ("mycielski_6", mycielski_graph(6)),
    ]
    # random graphs: vary size and density
    for n in (100, 500, 1000):
        for p in (0.05, 0.3, 0.7):
            suite.append((f"random_n{n}_p{p}", random_graph(n, p, seed)))
    # planted graphs: hidden coloring with k colors
    for n in (100, 500, 1000):
        for k in (5, 20):
            suite.append((f"planted_n{n}_k{k}", planted_graph(n, k, 0.3, seed)))
    for name, G in suite:
        G.graph["name"] = name
    return suite


# =====================================================================
# PART 2: EVALUATION PIPELINE
# networkx graph -> adjacency list -> algorithm -> list of colors -> check.
# =====================================================================

def to_adj(G):
    """nx.Graph (nodes 0..n-1) -> list of lists. This is what algorithms receive."""
    return [sorted(G[v]) for v in range(G.number_of_nodes())]


def check_coloring(adj, colors):
    """
    Valid if (1) colors is a list with one int per node and
    (2) no edge connects two nodes with the same color.
    Returns (is_valid, number_of_colors_used).
    """
    if not isinstance(colors, (list, tuple)) or len(colors) != len(adj):
        return False, None
    if any(c is None for c in colors):
        return False, None
    for u, nbrs in enumerate(adj):
        for v in nbrs:
            if colors[u] == colors[v]:
                return False, None
    return True, len(set(colors))


def run_one(algo, adj, repeats=3):
    """
    Runs algo on a fresh copy of adj, `repeats` times.
    Time = the fastest run. Colors/validity are taken from the last run.
    """
    best_time = float("inf")
    colors = None
    for _ in range(repeats):
        fresh = [list(nbrs) for nbrs in adj]   # algo can't corrupt the input
        start = time.perf_counter()
        colors = algo(fresh)
        best_time = min(best_time, time.perf_counter() - start)
    valid, k = check_coloring(adj, colors)
    return best_time, valid, k, colors


def evaluate(algorithms, suite):
    """Runs every algorithm on every graph and prints a results table."""
    header = f"{'graph':<20}{'n':>6}{'edges':>8}{'best':>6}  {'algorithm':<14}{'colors':>7}{'time(s)':>10}"
    print(header)
    print("-" * len(header))
    for name, G in suite:
        adj = to_adj(G)
        best = G.graph["best_known"]
        for algo_name, algo in algorithms.items():
            t, valid, k, _ = run_one(algo, adj)
            colors_str = str(k) if valid else "INVALID"
            print(f"{name:<20}{G.number_of_nodes():>6}{G.number_of_edges():>8}"
                  f"{str(best) if best else '?':>6}  {algo_name:<14}{colors_str:>7}{t:>10.4f}")
        print()


# =====================================================================
# PART 3: VISUALIZATION
# Drawing only makes sense for small/sparse graphs (a 1000-node or 100k-edge
# graph is an unreadable blob). So we draw SMALL versions of each graph family
# (VIZ_SUITE below); the big graphs are for the numbers table only.
# =====================================================================

# Okabe-Ito palette: distinguishable for color-blind readers, prints OK in gray.
_PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
            "#D55E00", "#CC79A7", "#999999", "#000000", "#8DD3C7",
            "#BEBADA", "#FB8072", "#80B1D3", "#FDB462", "#B3DE69", "#FCCDE5"]


def _text_color(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 140 else "white"


def _layout(G):
    """Fixed positions so the 'input' and 'colored' panels line up."""
    n = G.number_of_nodes()
    kind = G.graph.get("name", "").split("_")[0]
    if kind in ("cycle", "complete"):
        return nx.circular_layout(G)
    if kind == "crown":      # a_i = 2i on the top row, b_i = 2i+1 on the bottom row
        return {v: (v // 2, 1 if v % 2 == 0 else 0) for v in range(n)}
    if kind == "grid":
        cols = G.graph["cols"]
        return {v: (v % cols, -(v // cols)) for v in range(n)}
    if n <= 60:
        return nx.kamada_kawai_layout(G)
    return nx.spring_layout(G, seed=0, k=1.5 / n ** 0.5, iterations=200)


def _draw(ax, G, pos, colors, title, highlight_conflicts=True):
    """Draw one panel. colors=None -> gray (uncolored input)."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    node_size = 700 if n <= 25 else 350 if n <= 60 else 90 if n <= 150 else 25
    edge_w = 1.2 if m <= 150 else 0.6 if m <= 600 else 0.25
    edge_a = 0.9 if m <= 150 else 0.6 if m <= 600 else 0.35
    fs = 11 if n <= 25 else 9
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_w, alpha=edge_a, edge_color="#555555")
    if colors is not None and highlight_conflicts:
        bad = [(u, v) for u, v in G.edges if colors[u] == colors[v]]
        if bad:
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=bad, width=3, edge_color="red")
            title += f"\n{len(bad)} CONFLICTS (red)"
    cols = ["#DDDDDD"] * n if colors is None else [_PALETTE[c % len(_PALETTE)] for c in colors]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=cols,
                           edgecolors="black", linewidths=0.8)
    if n <= 60:
        for v, (x, y) in pos.items():
            txt = str(v) if colors is None else str(colors[v])
            tc = "black" if colors is None else _text_color(cols[v])
            ax.text(x, y, txt, ha="center", va="center", fontsize=fs,
                    fontweight="bold" if colors is not None else "normal", color=tc)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    ax.margins(0.1)


def _save(fig, name, save_dir, show):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.png")
    fig.savefig(path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return path


def visualize(G, colors, algo_name="", save_dir="viz", show=False):
    """Two panels: input graph (gray, node ids) | colored by `colors` (color index in node)."""
    import matplotlib.pyplot as plt
    name = G.graph.get("name", "graph")
    pos = _layout(G)
    best = G.graph["best_known"]
    k = len(set(colors))
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="white")
    _draw(axes[0], G, pos, None,
          f"Input: {name}\n(n = {G.number_of_nodes()}, edges = {G.number_of_edges()}; number = node id)")
    _draw(axes[1], G, pos, colors,
          f"{algo_name}: {k} colors (best known: {best if best else '?'})\n(number = color index)")
    return _save(fig, f"{name}_{algo_name}", save_dir, show)


# ---------------------------------------------------------------------
# SHOWCASES: one picture per graph family, drawn so the family's defining
# property is visible to the eye. (Pictures only; not used by the algorithms.)
# ---------------------------------------------------------------------

def _exact_coloring(adj, k):
    """Tiny backtracking, only for drawing the optimum on small graphs."""
    n = len(adj)
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    col = [-1] * n
    def go(i):
        if i == n:
            return True
        v = order[i]
        for c in range(k):
            if all(col[u] != c for u in adj[v]):
                col[v] = c
                if go(i + 1):
                    return True
                col[v] = -1
        return False
    return col if go(0) else None


def showcase_crown(n=6, algo=None, algo_name="greedy", save_dir="viz", show=False):
    """
    PROPERTY: needs only 2 colors, but order-dependent greedy uses n.
    Drawn as two rows. Panel 3 shows the 2-coloring: top row one color, bottom row another.
    """
    import matplotlib.pyplot as plt
    G = crown_graph(n)
    G.graph["name"] = f"crown_{n}"
    pos = {v: (v // 2, 1 if v % 2 == 0 else 0) for v in range(2 * n)}
    greedy = (algo or example_greedy)(to_adj(G))
    opt = [v % 2 for v in range(2 * n)]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5), facecolor="white")
    _draw(axes[0], G, pos, None, f"Crown graph, n={n}: every a (top) is linked\nto every b (bottom) except its twin")
    _draw(axes[1], G, pos, greedy, f"{algo_name}, visiting 0,1,2,...: {len(set(greedy))} colors\n(each twin pair gets a new color)")
    _draw(axes[2], G, pos, opt, "Optimal: 2 colors\n(top row one color, bottom row the other)")
    return _save(fig, f"showcase_crown_{n}_{algo_name}", save_dir, show)


def showcase_planted(n=24, k=4, p=0.35, seed=42, algo=None, algo_name="greedy",
                     save_dir="viz", show=False):
    """
    PROPERTY: looks random, but hides a valid k-coloring (no edge inside a hidden group).
    Panel 1: input as a tangle. Panel 2: algorithm result on the tangle.
    Panel 3: SAME graph with nodes regrouped by hidden group -> edges only run between clusters.
    """
    import math
    import matplotlib.pyplot as plt
    G = planted_graph(n, k, p, seed)
    G.graph["name"] = f"planted_n{n}_k{k}"
    tangle = nx.spring_layout(G, seed=1, iterations=200)
    hidden = [v % k for v in range(n)]
    pos = {}
    for g in range(k):
        members = [v for v in range(n) if hidden[v] == g]
        cx, cy = 3 * math.cos(2 * math.pi * g / k + math.pi / 4), 3 * math.sin(2 * math.pi * g / k + math.pi / 4)
        for i, v in enumerate(members):
            r = 0.9
            a = 2 * math.pi * i / len(members)
            pos[v] = (cx + r * math.cos(a), cy + r * math.sin(a))
    res = (algo or example_greedy)(to_adj(G))
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5), facecolor="white")
    _draw(axes[0], G, tangle, None, f"Input: n={n}, edges={G.number_of_edges()}\nlooks like a random tangle")
    _draw(axes[1], G, tangle, res, f"{algo_name}: {len(set(res))} colors\n(a {k}-coloring exists)")
    _draw(axes[2], G, pos, hidden, f"Same graph regrouped by hidden group: {k} colors\nno edge inside a cluster, so {k} colors suffice")
    return _save(fig, f"showcase_planted_n{n}_k{k}_{algo_name}", save_dir, show)


def showcase_mycielski(algo=None, algo_name="greedy", save_dir="viz", show=False):
    """
    PROPERTY: triangle-free (no 3 mutually adjacent nodes) yet needs 4 colors.
    Classic drawing: outer 5-cycle, inner ring of 'shadow' nodes, one center node.
    """
    import math
    import matplotlib.pyplot as plt
    G = mycielski_graph(4)
    G.graph["name"] = "mycielski_4"
    assert sum(nx.triangles(G).values()) == 0
    cyc = [0, 1, 2, 4, 3]                       # the outer 5-cycle, in order
    shadow = {0: 5, 1: 6, 2: 7, 3: 8, 4: 9}     # shadow[v] is adjacent to v's neighbors
    for v, s in shadow.items():
        assert set(G[s]) - {10} == set(G[v]) & set(cyc)
    pos = {10: (0, 0)}
    for i, v in enumerate(cyc):
        a = math.pi / 2 + 2 * math.pi * i / 5
        pos[v] = (2.0 * math.cos(a), 2.0 * math.sin(a))
        pos[shadow[v]] = (1.0 * math.cos(a), 1.0 * math.sin(a))
    adj = to_adj(G)
    res = (algo or example_greedy)(adj)
    opt = _exact_coloring(adj, 4)
    assert _exact_coloring(adj, 3) is None      # 3 colors is impossible
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5), facecolor="white")
    _draw(axes[0], G, pos, None, "Mycielski graph (Grotzsch): 11 nodes, 20 edges\n0 triangles anywhere")
    _draw(axes[1], G, pos, res, f"{algo_name}: {len(set(res))} colors")
    _draw(axes[2], G, pos, opt, "Optimal: 4 colors (3 is impossible)\nsparse and triangle-free, yet needs 4")
    return _save(fig, f"showcase_mycielski_4_{algo_name}", save_dir, show)


def build_viz_suite(seed=42):
    """Small, readable versions of each graph family, for pictures only."""
    G_grid = grid_graph(6, 8)
    G_grid.graph["cols"] = 8
    suite = [
        ("complete_8", complete_graph(8)),
        ("cycle_odd_11", cycle_graph(11)),
        ("grid_6x8", G_grid),
        ("crown_6", crown_graph(6)),          # greedy wastes colors here
        ("mycielski_4", mycielski_graph(4)),  # Grotzsch-like, 11 nodes
        ("mycielski_5", mycielski_graph(5)),  # 23 nodes
        ("random_n30_p0.15", random_graph(30, 0.15, seed)),
        ("planted_n30_k4", planted_graph(30, 4, 0.25, seed)),
    ]
    for name, G in suite:
        G.graph["name"] = name
    return suite


# =====================================================================
# PART 4: PLUG IN ALGORITHMS HERE
# =====================================================================

def example_greedy(adj):
    """Placeholder so the pipeline runs. Replace/add your teammates' algorithms."""
    colors = [-1] * len(adj)
    for v in range(len(adj)):
        used = {colors[u] for u in adj[v] if colors[u] != -1}
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    return colors


ALGORITHMS = {
    "greedy": example_greedy,
    # "dsatur": their_dsatur_function,
    # "backtracking": their_backtracking_function,
}


if __name__ == "__main__":
    suite = build_suite(seed=42)
    evaluate(ALGORITHMS, suite)
    for name, G in build_viz_suite():
        for algo_name, algo in ALGORITHMS.items():
            colors = algo(to_adj(G))
            print("saved", visualize(G, colors, algo_name))
    for algo_name, algo in ALGORITHMS.items():
        print("saved", showcase_crown(6, algo, algo_name))
        print("saved", showcase_planted(24, 4, 0.35, 42, algo, algo_name))
        print("saved", showcase_mycielski(algo, algo_name))
