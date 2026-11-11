"""DSATUR-based exact k-coloring (undirected simple graphs).

Input:  adj[v] lists neighbors of vertex v, and k available colors (0..k-1).
Output: a valid colors list or None if the graph is not k-colorable.

The fast initial attempt uses greedy DSATUR. If it exceeds k colors,
DSATUR-guided backtracking checks all valid possibilities.
"""


def dsatur(adj: list[list[int]], k: int) -> list[int] | None:
    n = len(adj)
    if n == 0:
        return []
    if k <= 0:
        return None
    if any(v in adj[v] for v in range(n)):
        return None  # Self-loop: no valid proper coloring.

    degree = [len(neighbors) for neighbors in adj]

    def pick_vertex(colors, forbidden):
        # Highest saturation, then degree, then lowest vertex ID.
        return max(
            (v for v in range(n) if colors[v] == -1),
            key=lambda v: (len(forbidden[v]), degree[v], -v),
        )

    # Try fast greedy DSATUR first: a successful result is always valid.
    greedy_colors = [-1] * n
    greedy_forbidden = [set() for _ in range(n)]
    for _ in range(n):
        v = pick_vertex(greedy_colors, greedy_forbidden)
        c = 0
        while c in greedy_forbidden[v]:
            c += 1
        if c >= k:
            break  # A greedy failure does NOT prove k-coloring impossible.
        greedy_colors[v] = c
        for u in adj[v]:
            if greedy_colors[u] == -1:
                greedy_forbidden[u].add(c)
    else:
        return greedy_colors

    # Exact fallback: backtrack with DSATUR's vertex-ordering heuristic.
    colors = [-1] * n
    # Counts are necessary: multiple colored neighbors can share a color.
    forbidden_counts = [{} for _ in range(n)]

    def search(assigned: int, used: int) -> bool:
        if assigned == n:
            return True
        v = pick_vertex(colors, forbidden_counts)
        forbidden = forbidden_counts[v]

        # Color symmetry: reuse old colors or introduce one new color.
        for c in range(min(used + 1, k)):
            if c in forbidden:
                continue
            colors[v] = c
            for u in adj[v]:
                if colors[u] == -1:
                    counts = forbidden_counts[u]
                    counts[c] = counts.get(c, 0) + 1

            if search(assigned + 1, max(used, c + 1)):
                return True

            for u in adj[v]:
                if colors[u] == -1:
                    counts = forbidden_counts[u]
                    counts[c] -= 1
                    if counts[c] == 0:
                        del counts[c]
            colors[v] = -1
        return False

    return colors if search(0, 0) else None
