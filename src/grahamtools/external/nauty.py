from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Iterable

import networkx as nx


NAUTY_GENG = os.environ.get("NAUTY_GENG", "geng")
NAUTY_SHORTG = os.environ.get("NAUTY_SHORTG", "shortg")
NAUTY_DREADNAUT = os.environ.get("NAUTY_DREADNAUT", "dreadnaut")


def nauty_available() -> bool:
    """Returns True iff geng and shortg appear runnable."""
    return (
        shutil.which(NAUTY_GENG) is not None
        and shutil.which(NAUTY_SHORTG) is not None
    )


def dreadnaut_available() -> bool:
    """Returns True iff dreadnaut appears runnable."""
    return shutil.which(NAUTY_DREADNAUT) is not None


def _looks_like_graph6_line(line: str) -> bool:
    if not line:
        return False
    if line.startswith(">"):
        return False
    if (" " in line) or ("\t" in line):
        return False
    return True


# ---------------------------------------------------------------------------
# Graph6 encoding/decoding helpers
# ---------------------------------------------------------------------------

def edgelist_to_g6(edges: list[tuple[int, int]], n: int) -> str:
    """Convert an edge list on vertices {0..n-1} to a graph6 string."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return nx.to_graph6_bytes(G, header=False).decode("ascii").strip()


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

def canon_g6(g6: str) -> str:
    """Canonicalize a graph6 string using nauty shortg."""
    if not nauty_available():
        raise RuntimeError(
            "nauty not available (need 'geng' and 'shortg' in PATH, "
            "or set NAUTY_GENG/NAUTY_SHORTG)."
        )
    inp = (g6.strip() + "\n").encode("ascii")
    p = subprocess.run(
        [NAUTY_SHORTG, "-q"],
        input=inp,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    lines = [ln.strip() for ln in p.stdout.decode("ascii", errors="replace").splitlines()]
    g6_lines = [ln for ln in lines if _looks_like_graph6_line(ln)]
    if not g6_lines:
        raise RuntimeError(
            "shortg produced no graph6 output.\n"
            f"input={g6!r}\nstdout={p.stdout!r}\nstderr={p.stderr!r}"
        )
    return g6_lines[-1]


# ---------------------------------------------------------------------------
# Automorphism group order
# ---------------------------------------------------------------------------

def _g6_to_dreadnaut_input(g6: str) -> str:
    """Convert a graph6 string to dreadnaut adjacency input."""
    G = nx.from_graph6_bytes(g6.strip().encode("ascii"))
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        G = nx.Graph(G)
    n = G.number_of_nodes()
    lines = [f"n={n} g"]
    for v in range(n):
        neighbors = sorted(G.neighbors(v))
        if neighbors:
            lines.append(f"{v} : {' '.join(str(u) for u in neighbors)};")
        else:
            lines.append(f"{v} : ;")
    lines.append("x")
    lines.append("q")
    return "\n".join(lines) + "\n"


_GRPSIZE_RE = re.compile(r"grpsize=(\d+(?:\.\d+)?(?:\*10\^(\d+))?)")


def _parse_grpsize(output: str) -> int:
    """Parse 'grpsize=N' or 'grpsize=A*10^B' from dreadnaut output."""
    m = _GRPSIZE_RE.search(output)
    if not m:
        raise RuntimeError(f"Could not parse grpsize from dreadnaut output:\n{output}")
    base_str = m.group(1).split("*")[0]
    base = float(base_str)
    exp = int(m.group(2)) if m.group(2) else 0
    result = int(base * (10 ** exp))
    return result


def aut_size_g6(g6: str) -> int:
    """Compute |Aut(G)| for a graph given in graph6 format using dreadnaut."""
    if not dreadnaut_available():
        raise RuntimeError(
            "dreadnaut not available (need 'dreadnaut' in PATH, "
            "or set NAUTY_DREADNAUT)."
        )
    inp = _g6_to_dreadnaut_input(g6)
    p = subprocess.run(
        [NAUTY_DREADNAUT],
        input=inp.encode("ascii"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    combined = p.stdout.decode("ascii", errors="replace") + p.stderr.decode("ascii", errors="replace")
    return _parse_grpsize(combined)


def canon_label_g6(g6: str) -> tuple[str, int]:
    """Return (canonical_g6, aut_group_order) for a graph6 string.

    Uses shortg for canonical form and dreadnaut for aut size.
    """
    canonical = canon_g6(g6)
    aut = aut_size_g6(g6)
    return canonical, aut


# ---------------------------------------------------------------------------
# Graph generation (geng)
# ---------------------------------------------------------------------------

def geng_g6(
    n: int,
    *,
    connected: bool = False,
    max_degree: int | None = None,
    min_degree: int | None = None,
    bipartite: bool = False,
    triangle_free: bool = False,
    min_edges: int | None = None,
    max_edges: int | None = None,
) -> Iterable[str]:
    """Stream graph6 strings from nauty's geng.

    Parameters
    ----------
    n : int
        Number of vertices.
    connected : bool
        Only connected graphs (-c).
    max_degree : int, optional
        Maximum vertex degree (-D).
    min_degree : int, optional
        Minimum vertex degree (-d).
    bipartite : bool
        Only bipartite graphs (-b).
    triangle_free : bool
        Only triangle-free graphs (-t).
    min_edges, max_edges : int, optional
        Edge count bounds (appended after n as 'n min_edges:max_edges').
    """
    if not nauty_available():
        raise RuntimeError(
            "nauty not available (need 'geng' and 'shortg' in PATH, "
            "or set NAUTY_GENG/NAUTY_SHORTG)."
        )

    cmd = [NAUTY_GENG, "-q", "-g"]
    if connected:
        cmd.append("-c")
    if bipartite:
        cmd.append("-b")
    if triangle_free:
        cmd.append("-t")
    if min_degree is not None:
        cmd.append(f"-d{min_degree}")
    if max_degree is not None:
        cmd.append(f"-D{max_degree}")

    # Vertex count and optional edge bounds
    n_spec = str(n)
    if min_edges is not None or max_edges is not None:
        lo = str(min_edges) if min_edges is not None else ""
        hi = str(max_edges) if max_edges is not None else ""
        n_spec += f" {lo}:{hi}"
    cmd.append(n_spec)

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert p.stdout is not None

    for line in p.stdout:
        s = line.strip()
        if not s or s.startswith(">"):
            continue
        yield s

    if p.stderr is not None:
        _ = p.stderr.read()

    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"geng failed for n={n} with return code {p.returncode}")


def geng_connected_subcubic_g6(n: int) -> Iterable[str]:
    """Stream graph6 strings for connected graphs on n vertices with max degree <= 3."""
    return geng_g6(n, connected=True, max_degree=3)
