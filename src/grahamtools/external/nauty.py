from __future__ import annotations

import os
import shutil
import subprocess
from typing import Iterable


NAUTY_GENG = os.environ.get("NAUTY_GENG", "geng")
NAUTY_SHORTG = os.environ.get("NAUTY_SHORTG", "shortg")


def nauty_available() -> bool:
    """
    Returns True iff geng and shortg appear runnable (via PATH or env override).
    """
    geng_path = shutil.which(NAUTY_GENG)
    shortg_path = shutil.which(NAUTY_SHORTG)
    return geng_path is not None and shortg_path is not None


def _looks_like_graph6_line(line: str) -> bool:
    """
    Heuristic filter for graph6 tokens emitted by nauty tools.

    graph6 lines:
      - do not start with '>'
      - are typically a single token (no spaces)
      - are non-empty ASCII strings
    """
    if not line:
        return False
    if line.startswith(">"):
        return False
    if (" " in line) or ("\t" in line):
        return False
    return True


def canon_g6(g6: str) -> str:
    """
    Canonicalize a graph6 string using nauty shortg.

    Robustly extracts the last graph6-like line from stdout.
    """
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


def geng_connected_subcubic_g6(n: int) -> Iterable[str]:
    """
    Stream graph6 strings for connected graphs on n vertices with max degree <= 3.

    geng flags:
      -q quiet
      -g graph6 output
      -c connected
      -D3 max degree 3
    """
    if not nauty_available():
        raise RuntimeError(
            "nauty not available (need 'geng' and 'shortg' in PATH, "
            "or set NAUTY_GENG/NAUTY_SHORTG)."
        )

    cmd = [NAUTY_GENG, "-q", "-g", "-c", "-D3", str(n)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    assert p.stdout is not None
    for line in p.stdout:
        s = line.strip()
        if not s or s.startswith(">"):
            continue
        yield s

    # Drain stderr (debug use); do not spam
    if p.stderr is not None:
        _ = p.stderr.read()

    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"geng failed for n={n} with return code {p.returncode}")
