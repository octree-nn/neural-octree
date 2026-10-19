from __future__ import annotations

import sys

from neuoct.engine.solver import NeuOctSolver


def run_mode(mode: str) -> None:
    sys.argv.extend(["SOLVER.run", mode])
    NeuOctSolver.main()
