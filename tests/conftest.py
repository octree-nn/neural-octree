"""Test bootstrap for CPU-only hosts.

Importing ``neuoct`` pulls in the OCNN/triton stack, which cannot initialise on
a machine without a CUDA driver.  The new ``neuoct.rendering_rt`` package itself
only needs torch/numpy, so when OCNN is unavailable we register lightweight
package stubs (with correct ``__path__``) that let submodules import without
executing the heavy package ``__init__`` files.  When OCNN *is* importable --
i.e. on the normal GPU host -- nothing is stubbed and the real package is used.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
_TOOLS = _REPO / "tools" / "neuoct_rt"

# ``tools/neuoct_rt/render_scene.py`` owns the graph-preserving glTF loader and
# the legacy matrix camera adapter, so tests import it as a module.  Adding
# ``tools`` to ``sys.path`` here keeps that import independent of the working
# directory pytest is launched from.
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))
# ``src`` must be importable whether or not OCNN is present, so it is added
# unconditionally rather than only inside the CPU-stub branch below.
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_STUBBED_PACKAGES = (
    "neuoct",
    "neuoct.geometry",
    "neuoct.models",
    "neuoct.engine",
    "neuoct.data",
    "neuoct.rendering",
    "neuoct.config",
)


def _ocnn_available() -> bool:
    try:
        import ocnn  # noqa: F401

        return True
    except Exception:
        return False


def _install_cpu_stubs() -> None:
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    for name in list(sys.modules):
        if name == "neuoct" or name.startswith("neuoct."):
            del sys.modules[name]
    for name in _STUBBED_PACKAGES:
        module = types.ModuleType(name)
        module.__path__ = [str(_SRC / name.replace(".", "/"))]
        sys.modules[name] = module
    sys.modules["neuoct"].__version__ = "0.1.0"


if not _ocnn_available():
    _install_cpu_stubs()
