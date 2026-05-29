"""Compatibility shim for CoreX builds.

The upstream SGLang tree imports many helpers from ``sgl_kernel`` at module
import time. On Iluvatar/CoreX we route those helpers through ixformer's
SGLang-compatible contribution module instead of installing NVIDIA sgl-kernel.

TODO(ixformer): remove this shim once the platform ships a top-level
``sgl_kernel`` package (e.g. ixformer exposing/aliasing
``ixformer.contrib.sgl_kernel`` as ``sgl_kernel``, or a CoreX ``sgl_kernel``
wheel). When that lands, ``import sgl_kernel`` resolves natively and this
directory can be deleted with no other source changes.
"""

from ixformer.contrib.sgl_kernel import *  # noqa: F401,F403

try:
    from ixformer import inference as _ixf_inference
except Exception:
    _ixf_inference = None


def __getattr__(name):
    if _ixf_inference is not None:
        funcs = getattr(_ixf_inference, "functions", None)
        if funcs is not None and hasattr(funcs, name):
            return getattr(funcs, name)

    def _missing_sgl_kernel_symbol(*args, **kwargs):
        raise NotImplementedError(
            f"sgl_kernel.{name} is not available in the CoreX ixformer shim"
        )

    return _missing_sgl_kernel_symbol
