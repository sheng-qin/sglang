"""Quantization compatibility stubs for the CoreX ixformer shim."""


def __getattr__(name):
    def _missing_quantization_symbol(*args, **kwargs):
        raise NotImplementedError(
            f"sgl_kernel.quantization.{name} is not available in the CoreX ixformer shim"
        )

    return _missing_quantization_symbol
