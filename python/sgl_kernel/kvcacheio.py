"""KV cache I/O compatibility stubs for the CoreX ixformer shim."""


def __getattr__(name):
    def _missing_kvcacheio_symbol(*args, **kwargs):
        raise NotImplementedError(
            f"sgl_kernel.kvcacheio.{name} is not available in the CoreX ixformer shim"
        )

    return _missing_kvcacheio_symbol
