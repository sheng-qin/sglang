import os


def use_ixformer() -> bool:
    if os.getenv("SGLANG_USE_IXFORMER", "").lower() in ("1", "true", "yes", "on"):
        return True

    try:
        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()
    except Exception:
        return False

    if server_args is None:
        return False

    return "ixformer" in {
        getattr(server_args, "attention_backend", None),
        getattr(server_args, "prefill_attention_backend", None),
        getattr(server_args, "decode_attention_backend", None),
    }
