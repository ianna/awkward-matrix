def _cuda_argument(value, *, depth=0):
    """Never inspect array values: formatting a CUDA array can synchronize."""
    from awkward import contents, highlevel, index, record
    from awkward._errors import SlicingErrorContext

    # Full high-level types recursively visit every field and nested layout.
    # Capture only constant-size outer metadata, without touching form/type,
    # behavior, repr, or data buffers. Deferred notes must not retain the array.
    if isinstance(value, highlevel.Array):
        layout = value._layout
        return f"<Array layout={type(layout).__name__} length={layout.length}>"
    if isinstance(value, highlevel.Record):
        return f"<Record layout={type(value._layout.array).__name__}>"
    if isinstance(value, contents.Content):
        return f"<{type(value).__name__} length={len(value)}>"
    if isinstance(value, record.Record):
        return "<Record>"
    if isinstance(value, index.Index):
        return f"<{type(value).__name__} length={len(value)} dtype={value.dtype}>"
    if isinstance(value, slice):
        return SlicingErrorContext.format_slice(value)
    if value is None or type(value) in (bool, int, float, str, bytes):
        if isinstance(value, (str, bytes)):
            value = value[:72]
        return repr(value)
    if type(value) in (tuple, list) and depth < 3:
        parts = [_cuda_argument(x, depth=depth + 1) for x in value[:8]]
        if len(value) > 8:
            parts.append("...")
        if isinstance(value, tuple):
            return "(" + ", ".join(parts) + ("," if len(value) == 1 else "") + ")"
        return "[" + ", ".join(parts) + "]"
    return f"<{type(value).__name__}>"
