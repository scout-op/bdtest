import os

# Toggle with envs:
#   GCAD_DEBUG: '1'/True enabled (default), '0'/False disabled
#   GCAD_DEBUG_MAX: max prints per tag (default 3)

_RAW_DEBUG = os.environ.get('GCAD_DEBUG', '1')
try:
    DEBUG_ENABLED = str(_RAW_DEBUG).strip().lower() not in ('0', 'false', 'off', 'no')
except Exception:
    DEBUG_ENABLED = True

try:
    MAX_PRINTS = int(os.environ.get('GCAD_DEBUG_MAX', '3'))
except Exception:
    MAX_PRINTS = 3

_state = {}


def is_main_process() -> bool:
    """Return True only for rank-0 (or non-distributed)."""
    try:
        import torch.distributed as dist  # lazy import
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    # Fallback to env
    for key in ('RANK', 'LOCAL_RANK'):
        v = os.environ.get(key)
        if v is not None:
            try:
                return int(v) == 0
            except Exception:
                return True
    return True


def is_data_worker0() -> bool:
    """Return True if not in DataLoader worker, or worker id == 0."""
    try:
        from torch.utils.data import get_worker_info  # lazy import
        wi = get_worker_info()
        if wi is not None and wi.id != 0:
            return False
    except Exception:
        pass
    return True


def debug_print(tag: str, msg_or_fn):
    """Safely print a short debug line a limited number of times per tag.

    - Only prints on rank-0 and DataLoader worker-0
    - Limited by MAX_PRINTS per tag
    - Accepts a string or a zero-arg callable to lazily format the message
    """
    if not DEBUG_ENABLED:
        return
    if not is_main_process():
        return
    if not is_data_worker0():
        return
    cnt = _state.get(tag, 0)
    if cnt >= MAX_PRINTS:
        return
    try:
        msg = msg_or_fn() if callable(msg_or_fn) else str(msg_or_fn)
    except Exception as e:
        msg = f"[format-error] {e}"
    print(f"[GCAD][{tag}] {msg}", flush=True)
    _state[tag] = cnt + 1
