"""Pick a working NVIDIA GPU when CUDA is available, otherwise CPU."""

from __future__ import annotations


_works: dict[int, bool] = {}


def cuda_available() -> bool:
    try:
        import warnings

        import torch

        warnings.filterwarnings("ignore", message=r"(?s).*compute capability.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=r"(?s).*sm_61.*", category=UserWarning)
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _gpu_works(index: int) -> bool:
    if index in _works:
        return _works[index]
    import torch

    ok = False
    try:
        major, minor = torch.cuda.get_device_capability(index)
        if (int(major), int(minor)) < (7, 5):
            _works[index] = False
            return False
        value = torch.ones(8, device=f"cuda:{index}").mul(2).sum()
        torch.cuda.synchronize(index)
        ok = abs(float(value) - 16.0) < 1e-3
    except Exception:
        ok = False
    _works[index] = ok
    return ok


def best_cuda_index() -> int | None:
    """GPU with the most free memory among devices that can run a kernel."""
    if not cuda_available():
        return None
    import torch

    chosen = None
    free_best = -1
    for index in range(int(torch.cuda.device_count())):
        if not _gpu_works(index):
            continue
        try:
            free, _total = torch.cuda.mem_get_info(index)
        except Exception:
            free = 0
        if int(free) > free_best:
            free_best = int(free)
            chosen = index
    return chosen


def gpu_name(index: int) -> str:
    import torch

    try:
        return str(torch.cuda.get_device_name(index))
    except Exception:
        return f"GPU {index}"


def resolve_device(requested: str) -> tuple[object, str]:
    """Return ``(ultralytics device, backend label)``.

    ``CUDA`` and ``AUTO`` use the NVIDIA GPU that has the most free memory
    and can actually run a kernel. They fall back to CPU otherwise.
    """
    preference = (requested or "AUTO").upper()
    if preference == "CPU" or not cuda_available():
        return "cpu", "CPU"
    index = best_cuda_index()
    if index is None:
        return "cpu", "CPU"
    short = gpu_name(index).replace("NVIDIA GeForce ", "").strip() or f"GPU {index}"
    return index, f"CUDA {short}"
