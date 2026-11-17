"""Public request types only; never sample, dataset, or user identities."""

from contextlib import contextmanager
from contextvars import ContextVar
import torch

PUBLIC_TYPES = ("choice", "noul", "score")
_TASKS = ContextVar("deciform_public_request_types", default=None)


@contextmanager
def task_context(tasks):
    if tasks.dtype != torch.long or tasks.ndim != 1 or bool(((tasks < 0) | (tasks >= 3)).any()):
        raise ValueError("Expected public Choice/Noul/Score enum")
    token = _TASKS.set(tasks)
    try:
        yield
    finally:
        _TASKS.reset(token)
