"""Minimal stub of the torch package for testing."""

from typing import Any, Iterable, List
import numpy as np

class Tensor:
    def __init__(self, data: Iterable[Any]):
        self.data = np.array(data)

    def size(self, dim: int | None = None):
        return self.data.shape if dim is None else self.data.shape[dim]

    def tolist(self) -> List[Any]:
        return self.data.tolist()

    def unsqueeze(self, dim: int):
        self.data = np.expand_dims(self.data, axis=dim)
        return self

    def squeeze(self, dim: int | None = None):
        self.data = np.squeeze(self.data, axis=dim)
        return self

    def __repr__(self):
        return f"Tensor({self.data!r})"


def tensor(data: Iterable[Any], dtype: Any | None = None) -> Tensor:
    return Tensor(data)


def zeros(size: int, dtype: Any | None = None) -> Tensor:
    return Tensor([0] * size)


def cat(tensors: Iterable[Tensor], dim: int = 0) -> Tensor:
    arrays = [t.data for t in tensors]
    return Tensor(np.concatenate(arrays, axis=dim))


def stack(tensors: Iterable[Tensor], dim: int = 0) -> Tensor:
    arrays = [t.data for t in tensors]
    return Tensor(np.stack(arrays, axis=dim))


class cuda:
    @staticmethod
    def empty_cache() -> None:
        pass
