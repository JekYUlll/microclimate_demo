from __future__ import annotations

from typing import Any


class BaseModel:
    """模型基类（仅用于类型统一）。"""

    def fit(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - 由具体模型实现
        raise NotImplementedError

    def predict(self, *args: Any, **kwargs: Any):  # pragma: no cover - 由具体模型实现
        raise NotImplementedError
