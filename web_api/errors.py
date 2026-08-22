"""Stable mapping from algorithm exceptions to API error details."""

from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError as FutureCancelledError
from typing import Any

from llc_design.core.tank import GainNotReachableError

from .contracts import ErrorDetail

_MESSAGES = {
    "invalid_parameters": "参数不符合约束。",
    "gain_not_reachable": "在指定范围内无法达到目标增益。",
    "resource_not_found": "请求的资源不存在。",
    "task_timeout": "任务执行超时，请稍后重试。",
    "task_cancelled": "任务已取消。",
    "stability_gate_rejected": "稳定性门禁未通过，无法生成代码。",
    "internal_error": "服务内部错误，请联系管理员。",
}


def _exception_details(exc: BaseException) -> dict[str, Any]:
    return {"exception_type": type(exc).__name__}


def _is_stability_gate_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower().replace("_", "-")
    message = str(exc).lower().replace("_", "-")
    name_compact = "".join(character for character in name if character.isalnum())
    message_compact = "".join(character for character in message if character.isalnum())
    return (
        "stability-gate" in name
        or "stability-gate" in message
        or "stability gate" in message
        or "stabilitygate" in name_compact
        or "stabilitygate" in message_compact
        or "稳定性门禁" in message
    )


def map_exception(exc: BaseException, stage: str) -> ErrorDetail:
    """Map a known failure to a stable code and Chinese user-facing message."""

    if isinstance(exc, (FutureCancelledError, asyncio.CancelledError)):
        code = "task_cancelled"
        retryable = False
    elif isinstance(exc, TimeoutError):
        code = "task_timeout"
        retryable = True
    elif _is_stability_gate_error(exc):
        code = "stability_gate_rejected"
        retryable = False
    elif isinstance(exc, GainNotReachableError):
        code = "gain_not_reachable"
        retryable = False
    elif isinstance(exc, KeyError):
        code = "resource_not_found"
        retryable = False
    elif isinstance(exc, ValueError):
        code = "invalid_parameters"
        retryable = False
    else:
        code = "internal_error"
        retryable = False

    return ErrorDetail(
        code=code,
        stage=stage,
        message=_MESSAGES[code],
        details=_exception_details(exc),
        retryable=retryable,
    )


__all__ = ["map_exception"]
