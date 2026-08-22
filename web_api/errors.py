"""Stable mapping from algorithm exceptions to API error details."""

from __future__ import annotations

import asyncio
import re
from concurrent.futures import CancelledError as FutureCancelledError
from typing import Any

from llc_design.core.tank import GainNotReachableError
from llc_design.dynamics.plant import PlantModelError

from .contracts import ErrorCode, ErrorDetail

_MESSAGES = {
    ErrorCode.INVALID_PARAMETERS: "参数不符合约束。",
    ErrorCode.GAIN_NOT_REACHABLE: "在指定范围内无法达到目标增益。",
    ErrorCode.CORE_NO_SOLUTION: "未找到满足约束的磁芯方案。",
    ErrorCode.DEVICE_NOT_FOUND: "请求的器件不存在。",
    ErrorCode.RESOURCE_NOT_FOUND: "请求的资源不存在。",
    ErrorCode.INDUCTOR_NO_SOLUTION: "未找到满足约束的电感方案。",
    ErrorCode.MAGNETICS_NO_SOLUTION: "未找到满足约束的磁性器件方案。",
    ErrorCode.SIMULATION_NON_CONVERGENCE: "仿真未收敛，请检查参数或约束。",
    ErrorCode.UNSTABLE_LOOP: "控制环路未通过稳定性检查。",
    ErrorCode.STABILITY_GATE_REJECTED: "稳定性门禁未通过，无法生成代码。",
    ErrorCode.TASK_TIMEOUT: "任务执行超时，请稍后重试。",
    ErrorCode.TASK_CANCELLED: "任务已取消。",
    ErrorCode.TASK_EXPIRED: "任务已过期。",
    ErrorCode.INTERNAL_ERROR: "服务内部错误，请联系管理员。",
}


def _exception_details(exc: BaseException) -> dict[str, Any]:
    try:
        message = str(exc)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        message = ""
    # Preserve useful constraint/convergence context while preventing local
    # filesystem names from crossing the API boundary.
    safe_message = re.sub(r"(?:(?:[A-Za-z]:)?/)[^\s,;:]+", "<path>", message)
    safe_message = re.sub(r"[A-Za-z]:\\[^\s,;:]+(?:\\[^\s,;:]+)*", "<path>", safe_message)
    safe_message = re.sub(r"\\\\[^\\\s]+(?:\\[^\\\s]+)+", "<path>", safe_message)
    details: dict[str, Any] = {
        "exception_type": type(exc).__name__,
        "message": safe_message,
    }
    lowered = safe_message.lower()
    if any(token in lowered for token in ("constraint", "parameter", "invalid", "limit", "positive")):
        details["constraint"] = safe_message
    if any(token in lowered for token in ("converg", "residual", "iteration", "steady-state")):
        details["convergence"] = safe_message
    return details


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


def _message(exc: BaseException) -> str:
    try:
        return str(exc).lower()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return ""


def _contains_any(message: str, *terms: str) -> bool:
    return any(term in message for term in terms)


def map_exception(exc: BaseException, stage: str) -> ErrorDetail:
    """Map a known failure to a stable code and Chinese user-facing message."""

    if isinstance(exc, (FutureCancelledError, asyncio.CancelledError)):
        code = ErrorCode.TASK_CANCELLED
        retryable = False
    elif isinstance(exc, TimeoutError):
        code = ErrorCode.TASK_TIMEOUT
        retryable = True
    elif _is_stability_gate_error(exc):
        code = ErrorCode.STABILITY_GATE_REJECTED
        retryable = False
    elif "expired" in _message(exc) or "expiration" in _message(exc) or "expired" in type(exc).__name__.lower():
        code = ErrorCode.TASK_EXPIRED
        retryable = False
    elif isinstance(exc, PlantModelError) or _contains_any(
        _message(exc), "not converge", "non-convergence", "nonconvergence", "residual", "steady-state solve failed"
    ):
        code = ErrorCode.SIMULATION_NON_CONVERGENCE
        retryable = False
    elif isinstance(exc, GainNotReachableError):
        code = ErrorCode.GAIN_NOT_REACHABLE
        retryable = False
    elif _contains_any(
        _message(exc),
        "unstable loop",
        "loop is unstable",
        "closed-loop is unstable",
        "non-positive margin",
        "not stable",
        "stable=false",
    ):
        code = ErrorCode.UNSTABLE_LOOP
        retryable = False
    elif _contains_any(_message(exc), "no inductor", "inductor candidate", "inductor solution"):
        code = ErrorCode.INDUCTOR_NO_SOLUTION
        retryable = False
    elif _contains_any(_message(exc), "no suitable core", "magnetics", "magnetic component"):
        code = ErrorCode.MAGNETICS_NO_SOLUTION
        retryable = False
    elif _contains_any(_message(exc), "no feasible", "core no solution", "no core solution"):
        code = ErrorCode.CORE_NO_SOLUTION
        retryable = False
    elif _contains_any(
        _message(exc), "device not found", "device missing", "mosfet not found", "diode not found"
    ) or isinstance(exc, KeyError) and _contains_any(_message(exc), "device", "mosfet", "diode"):
        code = ErrorCode.DEVICE_NOT_FOUND
        retryable = False
    elif isinstance(exc, KeyError):
        code = ErrorCode.RESOURCE_NOT_FOUND
        retryable = False
    elif isinstance(exc, ValueError):
        code = ErrorCode.INVALID_PARAMETERS
        retryable = False
    else:
        code = ErrorCode.INTERNAL_ERROR
        retryable = False

    return ErrorDetail(
        code=code,
        stage=stage,
        message=_MESSAGES[code],
        details=_exception_details(exc),
        retryable=retryable,
    )


__all__ = ["map_exception"]
