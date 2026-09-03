"""Control Loop Designer API router.

Frontend -> FastAPI -> Control services -> llc_design kernel.  All responses
are schema-typed JSON; the browser never sees kernel objects.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.control_schema import (
    BodeResponse,
    CompensatorRequest,
    CompensatorResponse,
    ControllerConfigSchema,
    DigitalControllerRequest,
    DigitalControllerResponse,
    LoopGainRequest,
    LoopGainResponse,
    PlantModelResponse,
    PlantContext,
    ProtectionResponse,
)
from backend.services.control.compensator import design_compensator
from backend.services.control.digital_controller import (
    digital_controller_response,
    protection_supervisor,
)
from backend.services.control.loop_gain import bode_response, loop_gain_response
from backend.services.control.plant import plant_response

router = APIRouter(prefix="/api/control", tags=["control"])


def _guard(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - keep stack server-side
            raise HTTPException(status_code=500, detail=f"control calculation failed: {exc}") from exc
    return wrapper


@router.post("/plant", response_model=PlantModelResponse)
def control_plant(ctx: PlantContext) -> dict:
    return _guard(plant_response)(ctx)


@router.post("/compensator", response_model=CompensatorResponse)
def control_compensator(request: CompensatorRequest) -> dict:
    return _guard(design_compensator)(request)


@router.post("/loop_gain", response_model=LoopGainResponse)
def control_loop_gain(request: LoopGainRequest) -> dict:
    return _guard(loop_gain_response)(request.plant_context, request.controller)


@router.post("/bode", response_model=BodeResponse)
def control_bode(request: LoopGainRequest) -> dict:
    return _guard(bode_response)(request.plant_context, request.controller)


@router.post("/digital_controller", response_model=DigitalControllerResponse)
def control_digital_controller(request: DigitalControllerRequest) -> dict:
    return _guard(digital_controller_response)(request.plant_context, request.controller)


@router.post("/protection", response_model=ProtectionResponse)
def control_protection(ctx: PlantContext) -> dict:
    return _guard(protection_supervisor)(ctx)


# Re-exported so the frontend can discover the schema of PlantContext without
# duplicating it; kept out of the OpenAPI path intentionally.
__all__ = ["router", "ControllerConfigSchema", "PlantContext"]
