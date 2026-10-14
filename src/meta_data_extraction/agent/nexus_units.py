"""Exact, explicit temperature conversions without extraction or evidence claims.

Policy m05-temperature-v1 accepts a bounded plain decimal and an explicit unit
category/semantic role. It never parses a value or unit from raw_expression.
Celsius aliases are allowlisted and case-sensitive. Absolute temperatures use
T(K) = t(degC) + 273.15; differences use a scale factor of one, without offset.

The converted decimal is the exact arithmetic result, NOT a statement of new
measurement precision. Source decimal places and source/target reporting
resolution are retained separately. Reporting resolution is not uncertainty or
a significant-figure estimate. Scientific notation, uncertainties, ranges,
locale separators, and inferred units are intentionally unsupported. This helper
does not decide whether the supplied value is supported or physically plausible.
"""
from __future__ import annotations

from decimal import Decimal, Inexact, ROUND_HALF_EVEN, localcontext
import re
from typing import Literal

from pydantic import Field, model_validator

from .evidence import FrozenModel

CONVERSION_POLICY_VERSION = "m05-temperature-v1"
QuantityRole = Literal[
    "preparation_temperature", "acquisition_temperature", "temperature_difference",
]
Status = Literal["converted", "unavailable", "rejected"]
Reason = Literal[
    "explicit_conversion", "value_unavailable", "unit_unavailable",
    "raw_expression_unavailable", "unsupported_numeric_expression",
    "unit_category_mismatch", "unsupported_unit", "incompatible_units",
]
Operation = Literal[
    "identity", "celsius_to_kelvin", "kelvin_to_celsius", "temperature_difference",
]
_DECIMAL = re.compile(r"[+-]?[0-9]{1,30}(?:\.[0-9]{1,30})?", re.ASCII)
_TEMPERATURE_UNITS = {"K": "kelvin", "C": "celsius", "degC": "celsius", "°C": "celsius"}
# These are recognized solely to produce an explicit incompatible-unit state.
_OTHER_DIMENSIONS = frozenset({"m", "cm", "mm", "nm", "kg", "g", "s", "Pa", "eV"})


class QuantityRequest(FrozenModel):
    raw_expression: str | None = Field(default=None, min_length=1, max_length=2000, strict=True)
    value_text: str | None = Field(default=None, min_length=1, max_length=100, strict=True)
    from_unit: str | None = Field(default=None, min_length=1, max_length=32, strict=True)
    to_unit: str = Field(min_length=1, max_length=32, strict=True)
    role: QuantityRole
    unit_category: str = Field(min_length=1, max_length=100, strict=True)


def _evaluate(request: QuantityRequest) -> dict:
    result = {
        "status": "rejected", "reason": "unsupported_numeric_expression",
        "converted_value_text": None, "source_decimal_places": None,
        "source_resolution_text": None, "converted_resolution_text": None,
        "operation": None,
    }
    if request.value_text is None:
        return {**result, "status": "unavailable", "reason": "value_unavailable"}
    if request.from_unit is None:
        return {**result, "status": "unavailable", "reason": "unit_unavailable"}
    if request.raw_expression is None:
        return {**result, "reason": "raw_expression_unavailable"}
    if request.unit_category != "NX_TEMPERATURE":
        return {**result, "reason": "unit_category_mismatch"}
    if request.from_unit in _OTHER_DIMENSIONS or request.to_unit in _OTHER_DIMENSIONS:
        return {**result, "reason": "incompatible_units"}
    if request.from_unit not in _TEMPERATURE_UNITS or request.to_unit not in _TEMPERATURE_UNITS:
        return {**result, "reason": "unsupported_unit"}
    if _DECIMAL.fullmatch(request.value_text) is None:
        return result
    decimal_places = len(request.value_text.partition(".")[2])
    source_kind = _TEMPERATURE_UNITS[request.from_unit]
    target_kind = _TEMPERATURE_UNITS[request.to_unit]
    # Bound input to <= 60 decimal digits, then give exact arithmetic ample room.
    # Neither the caller's Decimal precision nor its rounding policy is inherited.
    with localcontext() as context:
        context.prec = 128
        context.rounding = ROUND_HALF_EVEN
        context.Emin = -999999
        context.Emax = 999999
        context.traps[Inexact] = True
        value = Decimal(request.value_text)
        resolution = format(Decimal(1).scaleb(-decimal_places), "f")
        if request.role == "temperature_difference":
            operation = "temperature_difference"
            converted = value
        elif source_kind == target_kind:
            operation = "identity"
            converted = value
        elif source_kind == "celsius":
            operation = "celsius_to_kelvin"
            converted = value + Decimal("273.15")
        else:
            operation = "kelvin_to_celsius"
            converted = value - Decimal("273.15")
        converted_text = format(converted, "f")
    return {
        "status": "converted", "reason": "explicit_conversion",
        "converted_value_text": converted_text,
        "source_decimal_places": decimal_places,
        "source_resolution_text": resolution, "converted_resolution_text": resolution,
        "operation": operation,
    }


class ConversionResult(FrozenModel):
    policy_version: Literal["m05-temperature-v1"] = CONVERSION_POLICY_VERSION
    request: QuantityRequest
    status: Status
    reason: Reason
    converted_value_text: str | None = Field(default=None, max_length=100, strict=True)
    source_decimal_places: int | None = Field(default=None, ge=0, le=30, strict=True)
    source_resolution_text: str | None = Field(default=None, max_length=32, strict=True)
    converted_resolution_text: str | None = Field(default=None, max_length=32, strict=True)
    operation: Operation | None = None
    support_status: Literal["not_assessed"] = "not_assessed"
    approval_status: Literal["unreviewed"] = "unreviewed"

    @model_validator(mode="after")
    def _integrity(self) -> ConversionResult:
        actual = self.model_dump(exclude={"policy_version", "request", "support_status", "approval_status"})
        if actual != _evaluate(self.request):
            raise ValueError("Conversion does not match the explicit versioned policy")
        return self


def convert_quantity(raw_expression: str | None, value_text: str | None,
                     from_unit: str | None, to_unit: str, *, role: QuantityRole,
                     unit_category: str) -> ConversionResult:
    """Compute only the explicitly supplied quantity; preserve its semantic role.

    Ill-typed requests raise Pydantic ValidationError. A missing quantity/unit or
    unsupported scientific expression returns an explicit non-converted state.
    Caller must separately verify evidence, extraction, role, and mapping review.
    """
    request = QuantityRequest(raw_expression=raw_expression, value_text=value_text,
                              from_unit=from_unit, to_unit=to_unit, role=role,
                              unit_category=unit_category)
    return ConversionResult(request=request, **_evaluate(request))


def verify_conversion(result: ConversionResult) -> ConversionResult:
    """Revalidate a retained conversion, including models made with model_copy."""
    return ConversionResult.model_validate(result)
