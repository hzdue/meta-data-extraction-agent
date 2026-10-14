"""Invented quantity checks only: conversion is neither evidence nor approval."""
from decimal import Decimal, ROUND_FLOOR, localcontext

import pytest
from pydantic import ValidationError

from meta_data_extraction.agent.nexus_units import (
    CONVERSION_POLICY_VERSION, ConversionResult, QuantityRequest,
    convert_quantity, verify_conversion,
)


def convert(value="25.0", from_unit="C", to_unit="K", *,
            role="acquisition_temperature", raw_expression="reported 25.0 C",
            unit_category="NX_TEMPERATURE"):
    return convert_quantity(raw_expression, value, from_unit, to_unit,
                            role=role, unit_category=unit_category)


@pytest.mark.parametrize("role", ["preparation_temperature", "acquisition_temperature"])
@pytest.mark.parametrize("value,from_unit,to_unit,expected,operation", [
    ("25", "C", "K", "298.15", "celsius_to_kelvin"),
    ("-273.15", "°C", "K", "0.00", "celsius_to_kelvin"),
    ("0.00", "K", "degC", "-273.15", "kelvin_to_celsius"),
    ("298.150", "K", "C", "25.000", "kelvin_to_celsius"),
    ("25.00", "°C", "degC", "25.00", "identity"),
    ("300.0", "K", "K", "300.0", "identity"),
])
def test_explicit_absolute_temperature_conversion(role, value, from_unit, to_unit,
                                                  expected, operation):
    result = convert(value, from_unit, to_unit, role=role)
    assert result.status == "converted"
    assert result.converted_value_text == expected
    assert result.operation == operation
    assert result.request.role == role
    assert result.support_status == "not_assessed"
    assert result.approval_status == "unreviewed"
    assert result.policy_version == CONVERSION_POLICY_VERSION
    assert verify_conversion(result) == result


@pytest.mark.parametrize("value,from_unit,to_unit", [
    ("25.0", "C", "K"), ("25.0", "K", "C"),
    ("-4.00", "degC", "K"), ("0", "K", "°C"), ("2.5", "K", "K"),
])
def test_temperature_differences_have_no_offset(value, from_unit, to_unit):
    result = convert(value, from_unit, to_unit, role="temperature_difference")
    assert result.converted_value_text == value
    assert result.operation == "temperature_difference"


@pytest.mark.parametrize("value,places,resolution,exact", [
    ("25", 0, "1", "298.15"),
    ("25.0", 1, "0.1", "298.15"),
    ("25.00", 2, "0.01", "298.15"),
    ("25.000", 3, "0.001", "298.150"),
    ("+00025.0000", 4, "0.0001", "298.1500"),
])
def test_raw_precision_is_preserved_separately_from_exact_arithmetic(value, places,
                                                                     resolution, exact):
    raw = f"The fictional acquisition was at {value} °C."
    result = convert(value, "°C", "K", raw_expression=raw)
    assert result.request.raw_expression == raw
    assert result.request.value_text == value
    assert result.request.from_unit == "°C"
    assert result.source_decimal_places == places
    assert result.source_resolution_text == resolution
    assert result.converted_resolution_text == resolution
    assert result.converted_value_text == exact
    # Decimal places describe reporting granularity; the policy never invents
    # an uncertainty estimate or treats offset digits as measured precision.
    assert "uncertainty" not in ConversionResult.model_fields


def test_same_quantity_does_not_transfer_preparation_to_acquisition_role():
    preparation = convert(role="preparation_temperature")
    acquisition = convert(role="acquisition_temperature")
    assert preparation.converted_value_text == acquisition.converted_value_text
    assert preparation.request.role != acquisition.request.role
    assert preparation != acquisition


def test_ambiguous_raw_expression_is_preserved_not_parsed_as_evidence():
    result = convert("25.0", raw_expression="Preparation was 900 C; acquisition was 25.0 C.")
    assert result.request.raw_expression == "Preparation was 900 C; acquisition was 25.0 C."
    assert result.converted_value_text == "298.15"
    assert result.support_status == "not_assessed"


@pytest.mark.parametrize("value", [
    "1e3", "1E+3", "NaN", "Infinity", "-inf", "25 ± 1", "25(1)",
    "20-25", "20 to 25", "1,000", "25,0", " 25", "25 ", "25\n",
    ".5", "5.", "１.５", "25 C", "None", "9" * 31, "0." + "1" * 31,
])
def test_unsupported_numeric_expression_is_explicit(value):
    result = convert(value)
    assert result.status == "rejected"
    assert result.reason == "unsupported_numeric_expression"
    assert result.converted_value_text is None
    assert result.source_decimal_places is None
    assert result.operation is None
    assert verify_conversion(result) == result


@pytest.mark.parametrize("value,from_unit,raw,reason", [
    (None, "C", "not reported", "value_unavailable"),
    (None, None, None, "value_unavailable"),
    ("25", None, "25", "unit_unavailable"),
])
def test_missing_values_stay_unavailable(value, from_unit, raw, reason):
    result = convert(value, from_unit, raw_expression=raw)
    assert result.status == "unavailable"
    assert result.reason == reason
    assert result.converted_value_text is None
    assert result.converted_resolution_text is None


def test_value_without_original_expression_is_rejected():
    result = convert(raw_expression=None)
    assert result.status == "rejected"
    assert result.reason == "raw_expression_unavailable"


@pytest.mark.parametrize("category", ["NX_LENGTH", "NX_ANY", "temperature", "NX_MASS"])
def test_explicit_unit_category_must_match(category):
    result = convert(unit_category=category)
    assert result.status == "rejected"
    assert result.reason == "unit_category_mismatch"


@pytest.mark.parametrize("source,target,reason", [
    ("m", "K", "incompatible_units"), ("C", "s", "incompatible_units"),
    ("g", "K", "incompatible_units"), ("degF", "K", "unsupported_unit"),
    ("K", "°F", "unsupported_unit"), ("k", "K", "unsupported_unit"),
    ("celsius", "K", "unsupported_unit"), ("C ", "K", "unsupported_unit"),
])
def test_no_inferred_or_incompatible_units(source, target, reason):
    result = convert("25", source, target)
    assert result.status == "rejected"
    assert result.reason == reason
    assert result.converted_value_text is None


@pytest.mark.parametrize("value", [25, 25.0, Decimal("25"), True, b"25", [], {}])
def test_non_string_values_are_not_silently_coerced(value):
    with pytest.raises(ValidationError):
        convert(value)


@pytest.mark.parametrize("kwargs", [
    {"value_text": ""}, {"raw_expression": ""}, {"raw_expression": 5},
    {"from_unit": 5}, {"to_unit": 5}, {"unit_category": 1},
    {"role": "temperature"}, {"role": None}, {"value_text": "0" * 101},
    {"raw_expression": "x" * 2001}, {"target_role": "acquisition_temperature"},
])
def test_typed_bounded_request_rejects_invalid_inputs(kwargs):
    supplied = dict(raw_expression="25 C", value_text="25", from_unit="C", to_unit="K",
                    role="preparation_temperature", unit_category="NX_TEMPERATURE")
    with pytest.raises(ValidationError):
        QuantityRequest(**(supplied | kwargs))


def test_exact_arithmetic_is_independent_of_callers_decimal_context():
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_FLOOR
        context.Emax = 2
        context.Emin = -2
        result = convert("123456789012345678901234567890.123456789012345678901234567890")
        assert result.converted_value_text == "123456789012345678901234568163.273456789012345678901234567890"
        assert result.source_decimal_places == 30
        zero = convert("273.15", "K", "C")
        assert zero.converted_value_text == "0.00"


def test_signed_zero_and_original_lexical_form_are_retained():
    result = convert("-000.00", "K", "K", raw_expression="-000.00 K")
    assert result.converted_value_text == "-0.00"
    assert result.request.value_text == "-000.00"
    assert result.request.raw_expression == "-000.00 K"


@pytest.mark.parametrize("field,replacement", [
    ("converted_value_text", "999.99"), ("source_decimal_places", 2),
    ("source_resolution_text", "0.01"), ("converted_resolution_text", "0.01"),
    ("operation", "identity"), ("status", "unavailable"),
    ("reason", "value_unavailable"), ("policy_version", "future-policy"),
    ("support_status", "supported"), ("approval_status", "approved"),
])
def test_serialized_result_cannot_misrepresent_the_policy(field, replacement):
    payload = convert().model_dump(mode="json")
    payload[field] = replacement
    with pytest.raises(ValidationError):
        ConversionResult.model_validate(payload)


def test_unavailable_value_cannot_be_filled_by_required_field_status():
    result = convert(None, None, raw_expression=None)
    payload = result.model_dump(mode="json")
    payload.update(status="converted", reason="explicit_conversion",
                   converted_value_text="273.15", source_decimal_places=0,
                   source_resolution_text="1", converted_resolution_text="1",
                   operation="celsius_to_kelvin")
    with pytest.raises(ValidationError):
        ConversionResult.model_validate(payload)


def test_copied_model_tampering_is_rejected_and_models_are_immutable():
    result = convert()
    with pytest.raises(ValidationError):
        result.converted_value_text = "0"
    with pytest.raises(ValidationError):
        result.request.role = "preparation_temperature"
    with pytest.raises(ValidationError):
        verify_conversion(result.model_copy(update={"converted_value_text": "0"}))
    forged_request = result.request.model_copy(update={"value_text": 25})
    with pytest.raises(ValidationError):
        verify_conversion(result.model_copy(update={"request": forged_request}))


def test_request_unpack_and_json_round_trip():
    request = QuantityRequest(raw_expression="Synthetic 20.00 °C", value_text="20.00",
                              from_unit="°C", to_unit="K", role="acquisition_temperature",
                              unit_category="NX_TEMPERATURE")
    result = convert_quantity(**request.model_dump())
    assert ConversionResult.model_validate_json(result.model_dump_json()) == result
    assert result.request == request
