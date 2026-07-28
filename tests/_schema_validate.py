"""Dependency-free JSON-Schema validator for the RayD contract schemas.

Implements only the JSON-Schema draft 2020-12 constructs used by
``contracts/public_api.schema.json`` and
``contracts/path_exchange.schema.json``. Any schema keyword or
``type`` value it does not implement raises ``NotImplementedError`` so that
schema growth cannot silently skip validation.
"""

from __future__ import annotations

from typing import Any, Callable


class SchemaValidationError(Exception):
    """Raised when an instance violates its schema."""


_ANNOTATION_KEYWORDS = {"$schema", "$id", "title"}
_SUPPORTED_KEYWORDS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "uniqueItems",
    "minLength",
    "minimum",
    "const",
}

_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "string": lambda value: isinstance(value, str),
    "boolean": lambda value: isinstance(value, bool),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
}


def validate(instance: Any, schema: dict[str, Any], path: str = "") -> None:
    where = path or "<root>"
    unsupported = set(schema) - _SUPPORTED_KEYWORDS - _ANNOTATION_KEYWORDS
    if unsupported:
        raise NotImplementedError(f"unimplemented schema keyword(s) at {where}: {sorted(unsupported)}")

    if "type" in schema:
        check = _TYPE_CHECKS.get(schema["type"])
        if check is None:
            raise NotImplementedError(f"unimplemented schema type at {where}: {schema['type']!r}")
        if not check(instance):
            raise SchemaValidationError(f"{where}: expected type {schema['type']!r}")

    if "const" in schema and instance != schema["const"]:
        raise SchemaValidationError(f"{where}: {instance!r} does not equal const {schema['const']!r}")

    if "enum" in schema and instance not in schema["enum"]:
        raise SchemaValidationError(f"{where}: {instance!r} is not one of {schema['enum']}")

    if "minimum" in schema and instance < schema["minimum"]:
        raise SchemaValidationError(f"{where}: {instance!r} is below minimum {schema['minimum']}")

    if "minLength" in schema and len(instance) < schema["minLength"]:
        raise SchemaValidationError(f"{where}: length {len(instance)} is below minLength {schema['minLength']}")

    if isinstance(instance, dict):
        _validate_object(instance, schema, path)
    elif isinstance(instance, list):
        _validate_array(instance, schema, path)


def _validate_object(instance: dict[str, Any], schema: dict[str, Any], path: str) -> None:
    where = path or "<root>"
    for key in schema.get("required", []):
        if key not in instance:
            raise SchemaValidationError(f"{where}: missing required property {key!r}")

    properties = schema.get("properties", {})
    additional = schema.get("additionalProperties", True)
    for key, value in instance.items():
        child = f"{path}.{key}" if path else key
        if key in properties:
            validate(value, properties[key], child)
        elif additional is False:
            raise SchemaValidationError(f"{child}: additional property is not allowed")
        elif isinstance(additional, dict):
            validate(value, additional, child)


def _validate_array(instance: list[Any], schema: dict[str, Any], path: str) -> None:
    where = path or "<root>"
    if "items" in schema:
        for index, element in enumerate(instance):
            validate(element, schema["items"], f"{path}[{index}]")
    if schema.get("uniqueItems"):
        seen: list[Any] = []
        for element in instance:
            if element in seen:
                raise SchemaValidationError(f"{where}: duplicate item {element!r}")
            seen.append(element)
