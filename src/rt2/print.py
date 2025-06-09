from __future__ import annotations

from collections.abc import Iterable


def fieldFormat(field: str, value: any, unit: str = ""):
    if isinstance(value, Iterable) and not isinstance(value, str):
        value = str(value.__repr__())
    if unit:
        return "{field: <20}: {value: >24} ({unit})\n".format(field=field, value=value, unit=unit)
    else:
        return "{field: <20}: {value: >24}\n".format(field=field, value=value)


def nameFormat(name: str, max_length: int = 14):
    return "  [ {name: >{max_length}} ]\n".format(name=name[:max_length], max_length=max_length)
