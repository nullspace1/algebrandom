from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from algebrandom.core.primitive import RandomVariable

class RandomVariableSymbol(sp.Symbol):
    
    def __new__(cls, rv: RandomVariable) -> sp.Symbol:
        object = sp.Symbol.__new__(cls, rv.name + "[" + str(id(rv)) + "]")
        object._rv = rv
        object._rv_type = rv.type
        return object
