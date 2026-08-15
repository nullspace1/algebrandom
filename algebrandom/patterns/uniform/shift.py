from algebrandom.core.pattern import Pattern
import sympy as sp

from algebrandom.core.primitive import RandomVariable
from algebrandom.instances.uniform import Uniform


class UniformShiftPattern(Pattern):

    def match(self, expr: sp.Expr) -> tuple[sp.Expr, RandomVariable] | None:
        shift = self.wc_constant("uniform_shift")
        rv_wildcard = self.wc_random_variable("shifted_uniform", Uniform)
        pattern = shift + rv_wildcard

        matched = self.get_matched_pattern(expr, pattern)
        if matched is None:
            return None

        constant = matched.get_constant(shift)
        if constant == 0:
            return None

        rv = matched.get_rv(rv_wildcard, Uniform)
        shifted = Uniform(
            rv.min + constant,
            rv.max + constant,
            rv.name + "+" + str(constant),
            expression=matched.pattern,
        )
        return matched.pattern, shifted
